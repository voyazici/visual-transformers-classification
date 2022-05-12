import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataloader import default_collate
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, average_precision_score
from tqdm import tqdm
import os
from tensorboardX import SummaryWriter
from dataset import COCOMultiLabel, category_dict_sequential, category_dict_sequential_inv
from munkres import Munkres
from model import NetEncoderDecoder

m = Munkres()

def convert_to_array(scores, targets, target_lengths):
    scores = scores.data.cpu().numpy()
    targets = targets.data.cpu().numpy()
    number_class = 80
    N = scores.shape[0]
    preds = np.zeros((N, number_class), dtype=np.float32)
    labels = np.zeros((N, number_class), dtype=np.float32)
    number_time_steps = scores.shape[1]
    for i in range(N):
        preds_image = []
        for step_t in range(number_time_steps):
            step_pred = np.argmax(scores[i][step_t])
            if category_dict_sequential_inv[step_pred] == '<empty>':
                continue
            preds_image.append(step_pred)
        preds[i, preds_image] = 1
        labels_image = targets[i][0:target_lengths[i]]
        labels[i, labels_image] = 1
    return preds, labels

def order_the_targets(scores, targets, label_lengths_sorted):
    device = targets.device
    scores_tensor = scores.clone()
    scores = scores.data.cpu().numpy()
    time_steps = scores.shape[1]
    batch_size = scores.shape[0]
    targets = targets.data.cpu().numpy()
    targets_new = category_dict_sequential['<empty>'] * np.ones((batch_size, time_steps))
    N = scores.shape[0]
    indexes = np.argmax(scores, axis=2)

    losses_list = []
    for i in range(N):
        common_indexes = set(targets[i][:label_lengths_sorted[i]]).intersection(set(indexes[i]))
        diff_indexes = set(targets[i][:label_lengths_sorted[i]]).difference(set(indexes[i]))
        diff_indexes = list(diff_indexes)
        common_locs = {idx: (indexes[i] == idx).nonzero()[0].tolist()
                       for idx in common_indexes}
        for idx, locs in common_locs.items():
            targets_new[i][locs] = idx
        empty_locs = (targets_new[i] == category_dict_sequential['<empty>']).nonzero()[0].tolist()
        needed_loc = len(diff_indexes) - len(empty_locs)
        available_locs = []
        if diff_indexes:
            available_locs = empty_locs[:]
            for idx, locs in common_locs.items():
                if len(locs) > 1:
                    max_loc = locs[scores[i, locs, idx].argmax()]
                    locs_copy = locs[:]
                    locs_copy.remove(max_loc)
                    available_locs.extend(locs_copy)
            losses = -F.log_softmax(scores_tensor[i, available_locs][:, diff_indexes], dim=0).data.cpu().numpy().transpose(1, 0)
            m_indexes = m.compute(losses)
            for m_index, m_loc in m_indexes:
                targets_new[i, available_locs[m_loc]] = diff_indexes[m_index]
        else:
            losses_list.append(0)
    targets_new = torch.LongTensor(targets_new).to(device)
    return targets_new

def adjust_learning_rate(optimizer, shrink_factor):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr']))

def my_collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return default_collate(batch)

def mix_up_images(data):
    # mix up the first and second halves
    # keep the second half same
    images = torch.zeros_like(data)
    batch_size = data.shape[0]
    half = batch_size //2
    images[half:] = data[half:]
    images[:half] = (data[:half] + data[half:]) / 2
    return images

def mix_up_labels(target, target_lengths):
    # mix up the first and second halves
    # keep the second half same
    num_classes = 80
    temp = target_lengths.clone()
    batch_size = target.shape[0]
    half = batch_size // 2
    temp[:half] += temp[half:]
    new_target = num_classes * torch.ones(batch_size, temp.max()).type(torch.LongTensor).to(target.device)
    new_target_lengths = []
    for i in range(half):
        old_labels = target[i][:target_lengths[i]].tolist()
        new_labels = target[i + half][:target_lengths[i + half]].tolist()
        all_labels = list(set(old_labels + new_labels))
        new_target[i][:len(all_labels)] = torch.LongTensor(all_labels).to(new_target.device)
        new_target_lengths.append(len(all_labels))
    for i in range(half, batch_size):
        new_target[i][:target_lengths[i]] = target[i][:target_lengths[i]].clone()
        new_target_lengths.append(target_lengths[i].item())
    new_target_lengths = torch.LongTensor(new_target_lengths).to(target_lengths.device)
    return (new_target, new_target_lengths)

def train(args, model, device, train_loader, encoder_optim, decoder_optim,
          epoch, writer):
    model.train()
    if isinstance(model, nn.DataParallel):
        model.module.freeze("bn")
    else:
        model.freeze("bn")
    for batch_idx, batch in enumerate(train_loader):
        if batch is None:
            continue
        data, target, target_lengths = batch
        data, target = data.to(device), target.to(device)
        encoder_optim.zero_grad()
        decoder_optim.zero_grad()

        if args.mix_up and data.shape[0] % 2 == 0:
            data = mix_up_images(data)
            target, target_lengths = mix_up_labels(target, target_lengths)

        decoder_output = model(data)

        target_new = order_the_targets(decoder_output, target, target_lengths)
        loss = F.cross_entropy(decoder_output.permute(0, 2, 1), target_new)
        loss.backward()

        encoder_optim.step()
        decoder_optim.step()
        if batch_idx % args.log_interval == 0:
            preds, labels = convert_to_array(decoder_output, target, target_lengths)
            _, _, f1, _ = precision_recall_fscore_support(preds, labels, average='micro')
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tF1: {:.2f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(), 100*f1))
            writer.add_scalar('loss', loss.item(), epoch * len(train_loader) + batch_idx)
            writer.add_scalar('encoder_lr', encoder_optim.param_groups[0]['lr'],
                              epoch * len(train_loader) + batch_idx)
            writer.add_scalar('decoder_lr', decoder_optim.param_groups[0]['lr'],
                              epoch * len(train_loader) + batch_idx)
            writer.add_scalar('train_f1', f1 * 100, epoch * len(train_loader) + batch_idx)

def test(args, model, device, test_loader, threshold):
    model.eval()
    preds_all = None
    labels_all = None
    scores_all = None
    with torch.no_grad():
        for batch in tqdm(test_loader, total=len(test_loader)):
            if batch is None:
                continue
            data, target, target_lengths = batch
            data = data.to(device)
            decoder_output = model(data)
            scores = F.softmax(decoder_output, dim=2).data.cpu().numpy()

            preds, labels = convert_to_array(decoder_output, target, target_lengths)
            if labels_all is None:
                preds_all = preds
                labels_all = labels
                scores_all = scores
            else:
                preds_all = np.concatenate((preds_all, preds), axis=0)
                labels_all = np.concatenate((labels_all, labels), axis=0)
                scores_all = np.concatenate((scores_all, scores), axis=0)

    results = {'micro': None, 'macro': None}
    prec, recall, _, _ = precision_recall_fscore_support(labels_all,
                                                         preds_all,
                                                         average='macro')
    f1 = 2 * prec * recall / (prec + recall)
    results['macro'] = {'precision': prec, 'recall': recall, 'f1': f1}
    print('\nMACRO prec: {:.2f}, recall: {:.2f}, f1: {:.2f}\n'.format(
        100*prec, 100*recall, 100*f1))
    prec, recall, f1, _ = precision_recall_fscore_support(labels_all,
                                                          preds_all,
                                                          average='micro')
    results['micro'] = {'precision': prec, 'recall': recall, 'f1': f1}
    print('\nMICRO prec: {:.2f}, recall: {:.2f}, f1: {:.2f}\n'.format(
        100*prec, 100*recall, 100*f1))

    scores_max = scores_all.max(1)[:, :80]
    map_cats = []
    for j in range(80):
        map_cat = average_precision_score(labels_all[:, j], scores_max[:, j])
        map_cats.append(map_cat)
        print('%s,%.1f' % (category_dict_sequential_inv[j], map_cat * 100))
    print('mAP %.1f' % (np.mean(map_cats) * 100))
    print(','.join([('%.1f' % (x*100)) for x in map_cats]))
    mAP = np.mean(map_cats)
    results['mAP'] = mAP

    return results

def prepare_optimizer(optim_params, model_params):
    opt = optim_params['opt']
    assert opt in ['adam', 'sgd']
    lr = optim_params['lr']
    print(opt, lr)
    if opt == 'adam':
        optimizer = optim.Adam(model_params, lr=lr,
                               weight_decay=optim_params.get('weight_decay', 0.0))
    else:
        optimizer = optim.SGD(model_params, lr=lr,
                              weight_decay=optim_params.get('weight_decay', 0.0),
                              momentum=optim_params.get('momentum', 0.9))
    return optimizer

def main():
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('-batch_size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('-epochs', type=int, default=40, metavar='N',
                        help='number of epochs to train (default: 40)')
    parser.add_argument('-log_interval', type=int, default=50, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('-threshold', type=float, default=0.5,
                        help='threshold for the evaluation (default: 0.5)')
    parser.add_argument('-image_path', help='path for the training and validation folders')
    parser.add_argument('-num_workers', type=int, default=4)
    parser.add_argument('-snapshot', default=None)
    parser.add_argument('-resume', type=int, default=None)
    parser.add_argument('-test_model', action='store_true')
    parser.add_argument('-save_path')
    parser.add_argument('-input_size', default=None, type=int)
    parser.add_argument('-num_encoder_layers', type=int, default=1)
    parser.add_argument('-num_decoder_layers', type=int, default=2)
    parser.add_argument('-num_att_heads', type=int, default=1)
    parser.add_argument('-hidden_size', type=int, default=512)
    parser.add_argument('-weight_decay', type=float, default=0.0)
    parser.add_argument('-optim_params', type=str,
                        default="[{'opt': 'sgd', 'lr': 1e-3}, {'opt': 'adam', 'lr': 1e-4}]")
    parser.add_argument('-num_queries', type=int, default=25)
    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-mix_up', action='store_true')

    args = parser.parse_args()

    assert args.image_path is not None

    device = "cuda"
    save_path = args.save_path
    if not args.test_model:
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        log_path = os.path.join(save_path, 'logs')
        if not os.path.isdir(log_path):
            os.mkdir(log_path)
        else:
            if args.snapshot == None:
                raise ValueError('Delete the log path manually %s' % log_path)
        writer = SummaryWriter(log_dir=log_path)

    default_input_size = 288
    train_dataset = COCOMultiLabel(train=True,
                                   classification=False,
                                   image_path=args.image_path,
                                   image_size=args.input_size if args.input_size else default_input_size)
    test_dataset = COCOMultiLabel(train=False,
                                  classification=False,
                                  image_path=args.image_path,
                                  image_size=args.input_size if args.input_size else default_input_size)
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              num_workers=args.num_workers,
                              pin_memory=True,
                              shuffle=True,
                              drop_last=False,
                              collate_fn=my_collate)
    test_loader = DataLoader(test_dataset,
                             batch_size=args.batch_size,
                             num_workers=args.num_workers,
                             pin_memory=True,
                             shuffle=False,
                             drop_last=False,
                             collate_fn=my_collate)
    model = NetEncoderDecoder(num_encoder_layers=args.num_encoder_layers,
                              num_decoder_layers=args.num_decoder_layers,
                              num_att_heads=args.num_att_heads,
                              hidden_size=args.hidden_size,
                              num_queries=args.num_queries,
                              dropout=args.dropout).to(device)
    if args.snapshot:
        weights = torch.load(args.snapshot)
        weights = {key.replace('module.', ''): value for key, value in weights.items()}
        model.load_state_dict(weights)
        if args.test_model == False:
            if args.resume is not None:
                resume = args.resume
                print("Resuming at", resume)
            else:
                print("Training from scratch")
                resume = 1
        else:
            resume = 0
    else:
        resume = 1
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    highest_f1 = 0
    epochs_without_imp = 0
    all_epochs_without_imp = 0
    backbone_layers = {'conv1', 'bn1', 'layer1', 'layer2', 'layer3', 'layer4'}
    if not args.test_model:
        encoder_params = []
        decoder_params = []
        for param_name, param in model.named_parameters():
            if any([True if x in param_name else False for x in backbone_layers]):
                encoder_params.append({'params': param})
            else:
                decoder_params.append({'params': param})
        encoder_optim_params, decoder_optim_params = eval(args.optim_params)
        encoder_optim = prepare_optimizer(encoder_optim_params, encoder_params)
        decoder_optim = prepare_optimizer(decoder_optim_params, decoder_params) 

    for epoch in range(resume, args.epochs + 1):
        if args.test_model == False:
            train(args, model, device, train_loader,
                  encoder_optim, decoder_optim, epoch, writer)
            results = test(args, model, device, test_loader, args.threshold)
            for x in ['micro', 'macro']:
                for y in ['precision', 'recall', 'f1']:
                    writer.add_scalar('%s_%s' % (x, y), results[x][y] * 100, epoch)
            writer.add_scalar('mAP', results['mAP'] * 100, epoch)
            torch.save(model.state_dict(), args.save_path + "/checkpoint.pt")
            f1 = (results['macro']['f1'] + results['micro']['f1']) / 2
            if f1 > highest_f1:
                torch.save(model.state_dict(), args.save_path + "/BEST_checkpoint.pt")
                highest_f1 = f1
                epochs_without_imp = 0
            else:
                epochs_without_imp += 1
                if epochs_without_imp == 3:
                    adjust_learning_rate(encoder_optim, 0.1)
                    adjust_learning_rate(decoder_optim, 0.1)
                    epochs_without_imp = 0
                    all_epochs_without_imp += 1
                if all_epochs_without_imp == 4:
                    break
        else:
            results = test(args, model, device, test_loader, args.threshold)
            break

if __name__ == '__main__':
    main()
