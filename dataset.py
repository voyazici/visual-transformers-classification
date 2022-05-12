import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

def augmenter(image):
    return transforms.RandomHorizontalFlip(p=0.5)(
        transforms.ColorJitter(contrast=0.25)(
            transforms.RandomAffine(
                0, translate=(0.03, 0.03))(image)))


def process_image(image):
    means = [0.485, 0.456, 0.406]
    inv_stds = [1/0.229, 1/0.224, 1/0.225]

    image = Image.fromarray(image)
    image = transforms.ToTensor()(image)
    for channel, mean, inv_std in zip(image, means, inv_stds):
        channel.sub_(mean).mul_(inv_std)
    return image

categories = ['airplane', 'apple', 'backpack', 'banana', 'baseball bat',
              'baseball glove', 'bear', 'bed', 'bench', 'bicycle', 'bird',
              'boat', 'book', 'bottle', 'bowl', 'broccoli', 'bus', 'cake',
              'car', 'carrot', 'cat', 'cell phone', 'chair', 'clock', 'couch',
              'cow', 'cup', 'dining table', 'dog', 'donut', 'elephant',
              'fire hydrant', 'fork', 'frisbee', 'giraffe', 'hair drier',
              'handbag', 'horse', 'hot dog', 'keyboard', 'kite', 'knife',
              'laptop', 'microwave', 'motorcycle', 'mouse', 'orange', 'oven',
              'parking meter', 'person', 'pizza', 'potted plant', 'refrigerator',
              'remote', 'sandwich', 'scissors', 'sheep', 'sink', 'skateboard',
              'skis', 'snowboard', 'spoon', 'sports ball', 'stop sign', 'suitcase',
              'surfboard', 'teddy bear', 'tennis racket', 'tie', 'toaster', 'toilet',
              'toothbrush', 'traffic light', 'train', 'truck', 'tv', 'umbrella', 'vase',
              'wine glass', 'zebra']
category_dict_sequential = dict((category, count) for count, category in enumerate(categories))
category_dict_sequential['<empty>'] = len(categories)
category_dict_sequential_inv = dict((value, key)
                                    for key, value in category_dict_sequential.items())

class COCOMultiLabel(Dataset):
    def __init__(self, train, classification, image_path, image_size):
        super(COCOMultiLabel, self).__init__()
        self.train = train
        if self.train == True:
            self.coco_json = json.load(open('coco_train.json', 'r'))
            self.image_path = image_path + '/train2014/'
        elif self.train == False:
            self.coco_json = json.load(open('coco_val.json', 'r'))
            self.image_path = image_path + '/val2014/'

        category_counts = [len(value['categories']) for value in self.coco_json.values()]
        self.max_length = max(category_counts)
        self.image_size = image_size

        self.classification = classification
        self.fns = list(self.coco_json.keys())
        self.image_size = image_size
  
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.fns)

    def __getitem__(self, idx):
        json_key = self.fns[idx]
        categories_batch = self.coco_json[json_key]['categories']
        image_fn = self.image_path + json_key

        image = Image.open(image_fn)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        if self.train:
            try:
                image = augmenter(image)
            except IOError:
                print("augmentation error")
        try:
            image = self.transform(image)
        except IOError:
            print("transformer error")
            return None

        labels = []
        for category in categories_batch:
            labels.append(category_dict_sequential[category])
        for _ in range(self.max_length - len(categories_batch)):
            labels.append(category_dict_sequential['<empty>'])

        labels = torch.LongTensor(labels)
        label_number = len(categories_batch)
        return image, labels, label_number

