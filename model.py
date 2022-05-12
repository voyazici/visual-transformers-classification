import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise NotImplementedError

class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, object_queries, encoder_src):
        for layer in self.layers:
            object_queries, attn = layer(object_queries, encoder_src)

        return object_queries.unsqueeze(0), attn

class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead=1, dim_feedforward=2048, dropout=0.1,
                 activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, object_queries, encoder_src):
        q = k = object_queries
        tgt2 = self.self_attn(q, k, value=object_queries)[0]
        tgt = object_queries + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2, attn = self.multihead_attn(query=tgt,
                                         key=encoder_src,
                                         value=encoder_src)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, attn

class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, src):
        output = src
        for layer in self.layers:
            output, _ = layer(output)
        return output

class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead=1, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def forward(self, src):
        q = k = src
        src2, attn = self.self_attn(q, k, value=src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attn

class NetEncoderDecoder(nn.Module):
    def __init__(self, num_encoder_layers=None, num_decoder_layers=None, hidden_size=2048,
                 dropout=0.1, num_att_heads=1, num_queries=20):
        super(NetEncoderDecoder, self).__init__()

        resnet = torchvision.models.resnet101(True)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        num_classes = 80 + 1
        self.input_proj = None
        if hidden_size != 2048:
            self.input_proj = nn.Conv2d(2048, hidden_size, kernel_size=1)
        encoder_layer = TransformerEncoderLayer(hidden_size,
                                                nhead=num_att_heads)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_encoder_layers)
        decoder_layer = TransformerDecoderLayer(hidden_size, dropout=dropout,
                                                nhead=num_att_heads)
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_decoder_layers)
        self._reset_transformer_parameters()

        self.fc = nn.Linear(hidden_size, num_classes)
        self.query_embed = nn.Embedding(num_queries, hidden_size)

    def _reset_transformer_parameters(self):
        print('Resetting the transformer parameters')
        for p in self.transformer_encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for p in self.transformer_decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        bsize = x.shape[0]
        img_size = x.shape[2:]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.input_proj is not None:
            x = self.input_proj(x)
        x = self.forward_encoder(x)
        query_embed = self.query_embed.weight
        query_embed = query_embed.unsqueeze(1).repeat(1, bsize, 1)
        hs, decoder_attn = self.transformer_decoder(query_embed, x)
        hs = hs.transpose(1, 2).squeeze(0)
        out = self.fc(hs)
        return out

    def forward_encoder(self, x):
        bsize = x.shape[0]
        dim = x.shape[1]
        x = x.view(bsize, dim, -1)
        bs, num_channels, num_pixels = x.shape
        x = x.unsqueeze(3)
        x = x.view(bs, num_channels, -1).permute(2, 0, 1)
        x = self.transformer_encoder(x)
        return x

    def freeze_bn(self):
        print("Freezing bn.")
        backbone_layers = [self.conv1, self.bn1, self.layer1, self.layer2,
                           self.layer3, self.layer4]
        for layer in backbone_layers:
            for m in layer.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False

    def freeze(self, layer):
        if layer == 'bn':
            self.freeze_bn()
        else:
            raise NotImplementedError

