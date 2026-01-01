"""TALNet model variant configured for PAM WSSED."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Small convolutional block from the original TALNet implementation."""

    def __init__(self, n_input_feature_maps, n_output_feature_maps, kernel_size, batch_norm=False, pool_stride=None):
        super(ConvBlock, self).__init__()
        assert all(x % 2 == 1 for x in kernel_size)
        self.n_input = n_input_feature_maps
        self.n_output = n_output_feature_maps
        self.kernel_size = kernel_size
        self.batch_norm = batch_norm
        self.pool_stride = pool_stride
        # Keep parity with the upstream implementation: bias disabled when batch norm is on.
        self.conv = nn.Conv2d(
            self.n_input,
            self.n_output,
            self.kernel_size,
            padding=tuple(x // 2 for x in self.kernel_size),
            bias=not batch_norm,
        )
        if batch_norm:
            self.bn = nn.BatchNorm2d(self.n_output)
        nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, x):
        x = self.conv(x)
        if self.batch_norm:
            x = self.bn(x)
        x = F.relu(x)
        if self.pool_stride is not None:
            x = F.max_pool2d(x, self.pool_stride)
        return x


class PamTALNet(nn.Module):
    """TALNet architecture with configurable output size for PAM classes."""

    def __init__(self, args, n_classes: int):
        super(PamTALNet, self).__init__()
        self.__dict__.update(args.__dict__)  # Mimic original behavior
        self.n_classes = n_classes
        assert self.n_conv_layers % self.n_pool_layers == 0
        self.input_n_freq_bins = n_freq_bins = 64
        self.embedding_size = args.embedding_size
        self.conv = []
        pool_interval = self.n_conv_layers // self.n_pool_layers
        n_input = 1
        for i in range(self.n_conv_layers):
            if (i + 1) % pool_interval == 0:  # this layer has pooling
                n_freq_bins //= 2
                n_output = self.embedding_size // n_freq_bins
                pool_stride = (2, 2) if i < pool_interval * 2 else (1, 2)
            else:
                n_output = self.embedding_size * 2 // n_freq_bins
                pool_stride = None
            layer = ConvBlock(n_input, n_output, self.kernel_size, batch_norm=self.batch_norm, pool_stride=pool_stride)
            self.conv.append(layer)
            self.__setattr__("conv" + str(i + 1), layer)
            n_input = n_output
        self.gru = nn.GRU(self.embedding_size, self.embedding_size // 2, 1, batch_first=True, bidirectional=True)
        self.fc_prob = nn.Linear(self.embedding_size, self.n_classes)
        if self.pooling == "att":
            self.fc_att = nn.Linear(self.embedding_size, self.n_classes)
        # Better initialization
        nn.init.orthogonal_(self.gru.weight_ih_l0)
        nn.init.constant_(self.gru.bias_ih_l0, 0)
        nn.init.orthogonal_(self.gru.weight_hh_l0)
        nn.init.constant_(self.gru.bias_hh_l0, 0)
        nn.init.orthogonal_(self.gru.weight_ih_l0_reverse)
        nn.init.constant_(self.gru.bias_ih_l0_reverse, 0)
        nn.init.orthogonal_(self.gru.weight_hh_l0_reverse)
        nn.init.constant_(self.gru.bias_hh_l0_reverse, 0)
        nn.init.xavier_uniform_(self.fc_prob.weight)
        nn.init.constant_(self.fc_prob.bias, 0)
        if self.pooling == "att":
            nn.init.xavier_uniform_(self.fc_att.weight)
            nn.init.constant_(self.fc_att.bias, 0)

    def forward(self, x):
        # Input shape: (batch, time, freq)
        x = x.view((-1, 1, x.size(1), x.size(2)))
        for i in range(len(self.conv)):
            if self.dropout > 0:
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.conv[i](x)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view((-1, x.size(1), x.size(2) * x.size(3)))
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x, _ = self.gru(x)
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        frame_prob = torch.sigmoid(self.fc_prob(x))
        frame_prob = torch.clamp(frame_prob, 1e-7, 1 - 1e-7)
        if self.pooling == "max":
            global_prob, _ = frame_prob.max(dim=1)
            return global_prob, frame_prob
        elif self.pooling == "ave":
            global_prob = frame_prob.mean(dim=1)
            return global_prob, frame_prob
        elif self.pooling == "lin":
            global_prob = (frame_prob * frame_prob).sum(dim=1) / frame_prob.sum(dim=1)
            return global_prob, frame_prob
        elif self.pooling == "exp":
            global_prob = (frame_prob * frame_prob.exp()).sum(dim=1) / frame_prob.exp().sum(dim=1)
            return global_prob, frame_prob
        elif self.pooling == "att":
            frame_att = F.softmax(self.fc_att(x), dim=1)
            global_prob = (frame_prob * frame_att).sum(dim=1)
            return global_prob, frame_prob, frame_att
        raise ValueError(f"Unsupported pooling mode: {self.pooling}")

    def predict(self, x, verbose=True, batch_size=32):
        result = []
        for i in range(0, len(x), batch_size):
            with torch.no_grad():
                input_tensor = x[i : i + batch_size]
                output = self.forward(input_tensor)
                if not verbose:
                    output = output[:1]
                result.append([var.cpu().numpy() for var in output])
        concatenated = tuple(np.concatenate(items) for items in zip(*result))
        return concatenated if verbose else concatenated[0]
