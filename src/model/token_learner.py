import mindspore.nn as nn
import mindspore.ops as ops


class TokenLearnerModule(nn.Cell):
    def __init__(self, in_channels, num_tokens, num_groups, dropout_rate):
        super(TokenLearnerModule, self).__init__()
        self.in_channels = in_channels
        self.num_tokens = num_tokens
        self.num_groups = num_groups
        self.norm = nn.LayerNorm([self.in_channels])
        self.attention_maps = nn.SequentialCell([
            nn.Conv2d(self.in_channels, self.in_channels, 1, group=self.num_groups),
            nn.Conv2d(self.in_channels, self.num_tokens, 1)
        ])
        self.feat_conv = nn.Conv2d(self.in_channels, self.in_channels, 1, group=self.num_groups)
        self.softmax = nn.Softmax()
        self.dropout = nn.Dropout(1.0 - dropout_rate)

    def construct(self, x):
        bs, h, w, _ = x.shape

        selected = x
        selected = self.norm(selected)
        selected = selected.transpose(0, 3, 1, 2)
        selected = self.attention_maps(selected)
        selected = selected.transpose(0, 2, 3, 1)
        selected = selected.reshape(bs, h * w, -1)
        selected = selected.transpose(0, 2, 1)
        selected = self.softmax(selected)

        feat = x
        feat = feat.transpose(0, 3, 1, 2)
        feat = self.feat_conv(feat)
        feat = feat.transpose(0, 2, 3, 1)
        feat = feat.reshape(bs, h * w, -1)

        outputs = ops.matmul(selected, feat)
        outputs = self.dropout(outputs)
        return outputs, selected
