import torch
import torch.nn as nn
from weight_batch_norm import BatchWeightNorm2d

def conv3x3(i_c, o_c, stride=1):
    return nn.Conv2d(i_c, o_c, 3, stride, 1, bias=False)

class BatchNorm2d(nn.BatchNorm2d):
    def __init__(self, channels, momentum=1e-3, eps=1e-3):
        super().__init__(channels)
        self.update_batch_stats = True

    def forward(self, x):
        if self.update_batch_stats:
            return super().forward(x)
        else:
            return nn.functional.batch_norm(
                x, None, None, self.weight, self.bias, True, self.momentum, self.eps
            )

def relu():
    return nn.LeakyReLU(0.1)

class residual(nn.Module):
    def __init__(self, input_channels, output_channels, stride=1, activate_before_residual=False):
        super().__init__()
        layer = []
        if activate_before_residual:
            self.pre_act = nn.Sequential(
                BatchNorm2d(input_channels),
                relu()
            )
        else:
            self.pre_act = nn.Identity()
            layer.append(BatchNorm2d(input_channels))
            layer.append(relu())
        layer.append(conv3x3(input_channels, output_channels, stride))
        layer.append(BatchNorm2d(output_channels))
        layer.append(relu())
        layer.append(conv3x3(output_channels, output_channels))

        if stride >= 2 or input_channels != output_channels:
            self.identity = nn.Conv2d(input_channels, output_channels, 1, stride, bias=False)
        else:
            self.identity = nn.Identity()

        self.layer = nn.Sequential(*layer)

    def forward(self, x):
        x = self.pre_act(x)
        return self.identity(x) + self.layer(x)

class WRN(nn.Module):
    """ WRN28-width with leaky relu (negative slope is 0.1)"""
    def __init__(self, width, num_classes, transform_fn=None):
        super().__init__()

        self.init_conv = conv3x3(3, 16)

        filters = [16, 16*width, 32*width, 64*width]

        unit1 = [residual(filters[0], filters[1], activate_before_residual=True)] + \
            [residual(filters[1], filters[1]) for _ in range(1, 4)]
        self.unit1 = nn.Sequential(*unit1)

        unit2 = [residual(filters[1], filters[2], 2)] + \
            [residual(filters[2], filters[2]) for _ in range(1, 4)]
        self.unit2 = nn.Sequential(*unit2)

        unit3 = [residual(filters[2], filters[3], 2)] + \
            [residual(filters[3], filters[3]) for _ in range(1, 4)]
        self.unit3 = nn.Sequential(*unit3)

        self.unit4 = nn.Sequential(*[BatchNorm2d(filters[3]), relu(), nn.AdaptiveAvgPool2d(1)])

        self.output = nn.Linear(filters[3], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

        self.transform_fn = transform_fn

    def forward(self, x, return_feature=False):
        if self.training and self.transform_fn is not None:
            x = self.transform_fn(x)
        x = self.init_conv(x)
        x = self.unit1(x)
        x = self.unit2(x)
        x = self.unit3(x)
        f = self.unit4(x)
        c = self.output(f.squeeze())
        if return_feature:
            return [c, f]
        else:
            return c

    def update_batch_stats(self, flag):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.update_batch_stats = flag


class weight_residual(nn.Module):
    def __init__(self, input_channels, output_channels, stride=1, activate_before_residual=False):
        super().__init__()
        self.activate_before_residual = activate_before_residual
        self.bn1 = BatchWeightNorm2d(input_channels)
        self.relu1 = relu()

        self.cov1 = conv3x3(input_channels, output_channels, stride)
        self.bn2 = BatchWeightNorm2d(output_channels)
        self.relu2 = relu()
        self.cov2 = conv3x3(output_channels, output_channels)

        if stride >= 2 or input_channels != output_channels:
            self.identity = nn.Conv2d(input_channels, output_channels, 1, stride, bias=False)
        else:
            self.identity = nn.Identity()

    def forward(self, x, w):
        if self.activate_before_residual:
            x = self.relu1(self.bn1(x, w))
            out = x
        else:
            out = self.relu1(self.bn1(x, w))
        out = self.cov2(self.relu2(self.bn2(self.cov1(out), w)))

        return self.identity(x) + out


class weight_block1(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(weight_block1, self).__init__()

        self.m_block1 = weight_residual(input_channels, output_channels, activate_before_residual=True)
        self.m_block2 = weight_residual(output_channels, output_channels)
        self.m_block3 = weight_residual(output_channels, output_channels)
        self.m_block4 = weight_residual(output_channels, output_channels)

    def forward(self, x, w):
        x = self.m_block1(x, w)
        x = self.m_block2(x, w)
        x = self.m_block3(x, w)
        x = self.m_block4(x, w)
        return x

class weight_block2(nn.Module):
    def __init__(self, input_channels, output_channels, stride):
        super(weight_block2, self).__init__()

        self.m_block1 = weight_residual(input_channels, output_channels, stride)
        self.m_block2 = weight_residual(output_channels, output_channels)
        self.m_block3 = weight_residual(output_channels, output_channels)
        self.m_block4 = weight_residual(output_channels, output_channels)

    def forward(self, x, w):
        x = self.m_block1(x, w)
        x = self.m_block2(x, w)
        x = self.m_block3(x, w)
        x = self.m_block4(x, w)
        return x

class weight_WRN(nn.Module):
    """ WRN28-width with leaky relu (negative slope is 0.1)"""
    def __init__(self, width, num_classes, transform_fn=None):
        super().__init__()

        self.init_conv = conv3x3(3, 16)

        filters = [16, 16*width, 32*width, 64*width]

        self.unit1 = weight_block1(filters[0], filters[1])
        self.unit2 = weight_block2(filters[1], filters[2], 2)
        self.unit3 = weight_block2(filters[2], filters[3], 2)
        self.bn1 = BatchWeightNorm2d(filters[3])
        self.relu1 = relu()
        self.avepool = nn.AdaptiveAvgPool2d(1)
        self.output = nn.Linear(filters[3], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, BatchWeightNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

        self.transform_fn = transform_fn

    def forward(self, x, w, return_feature=False):
        if self.training and self.transform_fn is not None:
            x = self.transform_fn(x)
        x = self.init_conv(x)
        x = self.unit1(x, w)
        x = self.unit2(x, w)
        x = self.unit3(x, w)
        f = self.avepool(self.relu1(self.bn1(x, w)))
        c = self.output(f.squeeze())
        if return_feature:
            return [c, f]
        else:
            return c

    def update_batch_stats(self, flag):
        for m in self.modules():
            if isinstance(m, BatchWeightNorm2d):
                m.update_batch_stats = flag

    def reset_param(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, BatchWeightNorm2d):
                m.reset_running_stats()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)