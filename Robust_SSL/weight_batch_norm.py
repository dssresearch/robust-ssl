from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.parameter import Parameter
from torch.nn import functional as F
import torch
from torch.nn import init
from torch.nn.modules import Module
from torch import nn

class _NormBase2(Module):
    """Common base of _InstanceNorm and _BatchNorm"""
    _version = 2
    __constants__ = ['track_running_stats', 'momentum', 'eps', 'weight', 'bias',
                     'running_mean', 'running_var', 'num_batches_tracked',
                     'num_features', 'affine']

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(_NormBase2, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features))
            self.bias = Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def _check_input_dim(self, input):
        raise NotImplementedError

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        version = local_metadata.get('version', None)

        if (version is None or version < 2) and self.track_running_stats:
            # at version 2: added num_batches_tracked buffer
            #               this should have a default value of 0
            num_batches_tracked_key = prefix + 'num_batches_tracked'
            if num_batches_tracked_key not in state_dict:
                state_dict[num_batches_tracked_key] = torch.tensor(0, dtype=torch.long)

        super(_NormBase2, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)


class Batch_test(_BatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super(Batch_test, self).__init__(num_features, eps, momentum, affine)

    def forward(self, input):

        if self.training:
            # self.running_mean.mul_(1 - self.momentum).add_(input.mean([0, 2, 3]).detach() * self.momentum)
            # self.running_var.mul_(1 - self.momentum).add_(input.var([0, 2, 3]).detach() * self.momentum)

            return self.weight.view([1, self.num_features, 1, 1]).expand_as(input) * (input - input.mean([0, 2, 3]).view([1, self.num_features, 1, 1]).expand_as(input)) / (
                    input.var([0, 2, 3], unbiased=False).view([1, self.num_features, 1, 1]).expand_as(input) + self.eps).sqrt() + self.bias.view(
                [1, self.num_features, 1, 1]).expand_as(input)
        else:
            return self.weight.view([1, self.num_features, 1, 1]).expand_as(input) * (input - self.running_mean.view([1, self.num_features, 1, 1]).expand_as(input)) / (
                    self.running_var.view([1, self.num_features, 1, 1]).expand_as(input) + self.eps).sqrt() + self.bias.view([1, self.num_features, 1, 1]).expand_as(input)


class Batch_test2(_BatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super(Batch_test2, self).__init__(num_features, eps, momentum, affine)

    def forward(self, input):

        if self.training:
            # self.running_mean.mul_(1 - self.momentum).add_(input.mean([0, 2, 3]).detach() * self.momentum)
            # self.running_var.mul_(1 - self.momentum).add_(input.var([0, 2, 3]).detach() * self.momentum)

            return F.batch_norm(input, None, None, self.weight, self.bias, True, self.momentum, self.eps)
        else:
            return self.weight.view([1, self.num_features, 1, 1]).expand_as(input) * (input - self.running_mean.view([1, self.num_features, 1, 1]).expand_as(input)) / (
                    self.running_var.view([1, self.num_features, 1, 1]).expand_as(input) + self.eps).sqrt() + self.bias.view([1, self.num_features, 1, 1]).expand_as(input)


class _BatchNorm2d(_BatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super(_BatchNorm2d, self).__init__(num_features, eps, momentum, affine)

    def forward(self, input):
        self._check_input_dim(input)

        if self.training:
            self.running_mean.mul_(1 - self.momentum).add_(input.mean([0, 2, 3]).detach() * self.momentum)
            self.running_var.mul_(1 - self.momentum).add_(input.var([0, 2, 3]).detach() * self.momentum)

            return self.weight.view([1, self.num_features, 1, 1]).expand_as(input) * (input - input.mean([0, 2, 3]).view([1, self.num_features, 1, 1]).expand_as(input)) / (
                    input.var([0, 2, 3], unbiased=False).view([1, self.num_features, 1, 1]).expand_as(input) + self.eps).sqrt() + self.bias.view(
                [1, self.num_features, 1, 1]).expand_as(input)
        else:
            return self.weight.view([1, self.num_features, 1, 1]).expand_as(input) * (input - self.running_mean.view([1, self.num_features, 1, 1]).expand_as(input)) / (
                    self.running_var.view([1, self.num_features, 1, 1]).expand_as(input) + self.eps).sqrt() + self.bias.view([1, self.num_features, 1, 1]).expand_as(input)


class _BatchNorm2d_ntest(_BatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super(_BatchNorm2d_ntest, self).__init__(num_features, eps, momentum, affine)

    def forward(self, input):
        # self._check_input_dim(input)

        if self.training:
            self.running_mean.mul_(1 - self.momentum).add_(input.mean([0, 2, 3]).detach() * self.momentum)
            self.running_var.mul_(1 - self.momentum).add_(input.var([0, 2, 3]).detach() * self.momentum)

            return self.weight.view([1, self.num_features, 1, 1]).expand_as(input) * (input - input.mean([0, 2, 3]).view([1, self.num_features, 1, 1]).expand_as(input)) / (
                    input.var([0, 2, 3], unbiased=False).view([1, self.num_features, 1, 1]).expand_as(input) + self.eps).sqrt() + self.bias.view(
                [1, self.num_features, 1, 1]).expand_as(input)
        else:
            return self.weight.view([1, self.num_features, 1, 1]).expand_as(input) * (input - input.mean([0, 2, 3]).view([1, self.num_features, 1, 1]).expand_as(input)) / (
                    input.var([0, 2, 3], unbiased=False).view([1, self.num_features, 1, 1]).expand_as(input) + self.eps).sqrt() + self.bias.view(
                [1, self.num_features, 1, 1]).expand_as(input)


class BatchNorm2d(_BatchNorm2d):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(input.dim()))


class _BatchNorm1d(_BatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super(_BatchNorm1d, self).__init__(num_features, eps, momentum, affine)

    def forward(self, input):
        self._check_input_dim(input)

        if self.training:
            self.running_mean.mul_(1 - self.momentum).add_(input.mean([0]).detach() * self.momentum)
            self.running_var.mul_(1 - self.momentum).add_(input.var([0]).detach() * self.momentum)

            return self.weight.view([1, self.num_features]).expand_as(input) * (input - input.mean([0]).view([1, self.num_features]).expand_as(input)) / (
                    input.var([0], unbiased=False).view([1, self.num_features]).expand_as(input) + self.eps).sqrt() + self.bias.view([1, self.num_features]).expand_as(input)
        else:
            return self.weight.view([1, self.num_features]).expand_as(input) * (input - self.running_mean.view([1, self.num_features]).expand_as(input)) / (
                    self.running_var.view([1, self.num_features]).expand_as(input) + self.eps).sqrt() + self.bias.view([1, self.num_features]).expand_as(input)


class BatchNorm1d(_BatchNorm1d):
    def _check_input_dim(self, input):
        if input.dim() != 2:
            raise ValueError('expected 2D (got {}D input)'.format(input.dim()))


class _BatchWeightNorm1d(_NormBase2):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super(_BatchWeightNorm1d, self).__init__(num_features, eps, momentum, affine)
        self.update_batch_stats = True

    def forward(self, input, w):
        self._check_input_dim(input)
        if self.training:
            w_exd = w.view([w.shape[0], 1]).expand_as(input)
            mean = (w_exd * input).mean([0])
            mean = mean * input.shape[0] / w.sum()
            var_, var = get_var1d(input, mean, w)

            if self.update_batch_stats:
                self.running_mean.mul_(1 - self.momentum).add_(mean.detach() * self.momentum)
                self.running_var.mul_(1 - self.momentum).add_(var.detach() * self.momentum)

            return self.weight.view([1, self.num_features]).expand_as(input) * (input - mean.view([1, self.num_features]).expand_as(input)) / (
                    var_.view([1, self.num_features]).expand_as(input) + self.eps).sqrt() + self.bias.view([1, self.num_features]).expand_as(input)
        else:
            return self.weight.view([1, self.num_features]).expand_as(input) * (input - self.running_mean.view([1, self.num_features]).expand_as(input)) / (
                    self.running_var.view([1, self.num_features]).expand_as(input) + self.eps).sqrt() + self.bias.view([1, self.num_features]).expand_as(input)

    def reset_parameters(self):
        nn.init.constant_(self.weight, 1)
        nn.init.constant_(self.bias, 0)
        self.running_mean.zero_()
        self.running_var.fill_(1)

class BatchWeightNorm1d(_BatchWeightNorm1d):
    def _check_input_dim(self, input):
        if input.dim() != 2:
            raise ValueError('expected 2D (got {}D input)'.format(input.dim()))


class _BatchWeightNorm2d(_NormBase2):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super(_BatchWeightNorm2d, self).__init__(num_features, eps, momentum, affine)
        self.update_batch_stats = True
    def forward(self, input, w):
        self._check_input_dim(input)

        if self.training:
            w_exd = w.view([w.shape[0], 1, 1, 1]).expand_as(input)
            mean = (w_exd * input).mean([0, 2, 3])
            mean = mean * input.shape[0] / w.sum()
            var_, var = get_var2d(input, mean, w)

            if self.update_batch_stats:
                self.running_mean.mul_(1 - self.momentum).add_(mean.detach() * self.momentum)
                self.running_var.mul_(1 - self.momentum).add_(var.detach() * self.momentum)

            return self.weight.view([1, self.num_features, 1, 1]).expand_as(input) * (input - mean.view([1, self.num_features, 1, 1]).expand_as(input)) / (
                    var_.view([1, self.num_features, 1, 1]).expand_as(input) + self.eps).sqrt() + self.bias.view([1, self.num_features, 1, 1]).expand_as(input)
        else:
            return self.weight.view([1, self.num_features, 1, 1]).expand_as(input) * (input - self.running_mean.view([1, self.num_features, 1, 1]).expand_as(input)) / (
                    self.running_var.view([1, self.num_features, 1, 1]).expand_as(input) + self.eps).sqrt() + self.bias.view([1, self.num_features, 1, 1]).expand_as(input)

    def reset_parameters(self):
        nn.init.constant_(self.weight, 1)
        nn.init.constant_(self.bias, 0)
        self.running_mean.zero_()
        self.running_var.fill_(1)


class _BatchWeightNorm2d_ntest(_NormBase2):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super(_BatchWeightNorm2d_ntest, self).__init__(num_features, eps, momentum, affine)
        self.update_batch_stats = True
    def forward(self, input, w):

        if self.training:
            w_exd = w.view([w.shape[0], 1, 1, 1]).expand_as(input)
            mean = (w_exd * input).mean([0, 2, 3])
            mean = mean * input.shape[0] / w.sum()
            var_, var = get_var2d(input, mean, w)

            if self.update_batch_stats:
                self.running_mean.mul_(1 - self.momentum).add_(mean.detach() * self.momentum)
                self.running_var.mul_(1 - self.momentum).add_(var.detach() * self.momentum)

            return self.weight.view([1, self.num_features, 1, 1]).expand_as(input) * (input - mean.view([1, self.num_features, 1, 1]).expand_as(input)) / (
                    var_.view([1, self.num_features, 1, 1]).expand_as(input) + self.eps).sqrt() + self.bias.view([1, self.num_features, 1, 1]).expand_as(input)
        else:
            w_exd = w.view([w.shape[0], 1, 1, 1]).expand_as(input)
            mean = (w_exd * input).mean([0, 2, 3])
            mean = mean * input.shape[0] / w.sum()
            var_, var = get_var2d(input, mean, w)

            return self.weight.view([1, self.num_features, 1, 1]).expand_as(input) * (input - mean.view([1, self.num_features, 1, 1]).expand_as(input)) / (
                    var_.view([1, self.num_features, 1, 1]).expand_as(input) + self.eps).sqrt() + self.bias.view([1, self.num_features, 1, 1]).expand_as(input)

    def reset_parameters(self):
        nn.init.constant_(self.weight, 1)
        nn.init.constant_(self.bias, 0)
        self.running_mean.zero_()
        self.running_var.fill_(1)

class BatchWeightNorm2d(_BatchWeightNorm2d):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(input.dim()))

def get_var2d(x, new_mean, w):
    scale = x.size(0) * x.size(2) * x.size(3) * w.sum() / x.size(0)
    new_mean = new_mean.view([1, new_mean.shape[0], 1, 1]).expand_as(x)
    tmp = torch.pow((x - new_mean), 2)
    tem_v = tmp * w.view([w.shape[0], 1, 1, 1]).expand_as(tmp)
    new_var = tem_v.mean([0, 2, 3]) * x.size(0) / w.sum()
    runing_temp_var = new_var * scale / (scale - 1)
    return new_var, runing_temp_var

def get_var1d(x, new_mean, w):
    scale = x.size(0) * w.sum() / x.size(0)
    new_mean = new_mean.view([1, new_mean.shape[0]]).expand_as(x)
    tmp = torch.pow((x - new_mean), 2)
    tem_v = tmp * w.view([w.shape[0], 1]).expand_as(tmp)
    new_var = tem_v.mean([0]) * x.size(0) / w.sum()
    runing_temp_var = new_var * scale / (scale - 1)
    return new_var, runing_temp_var

