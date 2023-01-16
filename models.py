import torch
from pytorch_model_summary import summary
from torch import nn
from torchvision import models
from torchvision.models import ResNet18_Weights

import utils


class Resnet18With4Heads(nn.Module):
    def __init__(self):
        super(Resnet18With4Heads, self).__init__()
        self.number_of_heads = 4

        self.resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)

        self.init_block = nn.Sequential(*list(self.resnet.children())[0:4])
        self.hidden_blocks = nn.ModuleList(list(self.resnet.children())[4:8])
        self.head_blocks = nn.ModuleList([self._make_head_block(in_channels) for in_channels in [64, 128, 256, 512]])

    @staticmethod
    def _make_head_block(in_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, 512, kernel_size=3, stride=512 // in_channels, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(1),
            nn.Linear(in_features=512, out_features=10, bias=True)
        )

    def forward(self, x):
        x = self.init_block(x)
        y = []
        for hidden_block, head_block in zip(self.hidden_blocks, self.head_blocks):
            x = hidden_block(x)
            y.append(head_block(x))
        return y, None, None


class Resnet18FrozenWith4Heads(Resnet18With4Heads):
    def __init__(self):
        super(Resnet18FrozenWith4Heads, self).__init__()
        for child in self.resnet.children():
            for param in child.parameters():
                param.requires_grad = False


# noinspection PyMethodOverriding,PyAbstractClass
class TimeFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, evals, weights, gamma, iters=6):
        """
        :param ctx:
        :param evals: in (0, inf) - how much we like a cookie, in overleaf it's p
        :param weights: in [0, 1] - weights after subtracting bites of previous heads
        :param gamma: in [0, 1) - what fraction of sum of all initial (not current!!!) weights we'd like to eat, we assume initial weights were 1 so their sum is weights.size(0)
        :param iters: - how many steps of newton method
        :return: t s.t. weights[i] * exp(-t * evals[i]) equals to how much of i-th cookie should be left for the next heads
        """

        if gamma * weights.size(0) >= torch.sum(weights):
            raise Exception("no solution")  # exponent does not cross the line, we can't eat more than remains

        to_leave = torch.sum(weights) - gamma * weights.size(0)  # total weight of cookies which should be left
        t = torch.zeros(1, requires_grad=False).to(utils.get_device())
        for _ in range(iters):
            v = weights * torch.exp(-t * evals)         # vector batch_size x 1
            b = torch.sum(v) - to_leave                 # scalar: f(t) = w_1 * exp(-t * p_1) + ... - to_leave
            a = torch.squeeze(-v.t() @ evals, dim=1)    # scalar: f'(t) = -p_1 * w_1 * exp(-t * p_1)
            t = t - b / a                               # scalar: t_{n+1} = t_n - f(t_n) / f'(t_n)

        ctx.save_for_backward(evals, weights, t)

        return t

    @staticmethod
    def backward(ctx, grad_output):
        evals, weights, t = ctx.saved_tensors

        dtimefunction_dt = -torch.sum(evals * weights * torch.exp(-evals * t))
        dtimefunction_devals = -t * weights * torch.exp(-evals * t)
        dtimefunction_dweights = torch.exp(-evals * t) - 1

        dt_devals = -dtimefunction_devals / dtimefunction_dt
        dt_dweights = -dtimefunction_dweights / dtimefunction_dt

        return grad_output * dt_devals, grad_output * dt_dweights, None


class Resnet18With4HeadsDsob(nn.Module):
    def __init__(self):
        super(Resnet18With4HeadsDsob, self).__init__()
        self.number_of_heads = 4
        self.gammas = [0.25, 0.25, 0.25, None]
        # self.gammas = [0.05, 0.05, 0.05, None]

        self.resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)

        self.init_block = nn.Sequential(*list(self.resnet.children())[0:4])
        self.hidden_blocks = nn.ModuleList(list(self.resnet.children())[4:8])
        self.head_blocks = nn.ModuleList([self._make_head_block(in_channels) for in_channels in [64, 128, 256, 512]])

        self.eval_blocks = nn.ModuleList([self._make_eval_block(in_channels) for in_channels in [64, 128, 256, 512]])

        self.time_function = TimeFunction.apply

    @staticmethod
    def _make_head_block(in_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, 512, kernel_size=3, stride=512 // in_channels, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(1),
            nn.Linear(in_features=512, out_features=10, bias=True)
        )

    @staticmethod
    def _make_eval_block(in_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, 512, kernel_size=3, stride=512 // in_channels, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(1),
            nn.Linear(in_features=512, out_features=1, bias=True),
            # nn.Softmax(dim=0)
            # nn.Sigmoid()
        )

    def forward(self, x):
        x = self.init_block(x)
        y = []
        w = []
        evals = []
        weights = torch.ones(x.size(0), 1, requires_grad=False).to(utils.get_device())
        for hidden_block, head_block, eval_block, gamma in zip(self.hidden_blocks,
                                                               self.head_blocks,
                                                               self.eval_blocks,
                                                               self.gammas):
            x = hidden_block(x)
            eval = torch.exp(eval_block(x))
            if self.train:
                if gamma:
                    t = self.time_function(eval, weights, gamma)
                    consume_weights = weights - weights * torch.exp(-t * eval)
                    assert torch.isclose(torch.sum(consume_weights), torch.Tensor([gamma * x.size(0)]).to(utils.get_device()))  # we consumed batch_size * gamma
                else:
                    consume_weights = weights
                # weights = weights - consume_weights.detach()
                weights = weights - consume_weights
                w.append(consume_weights)

            y.append(head_block(x))
            evals.append(eval)
        return y, w, evals


class Resnet18FrozenWith4HeadsDsob(Resnet18With4HeadsDsob):
    def __init__(self):
        super(Resnet18FrozenWith4HeadsDsob, self).__init__()
        for child in self.resnet.children():
            for param in child.parameters():
                param.requires_grad = False


MODEL_NAME_MAP = {
    'resnet18_4heads': Resnet18With4Heads,
    'resnet18_frozen_4heads': Resnet18FrozenWith4Heads,
    'resnet18_4heads_dsob': Resnet18With4HeadsDsob,
    'resnet18_frozen_4heads_dsob': Resnet18FrozenWith4HeadsDsob,
}

if __name__ == '__main__':
    # print(summary(Resnet18With4Heads(), torch.zeros((1, 3, 224, 224)), show_input=False))
    # print(summary(Resnet18FrozenWith4Heads(), torch.zeros((1, 3, 224, 224)), show_input=False))
    print(summary(Resnet18With4HeadsDsob(), torch.zeros((2, 3, 224, 224)), show_input=False))
    # print(Resnet18With4HeadsDsob())
    # for child in Resnet18With4HeadsDsob().children():
    #     for param in child.parameters():
    #         print(param.requires_grad)
    # print(summary(Resnet18FrozenWith4HeadsDsob(), torch.zeros((1, 3, 224, 224)), show_input=False))

    # x = torch.zeros((5, 3, 224, 224))
    # labels = torch.ones((5,), dtype=torch.long)
    # model = Resnet18With4HeadsDsob()
    # outputs, consume_weights, _ = model(x)
    # criterion = nn.CrossEntropyLoss(reduction='none')
    # # for each head and sample calculate criterion loss and weigh it by consume_weights
    # losses = [criterion(head_outputs, labels) for head_outputs in outputs]
    # if consume_weights:
    #     losses = [l * w for l, w in zip(losses, consume_weights)]
    # # for each head aggregate the scaled criterion losses by taking mean, and sum all heads losses
    # loss = sum([torch.mean(l) for l in losses])
    #
    # make_dot(loss, params=dict(model.named_parameters())).render("graph2", format="png")
