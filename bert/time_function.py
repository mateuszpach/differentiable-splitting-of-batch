import torch


# noinspection PyAbstractClass
class TimeFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, evals, weights, gamma, iters=10):
        """
        :param ctx:
        :param evals: in (0, inf) - how much we like a cookie, in overleaf it's p
        :param weights: in [0, 1] - weights after subtracting bites of previous heads
        :param gamma: in [0, 1) - what fraction of sum of all initial (not current!!!) weights we'd like to eat, we assume initial weights were 1 so their sum is weights.size(0)
        :param iters: - how many steps of newton's method
        :return: t s.t. weights[i] * exp(-t * evals[i]) equals to how much of i-th cookie should be left for the next heads
        """

        if gamma * weights.size(0) >= torch.sum(weights):
            raise Exception("no solution")  # exponent does not cross the line, we can't eat more than remains

        to_leave = torch.sum(weights) - gamma * weights.size(0)  # total weight of cookies which should be left
        t = torch.zeros(1, requires_grad=False, device=evals.device)
        for _ in range(iters):
            v = weights * torch.exp(-t * evals)  # vector batch_size x 1
            b = torch.sum(v) - to_leave  # scalar: f(t) = w_1 * exp(-t * p_1) + ... - to_leave
            a = torch.squeeze(-v.t() @ evals, dim=1)  # scalar: f'(t) = -p_1 * w_1 * exp(-t * p_1)
            t = t - b / a  # scalar: t_{n+1} = t_n - f(t_n) / f'(t_n)

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
        # return grad_output * dt_devals, None, None
