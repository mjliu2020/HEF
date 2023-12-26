import torch


def CalSS(input1, input2):

    out = torch.mul(input1, input2)
    out = torch.sum(out, dim=2, keepdim=True)

    return out