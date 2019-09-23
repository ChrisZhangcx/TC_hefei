import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, preds: torch.tensor, labels: torch.tensor):
        smooth = 1.

        preds = torch.argmax(preds, dim=-1).float()
        preds.requires_grad = True
        labels = labels.float()

        pred_flat = preds.view(-1)
        label_flat = labels.view(-1)
        intersection = (pred_flat * label_flat).sum()

        score = (2. * intersection + smooth) / (pred_flat.sum() + label_flat.sum() + smooth)
        score = 1. - score.sum() / labels.size(0)
        return score


if __name__ == '__main__':
    pass
