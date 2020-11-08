import torch
import torch.nn as nn
import numpy as np


def iou(predict, target, eps=1e-6):
    dims = tuple(range(predict.ndimension())[1:])
    intersect = (predict * target).sum(dims)
    union = (predict + target - predict * target).sum(dims) + eps
    return (intersect / union).sum() / intersect.nelement()

def iou_loss(predict, target):
    return 1 - iou(predict, target)

def fscore(predict, target, eps=1e-6):
    dims = tuple(range(predict.ndimension())[1:])
    intersect = (predict * target).sum(dims)
    recall = intersect / (target.sum(dims) + eps)
    precision = intersect / (predict.sum(dims) + eps)
    fscore = 2 * precision * recall / (precision + recall + eps)
    return fscore.mean()

def fscore_loss(predict, target):
    return 1 - fscore(predict, target)

def multiview_loss(predicts, targets_a, targets_b, loss_func=iou_loss):
    loss = (loss_func(predicts[0][:, 3], targets_a[:, 3]) + \
            loss_func(predicts[1][:, 3], targets_a[:, 3]) + \
            loss_func(predicts[2][:, 3], targets_b[:, 3]) + \
            loss_func(predicts[3][:, 3], targets_b[:, 3])) / 4
    return loss