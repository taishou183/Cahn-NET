import torch
import time
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from pytorch_msssim import ssim
import torch.nn.functional as F
from skvideo.measure import niqe
import torchmetrics
torchmetrics.PeakSignalNoiseRatio

class AverageMeter():
    """ Computes and stores the average and current value """

    def __init__(self):
        self.reset()

    def reset(self):
        """ Reset all statistics """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """ Update statistics """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """ Computes the precision@k for the specified values of k """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # one-hot case
    if target.ndimension() > 1:
        target = target.max(1)[1]

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(1.0 / batch_size))

    return res


def compute_psnr_ssim(recoverd, clean):

    assert recoverd.shape == clean.shape, "Shapes of recovered and clean images must match."

    B, C, H, W = recoverd.shape

    mse = F.mse_loss(recoverd, clean, reduction='none').mean((1, 2, 3))
    psnr_val = 10 * torch.log10(1 / mse)

    down_ratio = max(1, round(min(H, W) / 256))

    recoverd_down = F.adaptive_avg_pool2d(recoverd, (int(H / down_ratio), int(W / down_ratio)))
    clean_down = F.adaptive_avg_pool2d(clean, (int(H / down_ratio), int(W / down_ratio)))

    ssim_val = ssim(recoverd_down, clean_down, data_range=1, size_average=False)


    avg_psnr = psnr_val.mean().item()
    avg_ssim = ssim_val.mean().item()

    return avg_psnr, avg_ssim, B


def compute_niqe(image):
    image = np.clip(image.detach().cpu().numpy(), 0, 1)
    image = image.transpose(0, 2, 3, 1)
    niqe_val = niqe(image)

    return niqe_val.mean()

class timer():
    def __init__(self):
        self.acc = 0
        self.tic()

    def tic(self):
        self.t0 = time.time()

    def toc(self):
        return time.time() - self.t0

    def hold(self):
        self.acc += self.toc()

    def release(self):
        ret = self.acc
        self.acc = 0

        return ret

    def reset(self):
        self.acc = 0