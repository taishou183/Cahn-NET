import numpy as np
import os
import torch
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader
import torch.nn.functional as f
import time
import torchvision.transforms as tfs
from data_utils import real_Dataset, RESIDE_Dataset, outRESIDE_Dataset

from utils.val_utils import AverageMeter, compute_psnr_ssim

device = torch.device("cuda:1")


def save_image_tensor(image_tensor, output_path="output/"):
    image_np = image_tensor.detach().cpu().numpy()[0]
    # print(image_np.shape)
    ar = np.clip(image_np * 255, 0, 255).astype(np.uint8)

    if image_np.shape[0] == 1:
        ar = ar[0]
    else:
        assert image_np.shape[0] == 3, image_np.shape
        ar = ar.transpose(1, 2, 0)
    p = Image.fromarray(ar)
    p.save(output_path)


def load_single_pair(hazy_path, gt_path):
    to_tensor = transforms.ToTensor()  # PIL → [0,1]
    hazy = to_tensor(Image.open(hazy_path).convert('RGB')).unsqueeze(0)  # [1,3,H,W]
    hazy = tfs.Normalize(mean=[0.64, 0.6, 0.58], std=[0.14, 0.15, 0.152])(hazy)
    gt = to_tensor(Image.open(gt_path).convert('RGB')).unsqueeze(0)  # [1,3,H,W]
    return hazy, gt


def test(net, testloader, output_path):
    net.eval()
    torch.cuda.empty_cache()

    psnr = AverageMeter()
    ssim = AverageMeter()
    times = []

    with torch.no_grad():
        progress_bar = tqdm(testloader, desc="Testing", leave=True)
        for (degrad_patch, clean_patch, clean_name, haze_name) in progress_bar:

            degrad_patch, clean_patch = degrad_patch.to(device), clean_patch.to(device)

            factor = 4

            h, w = degrad_patch.shape[2], degrad_patch.shape[3]
            H, W = ((h + factor) // factor) * factor, ((w + factor) // factor * factor)
            padh = H - h if h % factor != 0 else 0
            padw = W - w if w % factor != 0 else 0
            degrad_patch = f.pad(degrad_patch, (0, padw, 0, padh), 'reflect')

            start_time = time.time()
            restored = net(degrad_patch)[2] # [2]
            restored = restored[:, :, :h, :w]
            end_time = time.time()
            elapsed_time = end_time - start_time
            times.append(elapsed_time)

            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)
            progress_bar.set_postfix(psnr=f"{temp_psnr:.2f}", ssim=f"{temp_ssim:.4f}")

            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)

            save_image_tensor(restored, output_path + haze_name[0])
            del degrad_patch, clean_patch, restored
            torch.cuda.empty_cache()


        print(f"dehazing : psnr: {psnr.avg:.2f}, ssim: {ssim.avg:.4f}")
        return psnr.avg, ssim.avg


def test_out(net, testloader, output_path):
    net.eval()
    torch.cuda.empty_cache()

    psnr = AverageMeter()
    ssim = AverageMeter()
    times = []

    with torch.no_grad():
        progress_bar = tqdm(testloader, desc="Testing", leave=True)
        for (degrad_patch, clean_patch, clean_name) in progress_bar:

            degrad_patch, clean_patch = degrad_patch.to(device), clean_patch.to(device)

            h, w = degrad_patch.shape[2], degrad_patch.shape[3]

            overlap = 30
            if w > h:
                half_width = (w + overlap) // 2
                degrad_patch1 = degrad_patch[:, :, :, :half_width]
                degrad_patch2 = degrad_patch[:, :, :, half_width - overlap:]
            else:
                half_height = (h + overlap) // 2
                degrad_patch1 = degrad_patch[:, :, :half_height, :]
                degrad_patch2 = degrad_patch[:, :, half_height - overlap:, :]

            degrad_patch1_h, degrad_patch1_w = degrad_patch1.shape[2], degrad_patch1.shape[3]
            degrad_patch2_h, degrad_patch2_w = degrad_patch2.shape[2], degrad_patch2.shape[3]
            degrad_patch1 = factor(degrad_patch1)
            degrad_patch2 = factor(degrad_patch2)

            start_time = time.time()
            restored2 = net(degrad_patch2)[2]
            restored1 = net(degrad_patch1)[2]
            end_time = time.time()
            restored1 = restored1[:, :, :degrad_patch1_h, :degrad_patch1_w]
            restored2 = restored2[:, :, :degrad_patch2_h, :degrad_patch2_w]
            elapsed_time = end_time - start_time
            times.append(elapsed_time)

            if w > h:
                restored = torch.zeros_like(degrad_patch)
                restored[:, :, :, :half_width] = restored1
                restored[:, :, :, half_width - overlap:] = restored2

                restored[:, :, :, half_width - overlap:half_width] = (restored1[:, :, :, -overlap:] + restored2[:, :, :,
                                                                                                      :overlap]) / 2
                restored = restored[:, :, :h, :w]
            else:
                restored = torch.zeros_like(degrad_patch)
                restored[:, :, :half_height, :] = restored1
                restored[:, :, half_height - overlap:, :] = restored2

                restored[:, :, half_height - overlap:half_height, :] = (restored1[:, :, -overlap:, :] + restored2[:, :,
                                                                                                        :overlap,
                                                                                                        :]) / 2
                restored = restored[:, :, :h, :w]

            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)
            progress_bar.set_postfix(psnr=f"{temp_psnr:.2f}", ssim=f"{temp_ssim:.4f}")

            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)

            save_image_tensor(restored, output_path + clean_name[0])
            del degrad_patch, clean_patch, restored
            torch.cuda.empty_cache()

        print(f"dehazing : psnr: {psnr.avg:.2f}, ssim: {ssim.avg:.4f}")
        return psnr.avg, ssim.avg


def test_single(net, hazy_path, gt_path, output_path):
    degrad_patch, clean_patch = load_single_pair(hazy_path, gt_path)
    net.eval()
    torch.cuda.empty_cache()

    psnr = AverageMeter()
    ssim = AverageMeter()
    degrad_patch, clean_patch = degrad_patch.to(device), clean_patch.to(device)

    factor = 32

    h, w = degrad_patch.shape[2], degrad_patch.shape[3]
    H, W = ((h + factor) // factor) * factor, ((w + factor) // factor * factor)
    padh = H - h if h % factor != 0 else 0
    padw = W - w if w % factor != 0 else 0
    degrad_patch = f.pad(degrad_patch, (0, padw, 0, padh), 'reflect')

    restored = net(degrad_patch)[2]  # [2]
    restored = restored[:, :, :h, :w]

    temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)
    print(f"psnr={temp_psnr:.2f}, ssim={temp_ssim:.4f}")
    save_image_tensor(restored, output_path + 'temp.png')
    return temp_psnr, temp_ssim


def factor(degrad_patch):
    factor = 4
    h, w = degrad_patch.shape[2], degrad_patch.shape[3]
    H, W = ((h + factor) // factor) * factor, ((w + factor) // factor * factor)
    padh = H - h if h % factor != 0 else 0
    padw = W - w if w % factor != 0 else 0
    degrad_patch = f.pad(degrad_patch, (0, padw, 0, padh), 'reflect')
    return degrad_patch


output_path = r'./trained_models/GridDehazeNet/ohaze/images/'
os.makedirs(output_path, exist_ok=True)

if __name__ == '__main__':

    val_loader = DataLoader(
        dataset=real_Dataset(os.path.join('/media/StudentGroup/LZ/Dataset/', 'O_Haze/test'), train=False,
                             origin=True,
                             size='no',
                             format='.jpg'),
        batch_size=1, shuffle=False, num_workers=24)


    from train import ChanNetModel

    device = torch.device('cuda:1')
    map_location = torch.device('cpu')

    results = []
    folder_path = r'./trained_models/GridDehazeNet/ohaze/ckpt/ohaze-epoch=211-step=343440-loss=0.0355973.ckpt'
    hazy_path = r'/media/StudentGroup/LZ/Dataset/NH_HAZE/temp/hazy/dataset800/44_013.png'
    gt_path = r'/media/StudentGroup/LZ/Dataset/NH_HAZE/temp/clear/dataset800/44_013.png'

    print(folder_path)
    net = ChanNetModel.load_from_checkpoint(folder_path, map_location=map_location)
    net = net.to(device)
    # print(net.hparams)
    psnr_avg, ssim_avg = test(net, val_loader, output_path)
