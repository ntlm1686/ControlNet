from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
# from tutorial_dataset import MyDataset
from dataloaders import HCOCO
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from skimage import data
from skimage import exposure
from skimage.exposure import match_histograms
import seaborn as sns
from tqdm import tqdm

import torchshow as ts

import torchvision.transforms as T
import PIL

# from torchmetrics import SSIM, PSNR
from PIL import Image
import numpy as np
import cv2
import torch


def tensor2img(tnsr):
    return Image.fromarray(((tnsr + 1) * 127.5).permute(1, 2, 0).cpu().numpy().astype(np.uint8))

def rgb2hsv(rgb_tensor):
    return cv2.cvtColor(np.array(tensor2img(rgb_tensor)), cv2.COLOR_RGB2HSV)

def hsv2rgb(img):
    return torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_HSV2RGB)).permute(2, 0, 1) / 127.5 -1

class METRICS:
    """Peak Signal to Noise Ratio
    img1 and img2 have range [0, 255]"""

    def __init__(self):
        self.name = "PSNR"

    @staticmethod
    def __call__(img1, img2):
        mse = torch.mean((img1 - img2) ** 2)
        return 20 * torch.log10(255.0 / torch.sqrt(mse)), mse

totensor = T.PILToTensor()
topil = T.ToPILImage()
def save_png(x, path=None):
    img = topil((x+1)/2).resize((256, 256), resample=PIL.Image.BICUBIC)
    if path is not None:
        img.save(path)
    return totensor(img).float()

metrics = METRICS()
def evaluation(out, gt):
    # out = out.to(gt.device)
    # if len(out.shape) == 3:
    #     out = out.unsqueeze(0)
    # if len(gt.shape) == 3:
        # gt = gt.unsqueeze(0)
    # ev_results = metrics((out+1)*127.5, (gt+1)*127.5)
    ev_results = metrics(out, gt)
    return {
        'psnr': float(ev_results[0]),
        'mse': float(ev_results[1]),
    }

def transfer(output, input, mask):
    hsv_in, hsv_out = rgb2hsv(input), rgb2hsv(output)
    hsv_in_masked = hsv_in[:,:,2] * mask[0].cpu().numpy()
    hsv_out_masked = hsv_out[:,:,2] * mask[0].cpu().numpy()
    in_matched_flatten = match_histograms(
        hsv_in_masked[hsv_in_masked.nonzero()].flatten(),
        hsv_out_masked[hsv_out_masked.nonzero()].flatten(),
    )
    hsv_in_matched = hsv_in_masked.copy()
    hsv_in_matched[hsv_in_masked.nonzero()] = in_matched_flatten
    hsv_out_clone = hsv_out.copy()
    hsv_out_clone[:,:,2] = hsv_in_matched
    return hsv2rgb(hsv_out_clone) * mask[0].cpu() + input.cpu() * (1 - mask[0]).cpu()

# Configs
resume_path = '/home/jiajie/Code/public/ControlNet/lightning_logs/version_15/checkpoints/epoch=67-step=655315.ckpt'
batch_size = 4
logger_freq = 300
learning_rate = 1e-5
sd_locked = True
only_mid_control = False

# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/cldm_v21.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control

from dataloaders.iharmony import iHarmony4
# dname = "HAdobe5k"
dname = "HCOCO"
ds = iHarmony4('test', datasets=[dname,])
from torch.utils.data import DataLoader
dataloader = DataLoader(ds, num_workers=0, batch_size=16, shuffle=True)

# dataset = HCOCO('test', image_size=512)
# dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)

model = model.cuda()
logger = ImageLogger(batch_frequency=logger_freq)
trainer = pl.Trainer(
    gpus=1,
    precision=32,
    )


psnr_output_list = []
psnr_input_list = []
mse_output_list = []
mse_input_list = []
path_list = []

for batch in tqdm(list(dataloader)):
    images = model.log_images(batch, split='test', N=16)
    
    for ix in range(16):
        mask   = (images['control'][ix][3:] + 1) / 2;  # mask
        gt     = batch['jpg'].permute(0, 3, 1, 2)[ix];  # ground truth
        input  = images['control'][ix][:3];  # input
        output = images['samples_cfg_scale_9.00'][ix].clamp(-1, 1);  # output
        path = batch['path'][ix]

        output = transfer(output, input, mask)  # color transferring

        output = save_png(output, f"results_{dname}/pred/{path}")
        input = save_png(input, f"results_{dname}/comp/{path}")
        gt = save_png(gt, f"results_{dname}/gt/{path}")

        result_out = evaluation(output, gt)
        result_in = evaluation(input, gt)

        psnr_output_list.append(result_out['psnr'])
        psnr_input_list.append(result_in['psnr'])
        mse_output_list.append(result_out['mse'])
        mse_input_list.append(result_in['mse'])
        path_list.append(path)
        

    psnr_out = torch.tensor(psnr_output_list)
    psnr_in = torch.tensor(psnr_input_list)
    mse_out = torch.tensor(mse_output_list)
    mse_in = torch.tensor(mse_input_list)

    print("PSNR Pred", psnr_out.mean())
    print("PSNR Comp", psnr_in.mean())
    print("MSE Pred", mse_out.mean())
    print("MSE Comp", mse_in.mean())
    print("========")

    torch.save(psnr_out, f"results_{dname}/metrics/psnr_pred.pt")
    torch.save(psnr_in, f"results_{dname}/metrics/psnr_out.pt")
    torch.save(psnr_out, f"results_{dname}/metrics/mse_pred.pt")
    torch.save(psnr_in, f"results_{dname}/metrics/mse_out.pt")
    torch.save(path_list, f"results_{dname}/metrics/path_list.pt")