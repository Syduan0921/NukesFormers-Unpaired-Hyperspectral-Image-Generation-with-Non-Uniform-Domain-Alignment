"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from --checkpoints_dir and save the results to --results_dir.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for --num_test images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os

import hdf5storage
import torch
import imgvision as iv
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets.hyperspectral import HyperDatasetVal22, HyperDatasetVal, MixDatasetVal, HyperDatasetVal_CAVE
from models.HSCNN_Plus import HSCNN_Plus
from models.KAN_Ours import NukesFormers
from models.Restormer import Restormer
from models.hrnet import SGN
from models.networks import MST_Plus_Plus, HDNet
from utils import Loss_MRAE, Loss_RMSE, Loss_PSNR, AverageMeter, Loss_RMSE_VIS


def Validation(val_loader, G_A, E_A, G_B, E_B, out_root, label):
    # loss function
    criterion_mrae = Loss_MRAE()
    criterion_rmse = Loss_RMSE()
    criterion_psnr = Loss_PSNR()
    criterion_rmse_visi = Loss_RMSE_VIS()
    G_A.eval()
    E_A.eval()
    G_B.eval()
    E_B.eval()
    rmse_loss = []
    losses_mrae = AverageMeter()
    losses_rmse = AverageMeter()
    losses_psnr = AverageMeter()
    losses_ssim = AverageMeter()
    losses_sam = AverageMeter()
    for i, (input, target) in enumerate(val_loader):
        out_root_file = os.path.join(out_root, str(i)+".mat")

        input = input.cuda()
        target = target.cuda()
        with torch.no_grad():
            # compute output
            b, c, h_inp, w_inp = input.shape
            hb, wb = 8, 8
            pad_h = (hb - h_inp % hb) % hb
            pad_w = (wb - w_inp % wb) % wb
            data = F.pad(input, [0, pad_w, 0, pad_h], mode='reflect')
            B, _, H, W = data.shape
            data = G_A(E_A(data))
            null_space = G_A(E_A(G_B(E_B(data))))
            #output = 2 * data[:, :, :h_inp, :w_inp] - null_space[:, :, :h_inp, :w_inp]
            output = data[:, :, :h_inp, :w_inp]

            Metric = iv.spectra_metric(torch.squeeze(target[:, :, 128:-128, 128:-128], 0).detach().cpu().numpy(),
                                       torch.squeeze(output[:, :, 128:-128, 128:-128], 0).detach().cpu().numpy())
            loss_mrae = criterion_mrae(output[:, :, 128:-128, 128:-128], target[:, :, 128:-128, 128:-128])
            loss_rmse = torch.mean(criterion_rmse(output[:, :, 128:-128, 128:-128], target[:, :, 128:-128, 128:-128]))
            loss_rmse = torch.mean(loss_rmse)
            loss_psnr = criterion_psnr(output[:, :, 128:-128, 128:-128], target[:, :, 128:-128, 128:-128])
            loss_sam = Metric.SAM()
            loss_ssim = Metric.SSIM()
            rmse_loss.append(loss_rmse)
            #loss_rmse = criterion_rmse_visi(output, target)
            # hdf5storage.savemat(out_root_file, {label: torch.squeeze(output).detach().cpu().numpy()}, format='7.3')
            hhh = 1
        # record loss
        losses_mrae.update(loss_mrae.data)
        losses_rmse.update(loss_rmse.data)
        losses_psnr.update(loss_psnr.data)
        losses_ssim.update(loss_ssim)
        losses_sam.update(loss_sam)

    return losses_mrae.avg, losses_rmse.avg, losses_psnr.avg, losses_ssim.avg, losses_sam.avg

def model_write(model_base, output_file):
    model_list = ['Ours']
    for model_name in model_list:
        if model_name == 'Ours':
            model_path = model_base + "/latest_net_G_A.pth"
            model_path_2 = model_base + "/latest_net_E_A.pth"
            model_path_3 = model_base + "/latest_net_G_B.pth"
            model_path_4 = model_base + "/latest_net_E_B.pth"
            G_A = NukesFormers(31, 31, 31, stage=1).cuda()  # create a model given opt.model and other options
            G_B = NukesFormers(3, 3, 3, stage=1).cuda()  # create a model given opt.model and other options
        else:
            raise ValueError("Model not found")


        E_A = nn.Conv2d(3, 31, kernel_size=1, stride=1,
                        bias=False).cuda()  # create a model given opt.model and other options
        E_B = nn.Conv2d(31, 3, kernel_size=1, stride=1,
                        bias=False).cuda()  # create a model given opt.model and other options
        checkpoint_G_A = torch.load(model_path, map_location='cpu')
        checkpoint_E_A = torch.load(model_path_2, map_location='cpu')
        checkpoint_G_B = torch.load(model_path_3, map_location='cpu')
        checkpoint_E_B = torch.load(model_path_4, map_location='cpu')
        G_A.load_state_dict(checkpoint_G_A, strict=False)
        E_A.load_state_dict(checkpoint_E_A, strict=False)
        G_B.load_state_dict(checkpoint_G_B, strict=False)
        E_B.load_state_dict(checkpoint_E_B, strict=False)
        out = Validation(trainloader, G_A, E_A, G_B, E_B, out_root, model_name)
        print(out)

def main():
    model_base = '/home/data/dsy/code/Export-Unpaired_SR/checkpoints/10_14/NTIRE22_MST++/2024_10_17_17_20_16/'
    model_write(model_base, out_root)

if __name__ == '__main__':
    out_root = "Checkpoints/"
    # trainloader = DataLoader(HyperDatasetVal_CAVE(args=""),
    #                        batch_size=1, num_workers=8, drop_last=True)
    # trainloader = DataLoader(MixDatasetVal(args="/home/data/dsy/data/DCD_dataset/grss/val/"),
    #                        batch_size=1, num_workers=8, drop_last=True)
    # trainloader = DataLoader(MixDatasetVal(args="/home/data/duanshiyao/DCD_dataset/grss/val/"),
    #                        batch_size=1, num_workers=8, drop_last=True)
    trainloader = DataLoader(HyperDatasetVal22(args="/home/data/dsy/data/SSROriDataset/NTIRE2022"),
                        batch_size=1, num_workers=8, drop_last=True)
    main()
