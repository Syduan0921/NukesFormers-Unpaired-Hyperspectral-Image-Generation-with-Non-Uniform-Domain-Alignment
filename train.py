import time
import os

import numpy as np
import torch
import imgvision as iv
from thop import profile

from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from datasets.hyperspectral import ImageDataset, HyperDatasetVal, HyperDatasetValR, HyperDatasetVal22, MixDatasetVal, HyperDatasetVal_CAVE
from torch.utils.data import DataLoader
import datetime
from utils import time2file_name
from utils import initialize_logger
from utils import AverageMeter
from utils import Loss_MRAE, Loss_RMSE, Loss_PSNR, Loss_L1, Validation
import torch.nn.functional as F
from utils import measure_inference_speed

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    # output path
    date_time = str(datetime.datetime.now())
    date_time = time2file_name(date_time)
    opt.checkpoints_dir = opt.checkpoints_dir + date_time
    if not os.path.exists(opt.checkpoints_dir):
        os.makedirs(opt.checkpoints_dir)

    dataloader = DataLoader(ImageDataset(opt.dataroot, unaligned=True),
                            batch_size=opt.batchSize, shuffle=True, num_workers=opt.n_cpu, drop_last=True)
    dataset_size = len(dataloader)    # get the number of images in the dataset.

    #valloader = DataLoader(HyperDatasetVal(args="/home/data/dsy/data/SSROriDataset/NTIRE2020"),
    #                       batch_size=1, num_workers=opt.n_cpu, drop_last=True)
    valloader = DataLoader(HyperDatasetVal_CAVE(args="/home/data/dsy/data/DCD_dataset/CAVE/CAVE_Val/"),
                            batch_size=1, num_workers=opt.n_cpu, drop_last=True)
    # valloader = DataLoader(HyperDatasetVal22(args="/home/data/dsy/data/SSROriDataset/NTIRE2022"),
    #                       batch_size=1, num_workers=opt.n_cpu, drop_last=True)

    # valloader = DataLoader(MixDatasetVal(args="/home/data/dsy/data/DCD_dataset/grss/val/"),
    #                        batch_size=1, num_workers=opt.n_cpu, drop_last=True)

    model = create_model(opt)      # create a model given opt.model and other options

    input_size, input_size2 = (1, 3, 64, 64), (1, 31, 64, 64)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dummy_input, dummy_input2 = torch.randn(input_size).to(device), torch.randn(input_size2).to(device)

    # 计算self.netE_A的FLOPs
    flops_E, params_E = profile(model.netE_A, inputs=(dummy_input,))
    print(f"self.netE_A FLOPs: {flops_E / 1e9:.2f} GFLOPs, Params: {params_E / 1e6:.2f} M")

    # 计算self.netG_A的FLOPs
    flops_G, params_G = profile(model.netG_A, inputs=(model.netE_A(dummy_input),))
    print(f"self.netG_A FLOPs: {flops_G / 1e9:.2f} GFLOPs, Params: {params_G / 1e6:.2f} M")

    # 计算self.netE_A的FLOPs
    flops_EB, params_EB = profile(model.netE_B, inputs=(dummy_input2,))
    print(f"self.netE_B FLOPs: {flops_E / 1e9:.2f} GFLOPs, Params: {params_E / 1e6:.2f} M")

    # 计算self.netG_A的FLOPs
    flops_GB, params_GB = profile(model.netG_B, inputs=(model.netE_B(dummy_input2),))
    print(f"self.netG_B FLOPs: {flops_G / 1e9:.2f} GFLOPs, Params: {params_G / 1e6:.2f} M")

    # 计算串联模型self.netG_A(self.netE_A)的FLOPs
    # 注意：这仅计算两个模型单独的FLOPs之和，未考虑中间特征维度变化
    total_flops = flops_E + flops_G + flops_EB + flops_GB
    print(f"Total FLOPs (self.netG_A(self.netE_A)): {total_flops / 1e9:.2f} GFLOPs")

    # 测量Encoder的推理速度
    e_avg_time, e_std_time, e_fps = measure_inference_speed(
        model.netE_A, input_size, num_runs=200, warmup=50, device=device
    )
    print(f"Encoder 推理时间: {e_avg_time * 1000:.2f} ms/张 ± {e_std_time * 1000:.2f} ms, FPS: {e_fps:.2f}")

    # 测量Generator的推理速度
    # 注意：这里需要使用netE_A的输出作为输入
    with torch.no_grad():
        e_output = model.netE_A(torch.randn(input_size).to(device))
    g_avg_time, g_std_time, g_fps = measure_inference_speed(
        model.netG_A, e_output.shape, num_runs=200, warmup=50, device=device
    )
    print(f"Generator 推理时间: {g_avg_time * 1000:.2f} ms/张 ± {g_std_time * 1000:.2f} ms, FPS: {g_fps:.2f}")


    # 测量串联模型的推理速度
    def combined_model(x):
        return model.netG_A(model.netE_A(x))


    total_avg_time, total_std_time, total_fps = measure_inference_speed(
        combined_model, input_size, num_runs=200, warmup=50, device=device
    )
    print(
        f"串联模型 推理时间: {total_avg_time * 1000:.2f} ms/张 ± {total_std_time * 1000:.2f} ms, FPS: {total_fps:.2f}")

    print('The number of training images = %d' % dataset_size)

    total_iters = 0                # the total number of training iterations

    optimize_time = 0.1

    all_iteration = opt.batchSize * opt.n_epochs * 1000

    times = []

    # logging
    log_dir = os.path.join(opt.checkpoints_dir, 'train.log')
    logger = initialize_logger(log_dir)
    psnr_loss_max = 0
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch

        for i, data in enumerate(dataloader):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            batch_size = data["A"].size(0)
            total_iters += batch_size
            epoch_iter += batch_size
            if len(opt.gpu_ids) > 0:
                torch.cuda.synchronize()
            optimize_start_time = time.time()
            if epoch == opt.epoch_count and i == 0:
                model.data_dependent_initialize(data)
                model.setup(opt)               # regular setup: load and print networks; create schedulers
                model.parallelize()
            model.set_input(data)  # unpack data from dataset and apply preprocessing
            model.optimize_parameters(total_iters)   # calculate loss functions, get gradients, update network weights
            if len(opt.gpu_ids) > 0:
                torch.cuda.synchronize()
            optimize_time = (time.time() - optimize_start_time) / batch_size * 0.005 + 0.995 * optimize_time

            iter_data_time = time.time()

            if total_iters % 50 == 0:
                model.printer(total_iters, all_iteration)
            if total_iters % 2000 == 0:
                mrae_loss, rmse_loss, psnr_loss, sam_loss, ssim_loss = Validation(valloader, model)

                logger.info(" Iter[%06d], Epoch[%06d], Test MRAE: %.9f, "
                            "Test RMSE: %.9f, Test PSNR: %.9f , Test SSIM: %.9f , Test SAM: %.9f" % (
                                total_iters, total_iters // 1000, mrae_loss, rmse_loss, psnr_loss,
                                ssim_loss/np.pi*189,
                                sam_loss))
                model.save_networks(total_iters // 2000)
                print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
                if psnr_loss_max < psnr_loss:
                    psnr_loss_max = psnr_loss
                    model.save_networks('latest')
                    print('saving the model will bigger psnr in epoch %d, iters %d' % (epoch, total_iters))


        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
