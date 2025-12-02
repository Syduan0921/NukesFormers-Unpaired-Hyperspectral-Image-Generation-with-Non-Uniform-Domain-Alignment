import datetime
import sys
import logging
import time

from torch.autograd import Variable
import torch
from visdom import Visdom
import numpy as np
import torch.nn as nn
from torch.nn import init
import imgvision as iv

def Validation(val_loader, model):
    # loss function
    criterion_mrae = Loss_MRAE()
    criterion_rmse = Loss_RMSE()
    criterion_psnr = Loss_PSNR()

    model.eval()
    rmse_loss = []
    losses_mrae = AverageMeter()
    losses_rmse = AverageMeter()
    losses_psnr = AverageMeter()
    losses_sam = AverageMeter()
    losses_ssim = AverageMeter()
    for i, (input, target) in enumerate(val_loader):

        input = input.cuda()
        target = target.cuda()
        with torch.no_grad():
            # compute output
            output = model.validate(input)
            loss_mrae = criterion_mrae(output[:, :, 128:-128, 128:-128], target[:, :, 128:-128, 128:-128])
            loss_rmse = criterion_rmse(output[:, :, 128:-128, 128:-128], target[:, :, 128:-128, 128:-128])
            loss_psnr = criterion_psnr(output[:, :, 128:-128, 128:-128], target[:, :, 128:-128, 128:-128])
            Metric = iv.spectra_metric(torch.squeeze(target[:, :, 128:-128, 128:-128], 0).detach().cpu().numpy(),
                                       torch.squeeze(output[:, :, 128:-128, 128:-128], 0).detach().cpu().numpy())
            loss_sam = Metric.SAM()
            loss_ssim = Metric.SSIM()
            rmse_loss.append(loss_rmse)
        # record loss
        losses_mrae.update(loss_mrae.data)
        losses_rmse.update(loss_rmse.data)
        losses_psnr.update(loss_psnr.data)
        losses_sam.update(loss_sam)
        losses_ssim.update(loss_ssim)
    return losses_mrae.avg, losses_rmse.avg, losses_psnr.avg, losses_sam.avg, losses_ssim.avg

def initialize_logger(file_dir):
    logger = logging.getLogger()
    fhandler = logging.FileHandler(filename=file_dir, mode='a')
    formatter = logging.Formatter('%(asctime)s - %(message)s', "%Y-%m-%d %H:%M:%S")
    fhandler.setFormatter(formatter)
    logger.addHandler(fhandler)
    logger.setLevel(logging.INFO)
    return logger

def time2file_name(time):
    year = time[0:4]
    month = time[5:7]
    day = time[8:10]
    hour = time[11:13]
    minute = time[14:16]
    second = time[17:19]
    time_filename = year + '_' + month + '_' + day + '_' + hour + '_' + minute + '_' + second
    return time_filename

class Logger():
    def __init__(self, n_epochs, batches_epoch):
        self.viz = Visdom()
        self.n_epochs = n_epochs
        self.batches_epoch = batches_epoch
        self.epoch = 1
        self.batch = 1
        self.prev_time = time.time()
        self.mean_period = 0
        self.losses = {}
        self.loss_windows = {}
        self.image_windows = {}


    def log(self, losses=None, images=None):
        self.mean_period += (time.time() - self.prev_time)
        self.prev_time = time.time()

        sys.stdout.write('\rEpoch %03d/%03d [%04d/%04d] -- ' % (self.epoch, self.n_epochs, self.batch, self.batches_epoch))

        for i, loss_name in enumerate(losses.keys()):
            if loss_name not in self.losses:
                self.losses[loss_name] = losses[loss_name].data[0]
            else:
                self.losses[loss_name] += losses[loss_name].data[0]

            if (i+1) == len(losses.keys()):
                sys.stdout.write('%s: %.4f -- ' % (loss_name, self.losses[loss_name]/self.batch))
            else:
                sys.stdout.write('%s: %.4f | ' % (loss_name, self.losses[loss_name]/self.batch))

        batches_done = self.batches_epoch*(self.epoch - 1) + self.batch
        batches_left = self.batches_epoch*(self.n_epochs - self.epoch) + self.batches_epoch - self.batch
        sys.stdout.write('ETA: %s' % (datetime.timedelta(seconds=batches_left*self.mean_period/batches_done)))

        # Draw images
        for image_name, tensor in images.items():
            if image_name not in self.image_windows:
                self.image_windows[image_name] = self.viz.image(tensor2image(tensor.data), opts={'title':image_name})
            else:
                self.viz.image(tensor2image(tensor.data), win=self.image_windows[image_name], opts={'title':image_name})

        # End of epoch
        if (self.batch % self.batches_epoch) == 0:
            # Plot losses
            for loss_name, loss in self.losses.items():
                if loss_name not in self.loss_windows:
                    self.loss_windows[loss_name] = self.viz.line(X=np.array([self.epoch]), Y=np.array([loss/self.batch]),
                                                                    opts={'xlabel': 'epochs', 'ylabel': loss_name, 'title': loss_name})
                else:
                    self.viz.line(X=np.array([self.epoch]), Y=np.array([loss/self.batch]), win=self.loss_windows[loss_name], update='append')
                # Reset losses for next epoch
                self.losses[loss_name] = 0.0

            self.epoch += 1
            self.batch = 1
            sys.stdout.write('\n')
        else:
            self.batch += 1

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Loss_L1(nn.Module):
    def __init__(self):
        super(Loss_L1, self).__init__()

    def forward(self, outputs, label):
        assert outputs.shape == label.shape
        error = torch.abs(outputs - label)
        L1 = torch.mean(error.view(-1))
        return L1


def measure_inference_speed(model, input_size, num_runs=100, warmup=20, device="cuda:0"):
    """
    测量模型的推理速度

    参数:
    model: 待测试的模型
    input_size: 输入张量的形状 (batch_size, channels, height, width)
    num_runs: 测试运行次数
    warmup: 预热运行次数（不参与计时）
    device: 测试设备

    返回:
    avg_time: 平均推理时间 (秒/张)
    std_time: 标准差
    fps: 每秒处理帧数
    """
    # 设置为评估模式


    # 创建输入张量并移至设备
    dummy_input = torch.randn(input_size).to(device)

    # 用于存储每次运行的时间
    run_times = []

    # 预热阶段（GPU需要预热以达到最佳性能）
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)
            # 同步GPU以确保所有操作完成
            if str(device).startswith("cuda"):
                torch.cuda.synchronize()

    # 正式测试阶段
    with torch.no_grad():
        for _ in range(num_runs):
            start_time = time.time()
            _ = model(dummy_input)
            # 同步GPU以确保所有操作完成
            if str(device).startswith("cuda"):
                torch.cuda.synchronize()
            end_time = time.time()
            # 记录单次运行时间（秒）
            run_times.append(end_time - start_time)

    # 计算统计数据
    avg_time = np.mean(run_times)
    std_time = np.std(run_times)
    fps = 1.0 / avg_time

    return avg_time, std_time, fps

class Loss_MRAE(nn.Module):
    def __init__(self):
        super(Loss_MRAE, self).__init__()

    def forward(self, outputs, label):
        assert outputs.shape == label.shape
        error = torch.abs(outputs - label) / (label+1e-6)
        mrae = torch.mean(error.view(-1))
        return mrae

class Loss_RMSE_VIS(nn.Module):
    def __init__(self):
        super(Loss_RMSE_VIS, self).__init__()

    def forward(self, outputs, label):
        assert outputs.shape == label.shape
        error = outputs-label
        sqrt_error = torch.pow(error, 2)
        rmse = torch.sqrt(sqrt_error)
        return rmse

class Loss_RMSE(nn.Module):
    def __init__(self):
        super(Loss_RMSE, self).__init__()

    def forward(self, outputs, label):
        assert outputs.shape == label.shape
        error = outputs-label
        sqrt_error = torch.pow(error, 2)
        rmse = torch.sqrt(torch.mean(sqrt_error.view(-1)))
        return rmse

class Loss_PSNR(nn.Module):
    def __init__(self):
        super(Loss_PSNR, self).__init__()

    def forward(self, im_true, im_fake, data_range=255):
        N = im_true.size()[0]
        C = im_true.size()[1]
        H = im_true.size()[2]
        W = im_true.size()[3]
        Itrue = im_true.clamp(0., 1.).mul_(data_range).resize_(N, C * H * W)
        Ifake = im_fake.clamp(0., 1.).mul_(data_range).resize_(N, C * H * W)
        mse = nn.MSELoss(reduce=False)
        err = mse(Itrue, Ifake).sum(dim=1, keepdim=True).div_(C * H * W)
        psnr = 10. * torch.log((data_range ** 2) / err) / np.log(10.)
        return torch.mean(psnr)

def sam_calu(x, y):
    s = np.sum(np.dot(x, y))
    t = np.sqrt(np.sum(x ** 2)) * np.sqrt(np.sum(y ** 2))
    th = np.arccos(s / t)
    # print(s,t)
    return th