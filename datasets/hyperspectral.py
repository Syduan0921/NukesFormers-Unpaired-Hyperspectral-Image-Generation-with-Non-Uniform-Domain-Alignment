import glob
import random
import os

import cv2
import scipy.io
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import h5py
import torch

def normalize(data, max_val, min_val):
    return (data-min_val)/(max_val-min_val)

class MixDatasetVal(Dataset):
    def __init__(self, args, sr='/home/data/duanshiyao/USGS_Library/librarySR.mat', mode='val'):
        if mode != 'val':
            raise Exception("Invalid mode!", mode)
        data_path = args
        self.mat = data_path
        data_names = os.listdir(self.mat)
        self.keys = data_names
        random.shuffle(data_names)
        self.keys = [self.mat + data_names[i] for i in range(len(self.keys))]

        # self.keys.sort()
        #self.masked_position_generator = RandomMaskingGenerator(args.window_size, args.mask_ratio)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        #mat = h5py.File(self.keys[index], 'r')
        try:
            mat = scipy.io.loadmat(self.keys[index])
        except:
            mat = h5py.File(self.keys[index], 'r')
        hyper = np.float32(np.array(mat['cube']))
        #hyper = normalize(hyper, max_val=1., min_val=0.)
        hyper = np.transpose(hyper, [2, 0, 1])
        hyper = torch.Tensor(hyper)
        rgb = np.float32(np.array(mat['rgb']))
        rgb = np.transpose(rgb, [2, 0, 1])
        rgb = torch.Tensor(rgb)
        #mat.close()
        return rgb, hyper

class HyperDatasetVal22(Dataset):
    def __init__(self, args,mode='val'):
        if mode != 'val':
            raise Exception("Invalid mode!", mode)
        data_path = args
        self.mat = data_path +"/Valid_spectral/"
        self.rgb = data_path + "/Valid_RGB/"
        data_names = os.listdir(self.mat)
        self.keys = data_names
        self.keys.sort()
        #random.shuffle(data_names)
        self.keys = [self.mat + data_names[i] for i in range(len(self.keys))]
        self.rgbkeys = [self.rgb + data_names[i].split('.')[0]+".jpg" for i in range(len(self.keys))]

        # self.keys.sort()
        #self.masked_position_generator = RandomMaskingGenerator(args.window_size, args.mask_ratio)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        #mat = h5py.File(self.keys[index], 'r')
        try:
            mat = scipy.io.loadmat(self.keys[index])
        except:
            mat = h5py.File(self.keys[index], 'r')
        hyper = np.float32(np.array(mat['cube']))
        hyper = normalize(hyper, max_val=1., min_val=0.)

        rgb = cv2.imread(self.rgbkeys[index])  # imread -> BGR model
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = np.transpose(rgb, [2, 0, 1])
        rgb = normalize(np.float32(rgb), max_val=rgb.max(), min_val=rgb.min())

        if hyper.shape[0] == 31 and hyper.shape[2] == 512:
            pass
        elif hyper.shape[0] == 31 and hyper.shape[2] == 482:
            hyper = np.transpose(hyper, [0, 2, 1])
        else:
            print("There is a error")
        hyper = torch.Tensor(hyper)
        rgb = torch.Tensor(rgb)
        return rgb, hyper

class HyperDatasetValR(Dataset):
    def __init__(self, args,mode='val'):
        if mode != 'val':
            raise Exception("Invalid mode!", mode)
        data_path = args
        self.mat = data_path +"/NTIRE2020_Validation_Spectral/"
        self.rgb = data_path + "/NTIRE2020_Validation_RealWorld/"
        data_names = os.listdir(self.mat)
        self.keys = data_names
        random.shuffle(data_names)
        self.keys = [self.mat + data_names[i] for i in range(len(self.keys))]
        self.rgbkeys = [self.rgb + data_names[i].split('.')[0]+"_RealWorld.jpg" for i in range(len(self.keys))]

        # self.keys.sort()
        #self.masked_position_generator = RandomMaskingGenerator(args.window_size, args.mask_ratio)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        #mat = h5py.File(self.keys[index], 'r')
        mat = scipy.io.loadmat(self.keys[index])
        hyper = np.float32(np.array(mat['cube']))
        hyper = normalize(hyper, max_val=1., min_val=0.)

        rgb = cv2.imread(self.rgbkeys[index])  # imread -> BGR model
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = np.transpose(rgb, [2, 1, 0])
        rgb = normalize(np.float32(rgb), max_val=255., min_val=0.)

        hyper = np.transpose(hyper, [2, 1, 0])
        hyper = torch.Tensor(hyper)
        rgb = torch.Tensor(rgb)
        return rgb, hyper

class HyperDatasetVal(Dataset):
    def __init__(self, args,mode='val'):
        if mode != 'val':
            raise Exception("Invalid mode!", mode)
        data_path = args
        self.mat = data_path +"/NTIRE2020_Validation_Spectral/"
        self.rgb = data_path + "/NTIRE2020_Validation_Clean/"
        data_names = os.listdir(self.mat)
        self.keys = data_names
        random.shuffle(data_names)
        self.keys = [self.mat + data_names[i] for i in range(len(self.keys))]
        self.rgbkeys = [self.rgb + data_names[i].split('.')[0]+"_clean.png" for i in range(len(self.keys))]

        # self.keys.sort()
        #self.masked_position_generator = RandomMaskingGenerator(args.window_size, args.mask_ratio)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        #mat = h5py.File(self.keys[index], 'r')
        mat = scipy.io.loadmat(self.keys[index])
        hyper = np.float32(np.array(mat['cube']))
        hyper = normalize(hyper, max_val=1., min_val=0.)

        rgb = cv2.imread(self.rgbkeys[index])  # imread -> BGR model
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = np.transpose(rgb, [2, 1, 0])
        rgb = normalize(np.float32(rgb), max_val=255., min_val=0.)

        hyper = np.transpose(hyper, [2, 1, 0])
        hyper = torch.Tensor(hyper)
        rgb = torch.Tensor(rgb)
        return rgb, hyper

class HyperDatasetVal_CAVE(Dataset):
    def __init__(self, args,mode='val'):
        if mode != 'val':
            raise Exception("Invalid mode!", mode)
        data_path = args
        self.mat = data_path +"/spectrum/"
        self.rgb = data_path + "/NTIRE2020_Validation_RealWorld/"
        data_names = os.listdir(self.mat)
        self.keys = data_names
        self.keys = [self.mat + data_names[i] for i in range(len(self.keys))]
        self.rgbkeys = [self.rgb + data_names[i].split('.')[0]+"_RealWorld.jpg" for i in range(len(self.keys))]
        self.res_path = "/home/data/dsy/data/DCD_dataset/CAVE/resp.mat"
        self.keys.sort()
        #self.masked_position_generator = RandomMaskingGenerator(args.window_size, args.mask_ratio)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        #mat = h5py.File(self.keys[index], 'r')
        res = scipy.io.loadmat(self.res_path)['resp']
        try:
            mat = scipy.io.loadmat(self.keys[index])
        except:
            mat = h5py.File(self.keys[index], 'r')
        hyper = np.float32(np.array(mat['rad'])/(2**16-1))
        if hyper.shape[0] == 31:
            pass
        else:
            hyper = np.transpose(hyper, [2, 0, 1])
        rgb = np.transpose(np.tensordot(np.transpose(hyper, [1, 2, 0]), np.transpose(res, [1, 0]), (-1, 0)),
                           [2, 0, 1])


        hyper = torch.Tensor(hyper)
        rgb = torch.Tensor(rgb)
        return rgb, hyper


class ImageDatasetVal(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='val'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, '%s/rgb' % mode) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, '%s/hyper' % mode) + '/*.*'))

    def arguement(self, img, rotTimes, vFlip, hFlip):
        # Random rotation
        for j in range(rotTimes):
            img = np.rot90(img.copy(), axes=(1, 2))
        # Random vertical Flip
        for j in range(vFlip):
            img = img[:, :, ::-1].copy()
        # Random horizontal Flip
        for j in range(hFlip):
            img = img[:, ::-1, :].copy()
        return img

    def __getitem__(self, index):
        try:
            mat = h5py.File(self.files_A[index % len(self.files_A)], 'r')
        except:
            mat = scipy.io.loadmat(self.files_A[index % len(self.files_A)])
        rgb = np.float32(np.array(mat['rgb']))
        hyper = np.float32(np.array(mat['cube']))
        item_A, item_B = rgb, hyper

        if item_B.shape[2] == 31:
            item_B = torch.Tensor(np.transpose(item_B, [2, 0, 1]))
            item_A = torch.Tensor(np.transpose(item_A, [2, 0, 1]))
        elif item_B.shape[1] == 31:
            item_B = torch.Tensor(np.transpose(item_B, [1, 0, 2]))
            item_A = torch.Tensor(np.transpose(item_A, [1, 0, 2]))
        else:
            item_B = torch.Tensor(item_B)
            item_A = torch.Tensor(item_A)
        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, '%s/rgb' % mode) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, '%s/hyper' % mode) + '/*.*'))
        yues = 1

    def arguement(self, img, rotTimes, vFlip, hFlip):
        # Random rotation
        for j in range(rotTimes):
            img = np.rot90(img.copy(), axes=(1, 2))
        # Random vertical Flip
        for j in range(vFlip):
            img = img[:, :, ::-1].copy()
        # Random horizontal Flip
        for j in range(hFlip):
            img = img[:, ::-1, :].copy()
        return img

    def __getitem__(self, index):
        rotTimes = random.randint(0, 3)
        vFlip = random.randint(0, 1)
        hFlip = random.randint(0, 1)
        try:
            mat = h5py.File(self.files_A[index % len(self.files_A)], 'r')
        except:
            mat = scipy.io.loadmat(self.files_A[index % len(self.files_A)])
        rgb = np.float32(np.array(mat['rgb']))

        item_A = np.ascontiguousarray(self.arguement(rgb, rotTimes, vFlip, hFlip))

        if self.unaligned:
            try:
                mat = h5py.File(self.files_B[random.randint(0, len(self.files_B) - 1)], 'r')
            except:
                mat = scipy.io.loadmat(self.files_B[random.randint(0, len(self.files_B) - 1)])
            hyper = np.float32(np.array(mat['cube']))
        else:
            try:
                mat = h5py.File(self.files_B[index % len(self.files_B)], 'r')
            except:
                mat = scipy.io.loadmat(self.files_B[index % len(self.files_B)])
            hyper = np.float32(np.array(mat['cube']))
        item_B = np.ascontiguousarray(self.arguement(hyper, rotTimes, vFlip, hFlip))
        if item_B.shape[2] == 31:
            item_B = torch.Tensor(np.transpose(item_B, [2, 0, 1]))
            item_A = torch.Tensor(np.transpose(item_A, [2, 0, 1]))
        elif item_B.shape[1] == 31:
            item_B = torch.Tensor(np.transpose(item_B, [1, 0, 2]))
            item_A = torch.Tensor(np.transpose(item_A, [1, 0, 2]))
        else:
            item_B = torch.Tensor(item_B)
            item_A = torch.Tensor(item_A)
        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))



import os
import glob
import numpy as np
import random
from torch.utils.data import Dataset

class TrainRawDataset(Dataset):
    def __init__(self, hyper_data_root, crop_size, arg=True, bgr2rgb=False, stride=32):
        self.crop_size = crop_size
        self.hypers = []
        self.bgrs = []
        self.arg = arg
        self.stride = stride
        self.patch_per_img_list = []  # 存储每张图片的 patch 数量

        # 读取高光谱数据列表
        hyper_list = glob.glob(os.path.join(hyper_data_root, '*.npz'))
        hyper_list.sort()
        print(f'len(hyper) of ntire2022 dataset: {len(hyper_list)}')

        for i in range(len(hyper_list)):
            hyper_path = os.path.join(hyper_data_root, hyper_list[i])
            hyper_data = np.load(hyper_path)
            hyper, max_val = hyper_data["raw"], hyper_data["max_val"]

            # 动态获取图片的高度和宽度
            h, w = hyper.shape[:2]

            patch_per_line = (w - crop_size) // stride + 1
            patch_per_colum = (h - crop_size) // stride + 1
            patch_per_img = patch_per_line * patch_per_colum
            self.patch_per_img_list.append(patch_per_img)

            if hyper.shape[0] != h and hyper.shape[1] != h:
                hyper = np.transpose(hyper, [1, 0, 2])
                hyper = (hyper / max_val).astype(np.float32)
            elif hyper.shape[0] == h:
                hyper = (hyper / max_val).astype(np.float32)
            else:
                print("Data invalid!!!")
            hyper = np.transpose(hyper, [2, 0, 1])

            # 直接使用高光谱数据的前三个波段作为 bgr 数据
            bgr = np.concatenate([hyper[0:1], (hyper[1:2] + hyper[2:3]) / 2, hyper[3:4]])

            self.hypers.append(hyper)
            self.bgrs.append(bgr)
            print(f'Ntire2022 scene {i} is loaded.')

        self.img_num = len(self.hypers)
        self.length = sum(self.patch_per_img_list)

    def arguement(self, img, rotTimes, vFlip, hFlip):
        # Random rotation
        for j in range(rotTimes):
            img = np.rot90(img.copy(), axes=(1, 2))
        # Random vertical Flip
        for j in range(vFlip):
            img = img[:, :, ::-1].copy()
        # Random horizontal Flip
        for j in range(hFlip):
            img = img[:, ::-1, :].copy()
        return img

    def __getitem__(self, idx):
        stride = self.stride
        crop_size = self.crop_size
        img_idx = 0
        cumulative_patch = 0
        # 找到对应的图片索引
        for i in range(self.img_num):
            if idx < cumulative_patch + self.patch_per_img_list[i]:
                img_idx = i
                patch_idx = idx - cumulative_patch
                break
            cumulative_patch += self.patch_per_img_list[i]
        else:
            # 如果 for 循环没有被 break，说明 idx 超出了范围
            raise IndexError(f"Index {idx} is out of range.")

        h_idx = patch_idx // ((self.hypers[img_idx].shape[2] - crop_size) // stride + 1)
        w_idx = patch_idx % ((self.hypers[img_idx].shape[2] - crop_size) // stride + 1)
        bgr = self.bgrs[img_idx]
        hyper = self.hypers[img_idx]
        bgr = bgr[:, h_idx * stride:h_idx * stride + crop_size, w_idx * stride:w_idx * stride + crop_size]
        hyper = hyper[:, h_idx * stride:h_idx * stride + crop_size, w_idx * stride:w_idx * stride + crop_size]
        rotTimes = random.randint(0, 3)
        vFlip = random.randint(0, 1)
        hFlip = random.randint(0, 1)
        if self.arg:
            bgr = self.arguement(bgr, rotTimes, vFlip, hFlip)
            hyper = self.arguement(hyper, rotTimes, vFlip, hFlip)
        return np.ascontiguousarray(bgr), np.ascontiguousarray(hyper)

    def __len__(self):
        return sum(self.patch_per_img_list)

def main():
    TrainRawDatasets = TrainRawDataset("/home/data/dsy/code/Export-Unpaired_SR/data/raw/", crop_size=128, stride=32)
    for data in TrainRawDatasets:
        flag = 1

if __name__ == "__main__":
    main()