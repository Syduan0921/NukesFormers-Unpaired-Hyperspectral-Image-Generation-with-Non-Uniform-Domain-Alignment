import os.path

import h5py
import numpy as np
import scipy.io as sio


def main():
    grss_raw_path_hyper = "/home/data/dsy/data/DCD_dataset/grss/val/"
    grss_raw_path_rgb = "/home/data/dsy/data/DCD_dataset/grss/train/rgb"
    grss_new_path = "/home/data/dsy/data/DCD_dataset/grss/train_new"
    if not os.path.exists(grss_new_path):
        os.makedirs(grss_new_path)
    else:
        os.removedirs(grss_new_path)
        os.makedirs(grss_new_path)
    for root, dirs, files in os.walk(grss_raw_path_hyper):
        for index, file in enumerate(files):
            if file.endswith(".mat"):
                file_path = os.path.join(root, file)
                try:
                    mat_data = sio.loadmat(file_path)
                except:
                    mat_data = h5py.File(file_path, 'r')
                hyper_cube = np.float32(np.array(mat_data["cube"]))
                rgb_cube = np.float32(np.array(mat_data["rgb"]))
                channel_42 = hyper_cube[:, :, 41]
                rgb_cube_new = np.concatenate((rgb_cube, channel_42[:, :, np.newaxis]), axis=2)
                data_to_save = {
                    "rgb": rgb_cube_new,
                    "cube": hyper_cube
                }
                sio.savemat(file_path, data_to_save)
                print("The {} mat file has been processed.".format(file_path))


if __name__ == "__main__":
    main()
