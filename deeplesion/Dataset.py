import os
import os.path
import numpy as np
import random
import h5py
import torch
import torch.utils.data as udata
import PIL.Image as Image
from numpy.random import RandomState
import scipy.io as sio
import PIL
from PIL import Image
from .build_gemotry import initialization, build_gemotry
from sklearn.cluster import k_means
import scipy

param = initialization()
ray_trafo = build_gemotry(param)


def image_get_minmax():
    return 0.0, 1.0

def proj_get_minmax():
    return 0.0, 4.0

# hand-crafted prior image
sigma = 1
smFilter = sio.loadmat('deeplesion/gaussianfilter.mat')['smFilter']
miuAir = 0
miuWater=0.192
starpoint = np.zeros([3, 1])
starpoint[0] = miuAir
starpoint[1] = miuWater
starpoint[2] = 2 * miuWater

def nmarprior(im,threshWater,threshBone,miuAir,miuWater,smFilter):
    imSm = scipy.ndimage.filters.convolve(im, smFilter, mode='nearest')
    # print("imSm, h:, w:", imSm.shape[0], imSm.shape[1]) # imSm, h:, w: 416 416
    priorimgHU = imSm
    priorimgHU[imSm <= threshWater] = miuAir
    h, w = imSm.shape[0], imSm.shape[1]
    priorimgHUvector = np.reshape(priorimgHU, h*w)
    region1_1d = np.where(priorimgHUvector > threshWater)
    region2_1d = np.where(priorimgHUvector < threshBone)
    region_1d = np.intersect1d(region1_1d, region2_1d)
    priorimgHUvector[region_1d] = miuWater
    priorimgHU = np.reshape(priorimgHUvector,(h,w))
    return priorimgHU

def nmar_prior(XLI, M):
    XLI[M == 1] = 0.192
    h, w = XLI.shape[0], XLI.shape[1]
    im1d = XLI.reshape(h * w, 1)
    best_centers, labels, best_inertia = k_means(im1d, n_clusters=3, init=starpoint, max_iter=300)
    threshBone2 = np.min(im1d[labels ==2])
    threshBone2 = np.max([threshBone2, 1.2 * miuWater])
    threshWater2 = np.min(im1d[labels == 1])
    imPriorNMAR = nmarprior(XLI, threshWater2, threshBone2, miuAir, miuWater, smFilter)
    return imPriorNMAR


def normalize(data, minmax):
    data_min, data_max = minmax
    data = np.clip(data, data_min, data_max)
    data = (data - data_min) / (data_max - data_min)
    data = data.astype(np.float32)
    data = data*255.0
    data = np.transpose(np.expand_dims(data, 2), (2, 0, 1))
    return data


class MARTrainDataset(udata.Dataset):
    def __init__(self, dir, patchSize, mask):
        super().__init__()
        self.dir = dir
        self.train_mask = mask
        self.patch_size = patchSize
        self.txtdir = os.path.join(self.dir, 'train_640geo_dir.txt')
        self.mat_files = self._load_valid_files()
        self.rand_state = RandomState(66)

    def _load_valid_files(self):
        valid_files = []
        original_files = [f.strip() for f in open(self.txtdir, 'r').readlines()]

        for f in original_files:
            gt_path = os.path.join(self.dir, 'train_640geo', f)
            try:
                # Validate ground truth file
                with h5py.File(gt_path, 'r') as test_file:
                    if 'image' not in test_file:
                        raise KeyError(f"Missing 'image' dataset in {gt_path}")
                valid_files.append(f)
            except (OSError, KeyError) as e:
                print(f"Excluding corrupted file: {gt_path} - {str(e)}")

        return valid_files

    def __len__(self):
        return len(self.mat_files)

    def __getitem__(self, idx):
        for attempt in range(10):  # Max 10 attempts per item
            try:
                gt_dir = self.mat_files[idx]
                random_mask = random.randint(0, 89)

                # Build proper file paths
                file_dir = os.path.dirname(gt_dir)
                data_file = f"{random_mask}.h5"  # No zero-padding
                abs_dir = os.path.join(self.dir, 'train_640geo', file_dir, data_file)
                gt_absdir = os.path.join(self.dir, 'train_640geo', gt_dir)

                # Load data with explicit checks
                with h5py.File(gt_absdir, 'r') as gt_file:
                    Xgt = gt_file['image'][()]

                with h5py.File(abs_dir, 'r') as file:
                    Xma = file['ma_CT'][()]
                    Sma = file['ma_sinogram'][()]
                    XLI = file['LI_CT'][()]
                    SLI = file['LI_sinogram'][()]
                    Tr = file['metal_trace'][()]

                # Data processing
                Sgt = np.asarray(ray_trafo(Xgt))
                M512 = self.train_mask[:, :, random_mask]
                M = np.array(Image.fromarray(M512).resize((416, 416), PIL.Image.BILINEAR))
                Xprior = nmar_prior(XLI, M)

                # Normalization
                Xprior = normalize(Xprior, image_get_minmax())
                Xma = normalize(Xma, image_get_minmax())
                Xgt = normalize(Xgt, image_get_minmax())
                XLI = normalize(XLI, image_get_minmax())
                Sma = normalize(Sma, proj_get_minmax())
                Sgt = normalize(Sgt, proj_get_minmax())
                SLI = normalize(SLI, proj_get_minmax())
                Tr = 1 - Tr.astype(np.float32)
                Tr = np.transpose(np.expand_dims(Tr, 2), (2, 0, 1))
                Mask = M.astype(np.float32)
                Mask = np.transpose(np.expand_dims(Mask, 2), (2, 0, 1))

                return (torch.Tensor(Xma), torch.Tensor(XLI), torch.Tensor(Xgt),
                        torch.Tensor(Mask), torch.Tensor(Sma), torch.Tensor(SLI),
                        torch.Tensor(Sgt), torch.Tensor(Tr), torch.Tensor(Xprior))

            except (OSError, KeyError) as e:
                print(f"Error processing file: {abs_dir} - {str(e)}")
                # Try new random index if current one fails
                idx = random.randint(0, len(self.mat_files) - 1)
                continue

        # Fallback with zero tensors if all attempts fail
        dummy_shape = (1, 416, 416)  # Adjust dimensions to match your data
        return tuple(torch.zeros(dummy_shape) for _ in range(9))