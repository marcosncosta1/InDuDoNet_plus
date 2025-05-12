import os
import os.path
import argparse
from datetime import timedelta

import numpy as np
import torch
import time
import nibabel as nib
from skimage.transform import resize, radon, iradon  # For Radon if build_geometry not used/adapted
from skimage.filters import gaussian  # For blur XLI if chosen
import glob
import re
import scipy.ndimage  # For convolve in nmar_prior
import scipy.io as sio  # For loading smFilter.mat
from sklearn.cluster import k_means  # For nmar_prior

# Attempt to import InDuDoNet+ and geometry tools
# Adjust path if your network and deeplesion folders are structured differently
try:
    from network.indudonet_plus import InDuDoNet_plus

    # Check if build_geometry can be imported and used, otherwise fallback to skimage.radon
    # This is a placeholder, actual usage will depend on how build_gemotry is structured
    # from deeplesion.build_gemotry import initialization, build_gemotry
    # For now, we will assume we need to use skimage.radon for projections
    RADON_AVAILABLE = False  # Set to True if you can use the paper's geometry
    print("Attempting to use skimage.transform.radon for projections.")
except ImportError as e:
    print(f"Error importing InDuDoNet_plus or geometry tools: {e}")
    print("Please ensure 'network' and 'deeplesion' (if used) directories are in PYTHONPATH or current dir.")
    exit()

import warnings

warnings.filterwarnings("ignore")

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="InDuDoNet+_Test_Clinical")
# Model loading arguments
parser.add_argument("--model_dir", type=str, default="models/InDuDoNet+_latest.pt", help='path to model file')
parser.add_argument('--num_channel', type=int, default=32, help='the number of dual channels')
parser.add_argument('--T', type=int, default=4, help='the number of ResBlocks in every ProxNet')
parser.add_argument('--S', type=int, default=10, help='the number of total iterative stages')
parser.add_argument('--eta1', type=float, default=1.0, help='initialization for stepsize eta1')
parser.add_argument('--eta2', type=float, default=5.0, help='initialization for stepsize eta2')
parser.add_argument('--alpha', type=float, default=0.5, help='initialization for weight factor')
# Data path arguments
parser.add_argument("--input_low_dir", type=str, required=True, help='Directory containing X_low .nii.gz files')
parser.add_argument("--output_dir", type=str, required=True, help='Directory to save generated output .npy files')
# Processing arguments
parser.add_argument("--img_size", type=int, default=416, help='Target size for input slices (InDuDoNet+ uses 416x416)')
parser.add_argument("--window_min", type=int, default=-1000, help='Minimum HU value for windowing')
parser.add_argument("--window_max", type=int, default=1000, help='Maximum HU value for windowing')
parser.add_argument("--metal_threshold_hu", type=int, default=2500, help='HU threshold for image-domain metal mask')
parser.add_argument("--slice_axis", type=int, default=2, help='Axis along which to extract 2D slices')
parser.add_argument("--save_format", type=str, default="npy", choices=["npy", "png"],
                    help='Format to save output slices')
parser.add_argument("--xli_mode", type=str, default="approx_radon", choices=["copy", "blur", "approx_radon"],
                    help='Method to generate XLI image')
parser.add_argument("--blur_sigma", type=float, default=1.5, help='Sigma for Gaussian blur if xli_mode is "blur"')
parser.add_argument("--sm_filter_path", type=str, default="deeplesion/gaussianfilter.mat",
                    help='Path to smFilter for nmar_prior')
# GPU arguments
parser.add_argument("--use_GPU", type=bool, default=True, help='use GPU or not')
parser.add_argument("--gpu_id", type=str, default="0", help='GPU id')

opt = parser.parse_args()

# --- GPU Setup ---
if opt.use_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu"); print("CUDA not available, using CPU.")
else:
    device = torch.device("cpu"); print("Using CPU.")
if str(device) == "cuda": print(f"Using GPU: {opt.gpu_id}")

# --- Load smFilter for NMAR Prior ---
try:
    smFilter = sio.loadmat(opt.sm_filter_path)['smFilter']
    print(f"Loaded smFilter from {opt.sm_filter_path}")
except Exception as e:
    print(f"Error loading smFilter from {opt.sm_filter_path}: {e}")
    print("NMAR prior generation might fail or produce suboptimal results.")
    smFilter = np.ones((3, 3)) / 9  # Fallback to simple averaging filter


# --- Utility Functions ---
def mkdir(path):
    if not os.path.exists(path): os.makedirs(path); print(f"--- Created output directory: {path} ---")


def image_get_minmax(): return 0.0, 1.0  # For [0,1] scaled images before *255


def proj_get_minmax(): return 0.0, 4.0  # For sinograms before *255


def normalize_to_0_255(data, min_val, max_val):
    """Clips, normalizes to [0,1], then scales to [0,255]."""
    data_clipped = np.clip(data, min_val, max_val)
    if max_val > min_val:
        data_norm_01 = (data_clipped - min_val) / (max_val - min_val)
    else:
        data_norm_01 = np.zeros_like(data_clipped)
    return data_norm_01 * 255.0


def preprocess_indudonet_input(slice_data_hu, target_size, win_min, win_max, is_sinogram=False):
    """Preprocesses HU slice or sinogram data for InDuDoNet+ input."""
    if is_sinogram:
        min_val, max_val = proj_get_minmax()  # Use sinogram-specific range for initial normalization
    else:  # Is image
        slice_data_hu = np.clip(slice_data_hu, win_min, win_max)  # Apply HU window for images
        min_val, max_val = image_get_minmax()  # Then normalize this [0,1] range before scaling
        # Effectively, map [win_min, win_max] to [0,1] first for images
        if win_max > win_min:
            slice_data_hu = (slice_data_hu - win_min) / (win_max - win_min)
        else:
            slice_data_hu = np.zeros_like(slice_data_hu)

    # Now slice_data_hu is in [0,1] for images, or raw for sinograms
    # The original script's normalize function then clips this [0,1] or [0,4] to [min_val, max_val]
    # and scales to [0,255]. This seems like it implies the input data to normalize()
    # was already in the [0,1] or [0,4] range.

    data_scaled_0_255 = normalize_to_0_255(slice_data_hu, min_val, max_val)

    # Resize (for images; sinograms might have fixed size from Radon)
    if not is_sinogram:
        data_resized = resize(data_scaled_0_255, (target_size, target_size), anti_aliasing=True, preserve_range=True)
    else:
        data_resized = data_scaled_0_255  # Assume sinogram is already correct size from radon

    return torch.from_numpy(data_resized).float().unsqueeze(0).unsqueeze(0)


def nmarprior_adapted(im_hu, threshWater, threshBone, miuAir, miuWater, local_smFilter):
    """Adapted NMAR prior generation from the test script."""
    imSm = scipy.ndimage.filters.convolve(im_hu, local_smFilter, mode='nearest')
    priorimgHU = imSm.copy()  # Important to copy
    priorimgHU[imSm <= threshWater] = miuAir
    h, w = imSm.shape[0], imSm.shape[1]
    priorimgHUvector = np.reshape(priorimgHU, h * w)
    region1_1d = np.where(priorimgHUvector > threshWater)[0]
    region2_1d = np.where(priorimgHUvector < threshBone)[0]
    region_1d = np.intersect1d(region1_1d, region2_1d)
    priorimgHUvector[region_1d] = miuWater
    priorimgHU = np.reshape(priorimgHUvector, (h, w))
    return priorimgHU


def generate_xprior_hu(xli_hu, image_domain_metal_mask_binary, local_smFilter):
    """Generates Xprior in HU space."""
    print("    Generating Xprior_HU...")
    xli_for_prior = xli_hu.copy()
    # Original script sets metal regions in XLI to miuWater before k-means
    xli_for_prior[image_domain_metal_mask_binary == 1] = 0.192  # miuWater

    h, w = xli_for_prior.shape
    im1d = xli_for_prior.reshape(h * w, 1)

    # Define starpoints for k-means as in original script
    miuAir_prior = 0;
    miuWater_prior = 0.192
    starpoint_prior = np.zeros([3, 1])
    starpoint_prior[0] = miuAir_prior
    starpoint_prior[1] = miuWater_prior
    starpoint_prior[2] = 2 * miuWater_prior

    try:
        _, labels, _ = k_means(im1d, n_clusters=3, init=starpoint_prior, n_init=1,
                               max_iter=100)  # Reduced n_init and max_iter for speed
        threshBone2 = np.min(im1d[labels == 2]) if np.any(labels == 2) else 2 * miuWater_prior
        threshBone2 = np.max([threshBone2, 1.2 * miuWater_prior])
        threshWater2 = np.min(im1d[labels == 1]) if np.any(labels == 1) else miuWater_prior
    except Exception as e:  # Fallback if k-means fails
        print(f"    K-means failed for Xprior: {e}. Using default thresholds.")
        threshWater2 = miuWater_prior * 0.5
        threshBone2 = miuWater_prior * 1.5

    xprior_hu = nmarprior_adapted(xli_for_prior, threshWater2, threshBone2, miuAir_prior, miuWater_prior,
                                  local_smFilter)
    print("    Xprior_HU generated.")
    return xprior_hu


def generate_approx_radon_li_hu(xma_slice_hu, metal_region_mask_binary, theta_radon=None):
    """ Generates approximate LI in HU space using Radon transforms. """
    print("    Generating approximate XLI_HU (Radon)...")
    if theta_radon is None: theta_radon = np.linspace(0., 180., max(xma_slice_hu.shape[0], xma_slice_hu.shape[1]),
                                                      endpoint=False)  # Base on image dim

    norm_min_r, norm_max_r = -1024.0, 3071.0
    image_norm_r = np.clip((xma_slice_hu - norm_min_r) / (norm_max_r - norm_min_r), 0, 1)

    sinogram_ma = radon(image_norm_r, theta=theta_radon, circle=False)
    sinogram_metal_mask = radon(metal_region_mask_binary.astype(float), theta=theta_radon, circle=False)
    corruption_mask = sinogram_metal_mask > 1e-6

    sinogram_li = sinogram_ma.copy()
    sino_h, sino_w = sinogram_li.shape
    for i in range(sino_w):  # Iterate over angles
        col = sinogram_li[:, i];
        mask_col = corruption_mask[:, i]
        if np.any(mask_col):
            valid_idx = np.where(~mask_col)[0];
            invalid_idx = np.where(mask_col)[0]
            if len(valid_idx) > 1:
                col[invalid_idx] = np.interp(invalid_idx, valid_idx, col[valid_idx])
            elif len(valid_idx) == 1:
                col[invalid_idx] = col[valid_idx[0]]
            else:
                col[invalid_idx] = 0.0

    reconstructed_li_norm_r = iradon(sinogram_li, theta=theta_radon, circle=False, filter_name='ramp')
    reconstructed_li_hu = reconstructed_li_norm_r * (norm_max_r - norm_min_r) + norm_min_r
    print("    Approximate XLI_HU (Radon) generated.")
    return reconstructed_li_hu


def save_indudonet_output(output_tensor_0_255, filename, save_format="npy"):
    """Saves InDuDoNet+ output. Normalizes from [0,255] to [-1,1] before saving."""
    output_slice_0_255 = output_tensor_0_255.squeeze().cpu().numpy()
    output_slice_neg1_1 = (output_slice_0_255 / 255.0) * 2.0 - 1.0
    output_slice_neg1_1 = np.clip(output_slice_neg1_1, -1.0, 1.0)
    if save_format == "npy":
        np.save(filename, output_slice_neg1_1)
    elif save_format == "png":
        from PIL import Image
        output_scaled_uint8 = ((output_slice_neg1_1 + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(output_scaled_uint8).save(filename)


def load_nifti_slice_raw(filepath, slice_index, axis=2):
    try:
        nii_img = nib.load(filepath);
        data = nii_img.get_fdata()
        if axis == 0:
            slice_data = data[slice_index, :, :]
        elif axis == 1:
            slice_data = data[:, slice_index, :]
        else:
            slice_data = data[:, :, slice_index]
        return slice_data  # Return raw, no rotation for processing
    except Exception as e:
        print(f"Error loading NIfTI {filepath} slice {slice_index}: {e}"); return None


# --- Main Function ---
def main():
    start_time_main = time.time()
    mkdir(opt.output_dir)

    print('Loading InDuDoNet+ model ...')
    model = InDuDoNet_plus(opt).to(device)
    try:
        model.load_state_dict(torch.load(opt.model_dir, map_location=device))
        print(f"Model loaded successfully from {opt.model_dir}")
    except Exception as e:
        print(f"Error loading model from {opt.model_dir}: {e}"); return
    model.eval()

    low_files = sorted(glob.glob(os.path.join(opt.input_low_dir, '*.nii.gz')))
    if not low_files: print(f"Error: No input .nii.gz files found in {opt.input_low_dir}"); return
    print(f"Found {len(low_files)} input NIfTI volumes to process.")

    total_slices_processed = 0;
    total_script_time = 0

    # Define Radon angles (e.g., 640 views over 180 degrees for parallel beam)
    # The paper mentions "640 projection views are uniformly spaced in 360 degrees" for fan-beam.
    # Skimage radon uses 0-180 for parallel. We'll use a high number of angles.
    theta_radon = np.linspace(0., 180., 640, endpoint=False)  # Example: 640 views

    for i, low_filepath in enumerate(low_files):
        base_filename = os.path.basename(low_filepath).replace('.nii.gz', '')
        print(f"\nProcessing file {i + 1}/{len(low_files)}: {base_filename}")

        try:
            data_low_full_volume = nib.load(low_filepath).get_fdata()
            num_slices = data_low_full_volume.shape[opt.slice_axis]
            print(f"  Volume shape: {data_low_full_volume.shape}, Slices: {num_slices}")
            vol_start_time = time.time()

            for slice_idx in range(num_slices):
                slice_low_hu_raw = load_nifti_slice_raw(low_filepath, slice_idx, opt.slice_axis)
                if slice_low_hu_raw is None: continue

                # 1. Generate Image-Domain Metal Mask (metal=1, non-metal=0)
                image_domain_metal_mask_binary = (slice_low_hu_raw >= opt.metal_threshold_hu).astype(np.float32)
                if np.sum(image_domain_metal_mask_binary) == 0: continue  # Skip if no metal

                # 2. Generate XLI_hu (approximate Linear Interpolation image in HU)
                if opt.xli_mode == "approx_radon":
                    xli_hu = generate_approx_radon_li_hu(slice_low_hu_raw, image_domain_metal_mask_binary, theta_radon)
                elif opt.xli_mode == "blur":
                    xli_hu = gaussian(slice_low_hu_raw, sigma=opt.blur_sigma)
                else:  # copy mode
                    xli_hu = slice_low_hu_raw.copy()

                # 3. Generate Xprior_hu (NMAR-like prior in HU)
                xprior_hu = generate_xprior_hu(xli_hu, image_domain_metal_mask_binary, smFilter)

                # 4. Preprocess image-domain inputs for the network ([0,255] scaled)
                Xma_tensor = preprocess_indudonet_input(slice_low_hu_raw, opt.img_size, opt.window_min,
                                                        opt.window_max).to(device)
                XLI_tensor = preprocess_indudonet_input(xli_hu, opt.img_size, opt.window_min, opt.window_max).to(device)
                Xprior_tensor = preprocess_indudonet_input(xprior_hu, opt.img_size, opt.window_min, opt.window_max).to(
                    device)

                # 5. Simulate Sinogram-domain inputs
                #    Normalize image to [0,1] for radon, then scale sinogram to [0,255]

                # Sma (Sinogram of Xma)
                img_for_radon_xma = np.clip((slice_low_hu_raw - opt.window_min) / (opt.window_max - opt.window_min), 0,
                                            1) if (opt.window_max > opt.window_min) else np.zeros_like(slice_low_hu_raw)
                img_for_radon_xma_resized = resize(img_for_radon_xma, (opt.img_size, opt.img_size), anti_aliasing=True)
                sma_raw = radon(img_for_radon_xma_resized, theta=theta_radon, circle=False)
                Sma_tensor = preprocess_indudonet_input(sma_raw, opt.img_size, 0, 0, is_sinogram=True).to(
                    device)  # min/max for proj_get_minmax

                # SLI (Sinogram of XLI)
                img_for_radon_xli = np.clip((xli_hu - opt.window_min) / (opt.window_max - opt.window_min), 0, 1) if (
                            opt.window_max > opt.window_min) else np.zeros_like(xli_hu)
                img_for_radon_xli_resized = resize(img_for_radon_xli, (opt.img_size, opt.img_size), anti_aliasing=True)
                sli_raw = radon(img_for_radon_xli_resized, theta=theta_radon, circle=False)
                SLI_tensor = preprocess_indudonet_input(sli_raw, opt.img_size, 0, 0, is_sinogram=True).to(device)

                # Tr (Metal Trace in sinogram, inverted: 1 for non-metal, 0 for metal)
                # Resize image domain metal mask to network input size first
                metal_mask_resized_img_domain = resize(image_domain_metal_mask_binary, (opt.img_size, opt.img_size),
                                                       order=0, anti_aliasing=False, preserve_range=True)
                metal_mask_resized_img_domain = np.clip(metal_mask_resized_img_domain, 0, 1)

                tr_raw = radon(metal_mask_resized_img_domain.astype(float), theta=theta_radon, circle=False)
                tr_binary = (tr_raw > 1e-6).astype(np.float32)  # Binarize the trace
                tr_inverted = 1.0 - tr_binary  # Invert: 1 is non-metal region
                # Tr does not seem to be scaled to 255 in original script, just [0,1]
                Tr_tensor = torch.from_numpy(tr_inverted).float().unsqueeze(0).unsqueeze(0).to(device)

                # Run Inference
                with torch.no_grad():
                    ListX, ListS, ListYS = model(Xma_tensor, XLI_tensor, Sma_tensor, SLI_tensor, Tr_tensor,
                                                 Xprior_tensor)

                output_tensor_0_255 = ListX[-1]  # Final corrected image, expected in [0, 255] range

                # Save Output
                output_filename_base = f"{base_filename}_slice{slice_idx:04d}_indudonet_plus_out"
                output_filepath = os.path.join(opt.output_dir, f"{output_filename_base}.{opt.save_format}")
                save_indudonet_output(output_tensor_0_255, output_filepath, opt.save_format)

                total_slices_processed += 1
                if (slice_idx + 1) % 50 == 0:  # Log less frequently for speed
                    print(f"    Processed slice {slice_idx + 1}/{num_slices}")

            vol_end_time = time.time()
            current_vol_time = vol_end_time - vol_start_time
            print(f"  Finished volume in {current_vol_time:.2f} seconds.")
            total_script_time += current_vol_time
        except Exception as e:
            print(f"Error processing file {low_filepath}: {e}")
            import traceback
            traceback.print_exc()

    end_time_main = time.time()
    print("\n" + "=" * 30)
    print("Processing Complete.")
    print(f"Total slices processed: {total_slices_processed}")
    if total_slices_processed > 0:
        avg_time_slice = total_script_time / total_slices_processed
        print(f"Average inference time per slice (based on volume processing): {avg_time_slice:.4f} seconds")
    print(f"Generated output files saved in: {opt.output_dir}")
    print(f"Total script execution time: {timedelta(seconds=int(end_time_main - start_time_main))}")
    print("=" * 30)


if __name__ == "__main__":
    main()
