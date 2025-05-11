import os
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
from datetime import datetime

def evaluate_single_image(gt_image_path, recon_image_path, output_file="single_image_quality_metrics.txt", hu_min=-1000, hu_max=3000):
    """
    Compute PSNR and SSIM for a single ground truth (GT) and reconstructed image, and save results to a text file.

    Parameters:
        gt_image_path (str): Path to the ground truth image.
        recon_image_path (str): Path to the reconstructed image.
        output_file (str): File to save the results.
        hu_min (int): Minimum HU value in original data.
        hu_max (int): Maximum HU value in original data.
    """

    # Load images
    gt_img = np.squeeze(plt.imread(gt_image_path))
    recon_img = np.squeeze(plt.imread(recon_image_path))

    # Debug: Verify loaded image ranges
    print(f"\nProcessing image: {os.path.basename(gt_image_path)}")
    print(f"GT range: [{gt_img.min():.3f}, {gt_img.max():.3f}]")
    print(f"Reconstructed range: [{recon_img.min():.3f}, {recon_img.max():.3f}]")

    # Convert from PNG to HU scale
    if gt_img.max() <= 1.0:  # Normalized to [0, 1]
        gt_hu = gt_img * (hu_max - hu_min) + hu_min
        recon_hu = recon_img * (hu_max - hu_min) + hu_min
    else:  # Assume [0, 255]
        gt_hu = gt_img * (hu_max - hu_min) / 255 + hu_min
        recon_hu = recon_img * (hu_max - hu_min) / 255 + hu_min

    # Compute dynamic window size for SSIM
    min_dim = min(gt_hu.shape)
    win_size = min(7, min_dim)
    win_size = win_size - 1 if win_size % 2 == 0 else win_size

    # Compute metrics
    psnr_val = psnr(gt_hu, recon_hu, data_range=hu_max - hu_min)
    ssim_val = ssim(gt_hu, recon_hu, data_range=hu_max - hu_min, win_size=win_size, channel_axis=None)

    # Prepare results
    result_text = (
        f"Image Quality Evaluation\n"
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Ground Truth Image: {gt_image_path}\n"
        f"Reconstructed Image: {recon_image_path}\n"
        f"PSNR (dB): {psnr_val:.2f}\n"
        f"SSIM: {ssim_val:.4f}\n"
    )

    # Print results
    print(result_text)

    # Save results to a file
    with open(output_file, "w") as f:
        f.write(result_text)

    print(f"\nResults saved to: {os.path.abspath(output_file)}")

    return psnr_val, ssim_val

if __name__ == "__main__":
    # Set the path to your specific image
    ground_truth_image = "./results/deeplesion/Xgt/img_019/mask_007.png"  # Change this
    reconstructed_image = "./results/deeplesion/X/img_019/mask_007.png"  # Change this
    output_txt = "single_image_quality_metrics.txt"  # Output file name

    evaluate_single_image(ground_truth_image, reconstructed_image, output_txt)
