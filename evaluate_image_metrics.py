import os
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
from datetime import datetime


def evaluate_image_quality(results_dir, output_file="image_quality_metrics.txt"):
    # CT Hounsfield Unit (HU) range parameters - ADJUST THESE!
    hu_min = -1000  # Minimum HU value in original data
    hu_max = 3000  # Maximum HU value in original data

    psnr_values = []
    ssim_values = []

    xgt_dir = os.path.join(results_dir, 'Xgt')
    x_dir = os.path.join(results_dir, 'X')

    # Results table setup
    table_header = f"{'Image':<15}{'Mask':<10}{'PSNR (dB)':<15}{'SSIM':<15}\n"
    table_header += "-" * 55 + "\n"
    table_rows = []

    for img_folder in os.listdir(xgt_dir):
        xgt_img_path = os.path.join(xgt_dir, img_folder)
        x_img_path = os.path.join(x_dir, img_folder)

        if not os.path.isdir(x_img_path):
            continue

        for mask_idx in range(1, 11):
            try:
                # Load images
                gt_file = os.path.join(xgt_img_path, f"mask_{mask_idx:02d}.png")
                recon_file = os.path.join(x_img_path, f"mask_{mask_idx:02d}.png")

                gt_img = np.squeeze(plt.imread(gt_file))
                recon_img = np.squeeze(plt.imread(recon_file))

                # Debug: Verify loaded image ranges
                print(f"\n{img_folder}/mask_{mask_idx:02d}:")
                print(f"GT range: [{gt_img.min():.3f}, {gt_img.max():.3f}]")
                print(f"Recon range: [{recon_img.min():.3f}, {recon_img.max():.3f}]")

                # Convert from PNG to HU scale
                if gt_img.max() <= 1.0:  # Normalized to [0, 1]
                    gt_hu = gt_img * (hu_max - hu_min) + hu_min
                    recon_hu = recon_img * (hu_max - hu_min) + hu_min
                else:  # Assume [0, 255]
                    gt_hu = gt_img * (hu_max - hu_min) / 255 + hu_min
                    recon_hu = recon_img * (hu_max - hu_min) / 255 + hu_min

                # Dynamic window size for SSIM
                min_dim = min(gt_hu.shape)
                win_size = min(7, min_dim)
                win_size = win_size - 1 if win_size % 2 == 0 else win_size

                # Calculate metrics
                psnr_val = psnr(gt_hu, recon_hu, data_range=hu_max - hu_min)
                ssim_val = ssim(gt_hu, recon_hu,
                                data_range=hu_max - hu_min,
                                win_size=win_size,
                                channel_axis=None)

                psnr_values.append(psnr_val)
                ssim_values.append(ssim_val)

                # Add to table
                table_rows.append(
                    f"{img_folder:<15}{mask_idx:02d}{psnr_val:>15.2f}{ssim_val:>15.4f}"
                )

            except Exception as e:
                print(f"Error processing {img_folder}/mask_{mask_idx:02d}: {str(e)}")
                continue

    # Generate report
    summary = (
        f"\n{'Metric':<15}{'Mean ± Std':<25}\n"
        f"{'-' * 40}\n"
        f"{'PSNR (dB)':<15}{np.mean(psnr_values):.2f} ± {np.std(psnr_values):.2f}\n"
        f"{'SSIM':<15}{np.mean(ssim_values):.4f} ± {np.std(ssim_values):.4f}"
    )

    full_report = (
            f"Image Quality Evaluation Report\n"
            f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Directory: {results_dir}\n"
            f"HU Range: [{hu_min} {hu_max}]\n\n"
            f"{table_header}"
            + "\n".join(table_rows) +
            f"\n\n{summary}"
    )

    # Save and print
    print(full_report)
    with open(output_file, 'w') as f:
        f.write(full_report)
    print(f"\nReport saved to: {os.path.abspath(output_file)}")


if __name__ == "__main__":
    evaluate_image_quality("./results/deeplesion")