import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim
import argparse
import re


# --- Helper Functions ---
def to_grayscale(image):
    if isinstance(image, np.ndarray):
        if image.ndim == 3 and image.shape[-1] == 1:  # (H, W, 1)
            return np.squeeze(image, axis=-1)
        elif image.ndim == 3 and image.shape[-1] > 1:  # If RGB or RGBA
            # Simple grayscale conversion: average or luminosity. Using mean for simplicity.
            # For more accurate, use weighted: 0.299*R + 0.587*G + 0.114*B
            return np.mean(image, axis=-1) if image.shape[-1] in [3, 4] else image[..., 0]
        elif image.ndim == 4 and image.shape[0] == 1:  # (1, H, W, C) or (1, H, W)
            squeezed_image = np.squeeze(image, axis=0)
            return to_grayscale(squeezed_image)
        elif image.ndim != 2:
            print(f"Warning: to_grayscale received unexpected shape {image.shape}. Trying to squeeze.")
            try:
                squeezed = np.squeeze(image)
                if squeezed.ndim == 2:
                    return squeezed
                else:
                    print(f"Still not 2D after squeeze: {squeezed.shape}")
                    return image  # Fallback
            except:
                return image  # Fallback
    return image


def dynamic_win_size(image, default=7):
    if not isinstance(image, np.ndarray) or image.ndim != 2:
        # print(f"Warning: dynamic_win_size expected 2D numpy array, got {type(image)} shape {getattr(image, 'shape', 'N/A')}")
        return None
    h, w = image.shape
    min_dim = min(h, w)
    if min_dim < 3:
        return None
    win = min(default, min_dim)
    # Ensure win_size is odd and less than or equal to min_dim
    if win > min_dim: win = min_dim
    if win % 2 == 0:
        win -= 1
    return win if win >= 3 else None


# --- Mask Configuration (YOUR ACTUALS from testmask.npy) ---
ordered_mask_sizes = [  # Index 0 = mask_0/mask_01, Index 1 = mask_1/mask_02, ...
    371, 1338, 688, 84, 171,
    54, 1329, 3119, 182, 180
]
num_masks_total = len(ordered_mask_sizes)


def get_mask_group(mask_pixel_area):
    if mask_pixel_area in {54, 84}:
        return "G1"
    elif mask_pixel_area in {171, 180}:
        return "G2"
    elif mask_pixel_area in {182, 371}:
        return "G3"
    elif mask_pixel_area in {688, 1329}:
        return "G4"
    elif mask_pixel_area in {1338, 3119}:
        return "G5"
    else:
        print(f"Critical Warning: Mask area {mask_pixel_area} not in predefined groups.")
        return None


# --- End Mask Configuration ---

def get_mask_index_from_prefix_sequential(filename_prefix, num_masks=10):
    try:
        sequential_id = int(filename_prefix)
        if sequential_id <= 0: return None
        return (sequential_id - 1) % num_masks
    except ValueError:
        return None


def get_mask_index_from_mask_filename(mask_filename):
    match = re.search(r'mask_(\d+)\.png', mask_filename)
    if match:
        mask_num = int(match.group(1))
        if 1 <= mask_num <= num_masks_total: return mask_num - 1
    return None


def evaluate_metrics_by_size(
        results_base_dir, model_name,
        model_subdir_name=None, gt_subdir_name_flat="gt/image",
        is_nested_structure=False, model_output_folder_nested=None, gt_folder_nested="Xgt",
        output_dir="./metric_reports",
        force_hu_evaluation_for_image_files=False,
        hu_min_config=-1000, hu_max_config=3000
):
    metrics_by_group = {f"G{i + 1}": {"psnr_img": [], "ssim_img": [], "psnr_hu": [], "ssim_hu": [], "count": 0} for i in
                        range(5)}
    table_rows_individual = []

    # Determine if dedicated HU files are expected and paths exist
    # This flag primarily tracks if we *found and can process* dedicated _hu.png files.
    # `force_hu_evaluation_for_image_files` is a separate instruction for _image.png files.
    attempt_dedicated_hu_processing = True

    # --- Path and Filename Logic ---
    if is_nested_structure:
        print(f"Using NESTED structure for {model_name}")
        gt_parent_dir = os.path.join(results_base_dir, gt_folder_nested)
        net_parent_dir = os.path.join(results_base_dir, model_output_folder_nested)
        attempt_dedicated_hu_processing = False  # Nested InDuDoNet+ structure has no HU files specified

        if not os.path.isdir(gt_parent_dir): print(f"Error: Nested GT parent dir missing: {gt_parent_dir}"); return
        if not os.path.isdir(net_parent_dir): print(f"Error: Nested Net parent dir missing: {net_parent_dir}"); return

        base_image_folders = sorted([d for d in os.listdir(gt_parent_dir) if
                                     os.path.isdir(os.path.join(gt_parent_dir, d)) and d.startswith("img_")])
        file_iterator = [{"base_folder": bif, "mask_file": f"mask_{i + 1:02d}.png"} for bif in base_image_folders for i
                         in range(num_masks_total)]
        loop_type = "nested"
        if not base_image_folders: print(f"Error: No 'img_XXX' folders in {gt_parent_dir}"); return
    else:  # Flat structure
        print(f"Using FLAT structure for {model_name}")
        gt_image_dir = os.path.join(results_base_dir, gt_subdir_name_flat)

        if model_name.lower() in ["osc", "oscplus"]:
            net_image_dir = os.path.join(results_base_dir, model_subdir_name)
            attempt_dedicated_hu_processing = False
        else:
            net_image_dir = os.path.join(results_base_dir, model_subdir_name, "image")
            gt_hu_dir_path = os.path.join(results_base_dir, "gt", "hu")
            net_hu_dir_path = os.path.join(results_base_dir, model_subdir_name, "hu")
            if not (os.path.isdir(gt_hu_dir_path) and os.path.isdir(net_hu_dir_path)):
                print(
                    f"Warning: Dedicated HU directories not found for {model_name}. Will not process separate HU files.")
                attempt_dedicated_hu_processing = False

        if not os.path.isdir(gt_image_dir): print(f"Error: Flat GT image dir missing: {gt_image_dir}"); return
        if not os.path.isdir(net_image_dir): print(f"Error: Flat Net image dir missing: {net_image_dir}"); return

        all_gt_filenames_unsorted = [f for f in os.listdir(gt_image_dir) if
                                     f.endswith("_gt_image.png") and f.replace("_gt_image.png", "").isdigit()]
        if not all_gt_filenames_unsorted: print(f"Error: No suitable '*_gt_image.png' files in {gt_image_dir}"); return
        file_iterator = sorted(all_gt_filenames_unsorted, key=lambda x: int(x.replace("_gt_image.png", "")))
        loop_type = "flat"

    individual_header_cols = [f"{'Identifier':<30}", f"{'MaskIdx':>8}", f"{'MaskArea':>10}", f"{'Group':>8}",
                              f"{'PSNR(img)':>12}", f"{'SSIM(img)':>12}"]
    # Add HU columns to individual log only if we are *potentially* processing them
    # (either dedicated HU files or forced HU scaling on image files)
    if attempt_dedicated_hu_processing or force_hu_evaluation_for_image_files:
        individual_header_cols.extend([f"{'PSNR(hu)':>12}", f"{'SSIM(hu)':>12}"])
    individual_table_header = "".join(individual_header_cols) + "\n" + "-" * len("".join(individual_header_cols))
    processed_files_count = 0

    for item in file_iterator:
        mask_index = -1;
        display_identifier = ""
        gt_img_path = "";
        net_img_path = "";
        gt_hu_path = "";
        net_hu_path = ""

        if loop_type == "flat":
            gt_filename_flat = item
            prefix = gt_filename_flat.replace("_gt_image.png", "")
            display_identifier = gt_filename_flat
            mask_index = get_mask_index_from_prefix_sequential(prefix, num_masks_total)

            gt_img_path = os.path.join(gt_image_dir, gt_filename_flat)
            if model_name.lower() == "dicdnet":
                net_img_filename = f"{prefix}_dicdnet_image.png"
            elif model_name.lower() == "osc":
                net_img_filename = f"{prefix}_oscnet_image.png"
            elif model_name.lower() == "oscplus":
                net_img_filename = f"{prefix}_oscnetplus_image.png"
            else:
                net_img_filename = f"{prefix}_{model_name.lower()}_image.png"
            net_img_path = os.path.join(net_image_dir, net_img_filename)

            if attempt_dedicated_hu_processing:
                gt_hu_path = os.path.join(gt_hu_dir_path, f"{prefix}_gt_hu.png")
                if model_name.lower() == "dicdnet":
                    net_hu_filename = f"{prefix}_dicdnet_hu.png"
                else:
                    net_hu_filename = f"{prefix}_{model_name.lower()}_hu.png"
                net_hu_path = os.path.join(net_hu_dir_path, net_hu_filename)
        elif loop_type == "nested":
            base_folder = item["base_folder"];
            mask_file = item["mask_file"]
            display_identifier = f"{base_folder}/{mask_file}"
            mask_index = get_mask_index_from_mask_filename(mask_file)
            gt_img_path = os.path.join(gt_parent_dir, base_folder, mask_file)
            net_img_path = os.path.join(net_parent_dir, base_folder, mask_file)

        if mask_index is None or not (0 <= mask_index < len(ordered_mask_sizes)): continue
        current_mask_pixel_area = ordered_mask_sizes[mask_index]
        group_key = get_mask_group(current_mask_pixel_area)
        if not group_key: continue
        if not (os.path.exists(gt_img_path) and os.path.exists(net_img_path)): continue

        try:
            gt_img_raw = plt.imread(gt_img_path).astype(np.float64)
            net_img_raw = plt.imread(net_img_path).astype(np.float64)
        except Exception:
            continue

        gt_img_for_visual_eval = to_grayscale(np.squeeze(gt_img_raw))
        net_img_for_visual_eval = to_grayscale(np.squeeze(net_img_raw))
        if gt_img_for_visual_eval.shape != net_img_for_visual_eval.shape: continue

        # --- Image Metric Calculation (Visual or Forced HU) ---
        img_calc_gt = gt_img_for_visual_eval.copy()
        img_calc_net = net_img_for_visual_eval.copy()
        img_data_range_psnr = img_calc_gt.max() - img_calc_gt.min() if img_calc_gt.max() > img_calc_gt.min() else 1.0
        img_data_range_ssim = img_data_range_psnr  # Use same for SSIM base

        if force_hu_evaluation_for_image_files and not attempt_dedicated_hu_processing:
            temp_gt_max = gt_img_for_visual_eval.max()
            scaled_to_hu = False
            if temp_gt_max <= 1.0 + 1e-3 and gt_img_for_visual_eval.min() >= 0.0 - 1e-3:
                img_calc_gt = gt_img_for_visual_eval * (hu_max_config - hu_min_config) + hu_min_config
                img_calc_net = net_img_for_visual_eval * (hu_max_config - hu_min_config) + hu_min_config
                scaled_to_hu = True
            elif temp_gt_max <= 255.0 + 1e-3 and gt_img_for_visual_eval.min() >= 0.0 - 1e-3:
                img_calc_gt = gt_img_for_visual_eval * (hu_max_config - hu_min_config) / 255.0 + hu_min_config
                img_calc_net = net_img_for_visual_eval * (hu_max_config - hu_min_config) / 255.0 + hu_min_config
                scaled_to_hu = True
            if scaled_to_hu:
                img_data_range_psnr = hu_max_config - hu_min_config
                img_data_range_ssim = hu_max_config - hu_min_config

        psnr_img_val = np.nan
        if img_data_range_psnr == 0 and np.allclose(img_calc_gt, img_calc_net):
            psnr_img_val = np.inf
        elif img_data_range_psnr > 0:
            psnr_img_val = sk_psnr(img_calc_gt, img_calc_net, data_range=img_data_range_psnr)

        ssim_img_val = np.nan
        win_size_img = dynamic_win_size(img_calc_gt)
        if win_size_img is not None:
            try:
                ssim_img_val = sk_ssim(img_calc_gt, img_calc_net, data_range=img_data_range_ssim, win_size=win_size_img,
                                       channel_axis=None, gaussian_weights=True, use_sample_covariance=False, K1=0.01,
                                       K2=0.03)
            except ValueError:
                pass

        # --- Dedicated HU Metric Calculation ---
        psnr_hu_val, ssim_hu_val = np.nan, np.nan
        if attempt_dedicated_hu_processing and os.path.exists(gt_hu_path) and os.path.exists(net_hu_path):
            try:
                gt_hu_img_raw = plt.imread(gt_hu_path).astype(np.float64)
                net_hu_img_raw = plt.imread(net_hu_path).astype(np.float64)
                gt_hu_img = to_grayscale(np.squeeze(gt_hu_img_raw))
                net_hu_img = to_grayscale(np.squeeze(net_hu_img_raw))

                if gt_hu_img.shape == net_hu_img.shape:
                    data_range_actual_hu = hu_max_config - hu_min_config
                    if data_range_actual_hu == 0 and np.allclose(gt_hu_img, net_hu_img):
                        psnr_hu_val = np.inf
                    elif data_range_actual_hu > 0:
                        psnr_hu_val = sk_psnr(gt_hu_img, net_hu_img, data_range=data_range_actual_hu)
                    win_size_hu = dynamic_win_size(gt_hu_img)
                    if win_size_hu is not None:
                        try:
                            ssim_hu_val = sk_ssim(gt_hu_img, net_hu_img, data_range=data_range_actual_hu,
                                                  win_size=win_size_hu, channel_axis=None, gaussian_weights=True,
                                                  use_sample_covariance=False, K1=0.01, K2=0.03)
                        except ValueError:
                            pass
            except Exception:
                pass  # Error loading/processing HU

        if not (np.isnan(psnr_img_val) or np.isnan(ssim_img_val)):
            metrics_by_group[group_key]["psnr_img"].append(psnr_img_val)
            metrics_by_group[group_key]["ssim_img"].append(ssim_img_val)
            metrics_by_group[group_key]["psnr_hu"].append(psnr_hu_val)  # Will be NaN if not processed
            metrics_by_group[group_key]["ssim_hu"].append(ssim_hu_val)  # Will be NaN if not processed
            metrics_by_group[group_key]["count"] += 1
            processed_files_count += 1

        row_parts = [f"{display_identifier:<30}", f"{mask_index:>8}", f"{current_mask_pixel_area:>10}",
                     f"{group_key:>8}",
                     f"{psnr_img_val:>12.2f}", f"{ssim_img_val:>12.4f}"]
        if attempt_dedicated_hu_processing or force_hu_evaluation_for_image_files:
            row_parts.extend([f"{psnr_hu_val:>12.2f}", f"{ssim_hu_val:>12.4f}"])
        table_rows_individual.append("".join(row_parts))

    # --- Summary Report Generation ---
    report_lines = []
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    report_lines.append(f"Model: {model_name} - Image Quality Evaluation by Metal Size")
    report_lines.append(f"Date: {timestamp}")
    report_lines.append(f"Results Base Directory: {os.path.abspath(results_base_dir)}")
    if is_nested_structure:
        report_lines.append(f"Structure: Nested (GT: {gt_folder_nested}, ModelOut: {model_output_folder_nested})")
    else:
        report_lines.append(f"Structure: Flat (GT: {gt_subdir_name_flat}, ModelSubdir: {model_subdir_name})")
    report_lines.append(f"Forced HU eval for image files: {force_hu_evaluation_for_image_files}")
    if force_hu_evaluation_for_image_files or attempt_dedicated_hu_processing: report_lines.append(
        f"HU eval range used: [{hu_min_config}, {hu_max_config}]")
    report_lines.append(f"Total files iterated: {len(file_iterator)}")
    report_lines.append(f"Total files with metrics collected: {processed_files_count}")

    report_lines.append("\n--- Metrics by Metal Size Group (mean ± std) ---")
    header_cols_summary = [f"{'Group Label':<28}", f"{'Count':>7}", f"{'PSNR(img)':>20}", f"{'SSIM(img)':>22}"]
    # Determine if HU metrics should be in summary table based on if they were possibly processed
    show_hu_in_summary = attempt_dedicated_hu_processing or force_hu_evaluation_for_image_files
    if show_hu_in_summary:
        header_cols_summary.extend([f"{'PSNR(hu)':>20}", f"{'SSIM(hu)':>22}"])
    header_line1 = "".join(header_cols_summary)
    header_line2 = "-" * len(header_line1)
    report_lines.append(header_line1);
    report_lines.append(header_line2)

    group_display_config = [
        ("G5", "Large       ({1338, 3119}px)"), ("G4", "Large-Med   ({688, 1329}px)"),
        ("G3", "Medium      ({182, 371}px)"), ("G2", "Med-Small   ({171, 180}px)"),
        ("G1", "Small       ({54, 84}px)")
    ]
    all_psnr_img, all_ssim_img, all_psnr_hu, all_ssim_hu = [], [], [], []

    for group_key_internal, group_label_display in group_display_config:
        data = metrics_by_group[group_key_internal];
        count = data["count"]
        all_psnr_img.extend(x for x in data["psnr_img"] if not np.isnan(x))
        all_ssim_img.extend(x for x in data["ssim_img"] if not np.isnan(x))
        if show_hu_in_summary:
            all_psnr_hu.extend(x for x in data["psnr_hu"] if not np.isnan(x))
            all_ssim_hu.extend(x for x in data["ssim_hu"] if not np.isnan(x))

        line_parts = [f"{group_label_display:<28}", f"{count:>7}"]
        if count == 0:
            line_parts.extend([f"{'N/A':>20}", f"{'N/A':>22}"])
            if show_hu_in_summary: line_parts.extend([f"{'N/A':>20}", f"{'N/A':>22}"])
        else:
            psnr_img_m, psnr_img_s = np.nanmean(data["psnr_img"]), np.nanstd(data["psnr_img"])
            ssim_img_m, ssim_img_s = np.nanmean(data["ssim_img"]), np.nanstd(data["ssim_img"])
            line_parts.extend(
                [f"{psnr_img_m:>12.2f} ± {psnr_img_s:<5.2f}", f"{ssim_img_m:>14.4f} ± {ssim_img_s:<5.4f}"])
            if show_hu_in_summary:
                if any(not np.isnan(x) for x in data["psnr_hu"]):
                    psnr_hu_m, psnr_hu_s = np.nanmean(data["psnr_hu"]), np.nanstd(data["psnr_hu"])
                    ssim_hu_m, ssim_hu_s = np.nanmean(data["ssim_hu"]), np.nanstd(data["ssim_hu"])
                    line_parts.extend(
                        [f"{psnr_hu_m:>12.2f} ± {psnr_hu_s:<5.2f}", f"{ssim_hu_m:>14.4f} ± {ssim_hu_s:<5.4f}"])
                else:
                    line_parts.extend([f"{'N/A':>20}", f"{'N/A':>22}"])
        report_lines.append("".join(line_parts))
    report_lines.append(header_line2)

    report_lines.append("\n--- Overall Summary (mean ± std across all successfully processed files) ---")
    if all_psnr_img:
        report_lines.append(
            f"PSNR(img): {np.nanmean(all_psnr_img):.2f} ± {np.nanstd(all_psnr_img):.2f} (from {len(all_psnr_img)} images)")
        report_lines.append(f"SSIM(img): {np.nanmean(all_ssim_img):.4f} ± {np.nanstd(all_ssim_img):.4f}")
        if show_hu_in_summary:
            if all_psnr_hu:
                report_lines.append(f"PSNR(hu):  {np.nanmean(all_psnr_hu):.2f} ± {np.nanstd(all_psnr_hu):.2f}")
                report_lines.append(f"SSIM(hu):  {np.nanmean(all_ssim_hu):.4f} ± {np.nanstd(all_ssim_hu):.4f}")
            else:
                report_lines.extend(
                    ["PSNR(hu): N/A (no valid HU data processed)", "SSIM(hu): N/A (no valid HU data processed)"])
    else:
        report_lines.append("No data for overall summary.")

    report_lines.append("\n--- Individual File Metrics Log ---")
    report_lines.append(individual_table_header)
    report_lines.extend(table_rows_individual)

    final_report_str = "\n".join(report_lines)
    # print(final_report_str) # Optional: print to console

    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(output_dir, f"{model_name.lower().replace('+', 'plus')}_metrics_by_size.txt")
    with open(output_filename, 'w') as f:
        f.write(final_report_str)
    print(f"\nDetailed report saved to: {os.path.abspath(output_filename)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate Model image quality metrics by metal size, supporting multiple file structures.")
    parser.add_argument("--results_base_dir", type=str, required=True, help="Base directory for results.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model for reporting.")
    parser.add_argument("--output_dir", type=str, default="./metric_reports",
                        help="Directory for saving metrics report.")

    parser.add_argument("--model_subdir_name", type=str, default=None,
                        help="[FLAT only] Model's output subdir (e.g., 'DICDNet', 'osc'). If same as model_name, can be omitted.")
    parser.add_argument("--gt_subdir_name_flat", type=str, default="gt/image",
                        help="[FLAT only] GT images path relative to results_base_dir.")

    parser.add_argument("--is_nested_structure", action='store_true',
                        help="Flag for nested file structure (InDuDoNet+ like).")
    parser.add_argument("--model_output_folder_nested", type=str, default=None,
                        help="[NESTED only] Model's output folder (e.g., 'X').")
    parser.add_argument("--gt_folder_nested", type=str, default="Xgt", help="[NESTED only] GT folder (e.g., 'Xgt').")

    parser.add_argument("--force_hu_evaluation_for_image_files", action='store_true',
                        help="Scale _image.png to HU range for metrics if no dedicated _hu.png are processed.")
    parser.add_argument("--hu_min", type=int, default=-1000, help="Min HU for scaling.")
    parser.add_argument("--hu_max", type=int, default=3000, help="Max HU for scaling.")

    args = parser.parse_args()

    if not args.is_nested_structure and args.model_subdir_name is None:
        args.model_subdir_name = args.model_name
    if not args.is_nested_structure and args.model_name.lower() in ["osc", "oscplus"]:
        args.gt_subdir_name_flat = "gt"

    evaluate_metrics_by_size(
        results_base_dir=args.results_base_dir, model_name=args.model_name,
        model_subdir_name=args.model_subdir_name, gt_subdir_name_flat=args.gt_subdir_name_flat,
        is_nested_structure=args.is_nested_structure,
        model_output_folder_nested=args.model_output_folder_nested, gt_folder_nested=args.gt_folder_nested,
        output_dir=args.output_dir,
        force_hu_evaluation_for_image_files=args.force_hu_evaluation_for_image_files,
        hu_min_config=args.hu_min, hu_max_config=args.hu_max
    )