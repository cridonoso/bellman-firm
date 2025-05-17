import os
import glob
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import imageio
import shutil

# --- Configuration ---
BASE_DATA_DIR = os.path.join('.', 'backup', 'p7')
FIGURES_DIR = os.path.join('.', 'figuras', 'p7')
TEMP_FRAMES_DIR = os.path.join('.', 'temp_frames_combined_gif_v4_filenames') # New temp dir
DPI = 300
GIF_FRAME_DURATION = 5

# UPDATED CASES_DEFINITION with correct file prefixes
CASES_DEFINITION = {
    # internal_key: {"file_prefix": "actual_filename_prefix_from_user", "plot_title": "Title for Subplot"}
    "ia": {"file_prefix": "1_fixed", "plot_title": "Case (i.a)"},
    "ib": {"file_prefix": "1_proportional", "plot_title": "Case (i.b)"},
    "iia": {"file_prefix": "2_fixed", "plot_title": "Case (ii.a)"},
    "iib": {"file_prefix": "2_proportional", "plot_title": "Case (ii.b)"},
}
PLOT_ORDER = ["ia", "ib", "iia", "iib"] # This order should align with your desired subplot layout

# --- Helper function to extract sigma ---
def extract_sigma_from_parent_dirname(filepath):
    parent_dir_name = ""
    try:
        parent_dir_name = os.path.basename(os.path.dirname(filepath))
        # Assumes sigma is the last part if underscores are used (e.g., "results_sigma_0.05" -> "0.05")
        # Or if directory is just "0.05", it should also work.
        sigma_str = parent_dir_name.split('_')[-1]
        return float(sigma_str)
    except Exception as e:
        # This print will now only show if extraction actually fails for a specific directory name
        print(f"DEBUG_SIGMA_EXTRACTION: Failed for parent_dir '{parent_dir_name}' from path '{filepath}'. Error: {e}")
        return None

# --- Main script function ---
def create_combined_subplot_gif_with_actual_filenames():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    if os.path.exists(TEMP_FRAMES_DIR):
        shutil.rmtree(TEMP_FRAMES_DIR)
    os.makedirs(TEMP_FRAMES_DIR)

    print("--- Pass 1: Loading data with detailed debugging ---")
    all_data_by_case = {case_key: [] for case_key in CASES_DEFINITION}
    # Create a reverse map from (new) filename prefix to case_key for easy lookup
    filename_to_case_key_map = {info['file_prefix']: key for key, info in CASES_DEFINITION.items()}

    glob_pattern = os.path.join(BASE_DATA_DIR, '*', '*.mat') # User-confirmed glob pattern
    print(f"Using glob pattern: '{glob_pattern}'")
    print(f"Expecting .mat filenames like: {[(info['file_prefix'] + '.mat') for info in CASES_DEFINITION.values()]}")
    print(f"Expecting sigma to be extracted from parent directory names (e.g., 'sigma_0.05' or '0.05').")
    print(f"DEBUG: filename_to_case_key_map (using new file prefixes): {filename_to_case_key_map}\n")

    found_files_count = 0
    processed_files_count = 0

    for file_path in glob.glob(glob_pattern):
        found_files_count += 1
        print(f"\nDEBUG_FILE_LOOP: Processing raw file path: '{file_path}'")

        parent_dir_for_sigma_extraction = os.path.basename(os.path.dirname(file_path))
        print(f"DEBUG_FILE_LOOP: Parent directory for sigma: '{parent_dir_for_sigma_extraction}'")
        sigma = extract_sigma_from_parent_dirname(file_path)
        if sigma is None:
            print(f"DEBUG_FILE_LOOP: Sigma extraction FAILED. Skipping file.")
            continue
        print(f"DEBUG_FILE_LOOP: Extracted sigma: {sigma}")

        filename_with_ext = os.path.basename(file_path)
        filename_prefix_parsed = filename_with_ext.replace('.mat', '').lower() # Keep .lower() for robustness
        print(f"DEBUG_FILE_LOOP: Filename prefix for case (parsed, lowercase): '{filename_prefix_parsed}'")
        
        case_key = filename_to_case_key_map.get(filename_prefix_parsed)
        if case_key is None:
            print(f"DEBUG_FILE_LOOP: Case key extraction FAILED for prefix '{filename_prefix_parsed}'. Defined prefixes in map: {list(filename_to_case_key_map.keys())}. Skipping file.")
            continue
        print(f"DEBUG_FILE_LOOP: Extracted case_key: '{case_key}' for file prefix '{filename_prefix_parsed}'")
            
        try:
            backup_s = loadmat(file_path)
            z_s = backup_s['z'].flatten()
            k_lower_s = backup_s['k_lower'].flatten()
            k_upper_s = backup_s['k_upper'].flatten()
            k_star_s = backup_s['k_star'][0]

            if not (len(z_s) == len(k_lower_s) == len(k_upper_s) == len(k_star_s) and len(z_s) > 0):
                print(f"DEBUG_FILE_LOOP: Data length mismatch or empty data in '{file_path}'. Skipping.")
                continue
            
            all_data_by_case[case_key].append({
                'sigma': sigma, 'z': z_s, 'k_lower': k_lower_s,
                'k_upper': k_upper_s, 'k_star': k_star_s,
            })
            processed_files_count +=1
            print(f"DEBUG_FILE_LOOP: Successfully processed and categorized '{file_path}'.")
        except KeyError as e:
            print(f"DEBUG_FILE_LOOP: KeyError loading data from '{file_path}' for case '{case_key}': {e}. Skipping.")
        except Exception as e:
            print(f"DEBUG_FILE_LOOP: Generic error loading data from '{file_path}' for case '{case_key}': {e}. Skipping.")

    print(f"\n--- Summary of Pass 1 ---")
    print(f"Total files found by glob: {found_files_count}")
    print(f"Successfully processed and categorized files: {processed_files_count}")

    # --- Calculate per-case axis limits ---
    global_axis_limits_by_case = {}
    any_data_loaded_for_any_case = False
    for case_key, case_data_list in all_data_by_case.items():
        if not case_data_list:
            global_axis_limits_by_case[case_key] = None
            print(f"Warning: No data loaded for {CASES_DEFINITION[case_key]['plot_title']}.")
            continue
        
        any_data_loaded_for_any_case = True
        case_data_list.sort(key=lambda item: item['sigma']) 

        min_z, max_z = float('inf'), float('-inf')
        min_k, max_k = float('inf'), float('-inf')
        for data_point in case_data_list: # Ensure data_point has z, k_lower, k_star, k_upper
            if data_point['z'].size > 0 : # Check for empty arrays before np.min/max
                min_z = min(min_z, np.min(data_point['z']))
                max_z = max(max_z, np.max(data_point['z']))
            if data_point['k_lower'].size > 0 and data_point['k_star'].size > 0:
                min_k = min(min_k, np.min(data_point['k_lower']), np.min(data_point['k_star']))
            if data_point['k_upper'].size > 0 and data_point['k_star'].size > 0:
                max_k = max(max_k, np.max(data_point['k_upper']), np.max(data_point['k_star']))
        
        if all(np.isfinite(val) for val in [min_z, max_z, min_k, max_k]): # check if all are finite numbers
             global_axis_limits_by_case[case_key] = {'min_z': min_z, 'max_z': max_z, 'min_k': min_k, 'max_k': max_k}
        else:
            global_axis_limits_by_case[case_key] = None # Mark as invalid
            print(f"Warning: Could not determine valid axis limits for {CASES_DEFINITION[case_key]['plot_title']}. Limits found: Z=({min_z},{max_z}), K=({min_k},{max_k})")


    if not any_data_loaded_for_any_case:
        print("CRITICAL: No data was successfully loaded for ANY case. Cannot create GIF. Please check file paths, names, .mat content, and review DEBUG messages.")
        if os.path.exists(TEMP_FRAMES_DIR): shutil.rmtree(TEMP_FRAMES_DIR)
        return

    all_sigmas_with_any_data = set()
    for case_key in CASES_DEFINITION: # Iterate through defined cases
        if case_key in all_data_by_case: # Check if case_key has any data
            for data_point in all_data_by_case[case_key]:
                all_sigmas_with_any_data.add(data_point['sigma'])
    
    if not all_sigmas_with_any_data:
        print("No sigma values associated with loaded data (all_data_by_case might be populated but 'sigma' values are missing or no cases had data). Cannot create GIF.")
        if os.path.exists(TEMP_FRAMES_DIR): shutil.rmtree(TEMP_FRAMES_DIR)
        return
        
    sorted_unique_sigmas = sorted(list(all_sigmas_with_any_data))
    print(f"\nData loaded. Will generate frames for sigma values: {sorted_unique_sigmas}")

    print("--- Pass 2: Generating combined frames for GIF ---")
    frame_files_combined = []

    for i, current_sigma in enumerate(sorted_unique_sigmas):
        fig, axes = plt.subplots(1, 4, figsize=(22, 5.5), sharey=False)
        fig.suptitle(f'Optimal Policy \\& Inaction Band for $\\sigma = {current_sigma:.2f}$', fontsize=16)

        for ax_idx, case_key in enumerate(PLOT_ORDER):
            ax = axes[ax_idx]
            case_info = CASES_DEFINITION[case_key]
            ax.set_title(case_info['plot_title'])

            data_for_plot = None
            if case_key in all_data_by_case and isinstance(all_data_by_case[case_key], list):
                for data_point in all_data_by_case[case_key]:
                    if abs(data_point['sigma'] - current_sigma) < 1e-6: # Floating point comparison
                        data_for_plot = data_point
                        break
            
            ax.set_xlabel('Productivity (z)')
            if ax_idx == 0: ax.set_ylabel('Capital (k)')

            if data_for_plot:
                data = data_for_plot
                sort_indices = np.argsort(data['z'])
                z_sorted, k_star_sorted = data['z'][sort_indices], data['k_star'][sort_indices]
                k_lower_sorted, k_upper_sorted = data['k_lower'][sort_indices], data['k_upper'][sort_indices]

                ax.plot(z_sorted, k_star_sorted, label='Optimal $k^*(z)$', color='black', linewidth=1.5)
                ax.fill_between(z_sorted, k_lower_sorted, k_upper_sorted, alpha=0.3, label='Inaction Band', color='deepskyblue')
                
                # MODIFIED LINE: Specify legend location
                ax.legend(loc='upper left', fontsize='x-small') 
            else:
                ax.text(0.5, 0.5, 'No data\nfor this $\\sigma$', ha='center', va='center', transform=ax.transAxes, fontsize=10, color='grey')

            case_limits = global_axis_limits_by_case.get(case_key)
            if case_limits: 
                ax.set_xlim(case_limits['min_z'], case_limits['max_z'])
                ax.set_ylim(case_limits['min_k'], case_limits['max_k'])
            else:
                 print(f"DEBUG_PLOT: No valid axis limits for case {case_key} for sigma {current_sigma}. Autoscale will be used by Matplotlib.")
            ax.grid(True, linestyle=':', alpha=0.6)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        frame_filename = os.path.join(TEMP_FRAMES_DIR, f"frame_sigma_{current_sigma:.3f}_{i:03d}.png")
        try:
            plt.savefig(frame_filename, dpi=DPI)
            frame_files_combined.append(frame_filename)
        except Exception as e:
            print(f"    Error saving frame '{frame_filename}': {e}")
        finally:
            plt.close(fig)
        
        if (i + 1) % 1 == 0: print(f"  Generated frame for sigma = {current_sigma:.2f} ({i+1}/{len(sorted_unique_sigmas)})")


    # --- Create a single GIF ---
    if frame_files_combined:
        gif_output_path_combined = os.path.join(FIGURES_DIR, "tc_all_cases_subplot_vs_sigma.gif")
        print(f"\n--- Creating combined GIF ({len(frame_files_combined)} frames) ---")

        # << AÑADE ESTA LÍNEA DE DEPURACIÓN >>
        print(f"DEBUG_GIF_CREATION: Usando GIF_FRAME_DURATION = {GIF_FRAME_DURATION} (tipo: {type(GIF_FRAME_DURATION)}) segundos por cuadro.")

        try:
            with imageio.get_writer(gif_output_path_combined, mode='I', duration=0.9, loop=0) as writer:
                for frame_filename in frame_files_combined:
                    image = imageio.imread(frame_filename)
                    writer.append_data(image)
            print(f"🎉 Combined GIF successfully saved to '{gif_output_path_combined}'")
        except Exception as e:
            print(f"  Error creating combined GIF: {e}")
    else:
        print("No frames were generated. Combined GIF not created.")