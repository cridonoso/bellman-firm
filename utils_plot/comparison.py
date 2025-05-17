import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from glob import glob
import matplotlib as mpl
import os
import re



def k_star_comparison(problem_main_name = 'p5',  problem_shadow_name = 'p3'):
    # --- Cargar Datos del Problema de Referencia (p3) ---
    folder_shadow = './backup/' + problem_shadow_name
    policy_paths_shadow = sorted(glob(os.path.join(folder_shadow, '*.mat')))
    z_shadow_all_cases, width_shadow_all_cases = [], []
    # (El resto del código de carga de datos de sombra permanece igual)
    if len(policy_paths_shadow) != 4:
        print(f"Advertencia: Se esperaban 4 archivos .mat para el problema {problem_shadow_name}, se encontraron {len(policy_paths_shadow)}")
        for _ in range(4):
            z_shadow_all_cases.append(None)
            width_shadow_all_cases.append(None)
    else:
        for i in range(len(policy_paths_shadow)): # Asegurar que solo procesamos los archivos encontrados
            try:
                backup_s = loadmat(policy_paths_shadow[i])
                z_s = backup_s['z']
                k_lower_s = backup_s['k_lower']
                k_upper_s = backup_s['k_upper']
                width_s = k_upper_s[:, 0] - k_lower_s[:, 0]
                z_shadow_all_cases.append(z_s[:, 0])
                width_shadow_all_cases.append(width_s)
            except Exception as e:
                print(f"Error cargando datos de sombra para {policy_paths_shadow[i]}: {e}")
                z_shadow_all_cases.append(None)
                width_shadow_all_cases.append(None)

    # --- Cargar Datos del Problema Principal (p5) y Graficar ---
    folder_main = './backup/' + problem_main_name
    policy_paths_main = sorted(glob(os.path.join(folder_main, '*.mat')))

    fig, axes = plt.subplots(1, 4, sharex=True, sharey=True, figsize=(13, 2.5), gridspec_kw={'wspace':0.05}, dpi=120)
    titles = ['Caso (i.a)', 'Caso (i.b)', 'Caso (ii.a)', 'Caso (ii.b)']

    if problem_main_name == 'p5' and problem_shadow_name == 'p3':
        label_shadow = r'Ancho ($F_0, P_0$)'
        label_main = r'Ancho ($F_1, P_1$)'
    else:
        label_shadow = f'Ancho ({problem_shadow_name})'
        label_main = f'Ancho ({problem_main_name})'

    for i, path_main in enumerate(policy_paths_main):
        if i >= 4: break
        try:
            backup_m = loadmat(path_main)
            z_m_data = backup_m['z'][:,0]
            width_m_data = backup_m['k_upper'][:, 0] - backup_m['k_lower'][:, 0]

            axes[i].plot(z_m_data, width_m_data, color='darkblue', linestyle='-', linewidth=1.5, label=label_main, zorder=2)

            z_s_current, width_s_current = None, None
            if i < len(z_shadow_all_cases) and i < len(width_shadow_all_cases):
                z_s_current = z_shadow_all_cases[i]
                width_s_current = width_shadow_all_cases[i]

            if z_s_current is not None and width_s_current is not None:
                axes[i].plot(z_s_current, width_s_current, color='k', linestyle='--', linewidth=1.5, label=label_shadow, zorder=1)

                if len(z_m_data) > 0 and len(z_s_current) > 0:
                    common_z_min = max(np.min(z_m_data), np.min(z_s_current))
                    common_z_max = min(np.max(z_m_data), np.max(z_s_current))

                    if common_z_max > common_z_min:
                        num_common_points = 200
                        common_z_grid = np.linspace(common_z_min, common_z_max, num=num_common_points)
                        interp_width_m = np.interp(common_z_grid, z_m_data, width_m_data)
                        interp_width_s = np.interp(common_z_grid, z_s_current, width_s_current)
                        diff_array = interp_width_m - interp_width_s

                        min_diff_val, max_diff_val = np.min(diff_array), np.max(diff_array)
                        idx_min_diff, idx_max_diff = np.argmin(diff_array), np.argmax(diff_array)
                        z_at_min_diff, z_at_max_diff = common_z_grid[idx_min_diff], common_z_grid[idx_max_diff]
                        
                        y_offset_text = 0.035 # Aumentado ligeramente para más separación
                        calc_z_range = common_z_max - common_z_min

                        # --- Texto para MAXIMA diferencia ---
                        z_coord_max = z_at_max_diff
                        ha_max = 'center'
                        if calc_z_range > 1e-6: # Evitar división por cero
                            relative_pos_max = (z_coord_max - common_z_min) / calc_z_range
                            if relative_pos_max < 0.20: ha_max = 'left'  # Si está en el 20% izquierdo
                            elif relative_pos_max > 0.80: ha_max = 'right' # Si está en el 20% derecho
                        
                        y_base_max = max(interp_width_m[idx_max_diff], interp_width_s[idx_max_diff])
                        y_pos_max_text = y_base_max + y_offset_text
                        va_max = 'bottom'
                        
                        axes[i].text(z_coord_max, y_pos_max_text, r"Max $\Delta$:"+f"{max_diff_val:+.2f}",
                                    horizontalalignment=ha_max, verticalalignment=va_max,
                                    fontsize=8, color='darkblue', weight='bold',
                                    bbox=dict(facecolor='white', alpha=0.85, pad=0.2, edgecolor='lightgray', boxstyle='round,pad=0.25'))

                        # --- Texto para MINIMA diferencia ---
                        z_coord_min = z_at_min_diff
                        ha_min = 'center'
                        if calc_z_range > 1e-6:
                            relative_pos_min = (z_coord_min - common_z_min) / calc_z_range
                            if relative_pos_min < 0.20: ha_min = 'left'
                            elif relative_pos_min > 0.80: ha_min = 'right'

                        y_base_min = max(interp_width_m[idx_min_diff], interp_width_s[idx_min_diff])
                        y_pos_min_text_initial = y_base_min + y_offset_text
                        va_min_initial = 'bottom'

                        # Ajuste para evitar superposición vertical
                        y_pos_min_final = y_pos_min_text_initial
                        va_min_final = va_min_initial

                        # Heurística simple para superposición: si las z son cercanas Y las y originales están cerca
                        if abs(z_coord_max - z_coord_min) < calc_z_range * 0.15: 
                            # Evaluar si las cajas de texto podrían solaparse horizontalmente debido a sus 'ha'
                            # Esta es una simplificación. El solapamiento real depende del ancho del texto.
                            # Si y_pos_max_text y y_pos_min_text_initial están cerca:
                            if abs(y_pos_max_text - y_pos_min_text_initial) < y_offset_text * 1.5: # Si las alturas son similares
                                # Mover el texto de MINIMA diferencia debajo de las líneas
                                y_base_min_alt = min(interp_width_m[idx_min_diff], interp_width_s[idx_min_diff])
                                y_pos_min_final = y_base_min_alt - y_offset_text
                                va_min_final = 'top' # Ajustar alineación vertical
                        
                        axes[i].text(z_coord_min, y_pos_min_final, r"Min $\Delta$"+f": {min_diff_val:+.2f}",
                                    horizontalalignment=ha_min, verticalalignment=va_min_final,
                                    fontsize=8, color='indigo', weight='bold',
                                    bbox=dict(facecolor='white', alpha=0.85, pad=0.2, edgecolor='lightgray', boxstyle='round,pad=0.25'))
            
            axes[i].set_title(titles[i])
            axes[i].grid(True, linestyle=':', alpha=0.7)

        except Exception as e:
            print(f"Error procesando el archivo {path_main} para el subgráfico {titles[i]}: {e}")
            axes[i].set_title(f"{titles[i]}\n(Error cargando datos)")
            axes[i].grid(True, linestyle=':', alpha=0.7)

    # --- Leyendas, Etiquetas y Guardado ---
    handles, labels = [], []
    processed_labels = set()
    for ax_idx, ax in enumerate(fig.axes):
        ax_handles, ax_labels = ax.get_legend_handles_labels()
        for h, l in zip(ax_handles, ax_labels):
            if l not in processed_labels:
                handles.append(h)
                labels.append(l)
                processed_labels.add(l)

    if handles:
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=2)

    axes[0].set_ylabel(r'Ancho de banda ($k_{sup} - k_{inf}$)')
    fig.text(0.5, -0.05, r'Productividad ($z$)', ha='center', va='center')

    plt.tight_layout(rect=[0, 0.12, 1, 0.91]) # Ajustado rect para más espacio vertical

    if not os.path.exists("./figuras"):
        os.makedirs("./figuras")

    try:
        save_name = f"./figuras/{problem_main_name}_vs_{problem_shadow_name}.pdf"
        fig.savefig(save_name, dpi=300, bbox_inches='tight', pad_inches=0.1)
        print(f"Gráfico guardado en: {save_name}")
    except Exception as e:
        print(f"Error al guardar la figura: {e}")

    plt.show()

def ss_separation(problem_main_name='p6', problem_shadow_name='p3'):
    # --- Definición de Problemas ---
    problem_main_name = 'p6' # Problema principal
    problem_shadow_name = 'p3' # Problema de referencia

    # --- Cargar Datos del Problema de Referencia (p3) ---
    folder_shadow = './backup/' + problem_shadow_name
    policy_paths_shadow = sorted(glob(os.path.join(folder_shadow, '*.mat')))
    z_shadow_all_cases, width_shadow_all_cases = [], []

    if len(policy_paths_shadow) != 4:
        print(f"Advertencia: Se esperaban 4 archivos .mat para {problem_shadow_name}, se encontraron {len(policy_paths_shadow)}")
        for _ in range(4):
            z_shadow_all_cases.append(None)
            width_shadow_all_cases.append(None)
    else:
        for i in range(len(policy_paths_shadow)):
            try:
                backup_s = loadmat(policy_paths_shadow[i])
                z_s_vals = backup_s['z'][:, 0]
                width_s_vals = backup_s['k_upper'][:, 0] - backup_s['k_lower'][:, 0]
                z_shadow_all_cases.append(z_s_vals)
                width_shadow_all_cases.append(width_s_vals)
            except Exception as e:
                print(f"Error cargando datos de sombra para {policy_paths_shadow[i]}: {e}")
                z_shadow_all_cases.append(None)
                width_shadow_all_cases.append(None)

    # Determinar el rango global de z para p3
    p3_z_min_global, p3_z_max_global = None, None
    limit_x_axis_to_p3_range = False
    all_z_shadow_points = np.concatenate([z_vals for z_vals in z_shadow_all_cases if z_vals is not None and len(z_vals) > 0])
    if len(all_z_shadow_points) > 0:
        p3_z_min_global = np.min(all_z_shadow_points)
        p3_z_max_global = np.max(all_z_shadow_points)
        if p3_z_max_global > p3_z_min_global:
            limit_x_axis_to_p3_range = True
        else:
            print("Rango de z para p3 no es válido o es un solo punto.")
    else:
        print("No se pudieron cargar datos de z para p3; no se limitará el eje x ni se calcularán límites de y basados en p3.")

    # Inicializar límites globales de Y
    overall_y_min_for_p3_range = +np.inf
    overall_y_max_for_p3_range = -np.inf
    valid_y_data_found_for_p3_range = False

    # --- Cargar Datos del Problema Principal (p6) y Graficar ---
    folder_main = './backup/' + problem_main_name
    policy_paths_main = sorted(glob(os.path.join(folder_main, '*.mat')))

    fig, axes = plt.subplots(1, 4, sharex=True, sharey=True, figsize=(13, 2.5), gridspec_kw={'wspace':0.05}, dpi=120) # Ligeramente más alto para leyenda
    titles = ['Caso (i.a)', 'Caso (i.b)', 'Caso (ii.a)', 'Caso (ii.b)']

    label_shadow = r'Ancho ($\sigma=0.05$)' # p3
    label_main = r'Ancho ($\sigma=0.15$)'   # p6

    for i, path_main in enumerate(policy_paths_main):
        if i >= 4: break
        try:
            backup_m = loadmat(path_main)
            z_m_data = backup_m['z'][:,0]
            width_m_data = backup_m['k_upper'][:, 0] - backup_m['k_lower'][:, 0]

            axes[i].plot(z_m_data, width_m_data, color='darkblue', linestyle='-', linewidth=1.5, label=label_main, zorder=2)

            z_s_current, width_s_current = None, None
            if i < len(z_shadow_all_cases) and i < len(width_shadow_all_cases):
                z_s_current = z_shadow_all_cases[i]
                width_s_current = width_shadow_all_cases[i]

            if z_s_current is not None and width_s_current is not None:
                axes[i].plot(z_s_current, width_s_current, color='k', linestyle='--', linewidth=1.5, label=label_shadow, zorder=1)

                if len(z_m_data) > 0 and len(z_s_current) > 0 and limit_x_axis_to_p3_range:
                    analysis_z_min = p3_z_min_global
                    analysis_z_max = p3_z_max_global
                    
                    if analysis_z_max > analysis_z_min:
                        num_common_points = 200
                        analysis_z_grid = np.linspace(analysis_z_min, analysis_z_max, num=num_common_points)
                        
                        interp_width_m = np.interp(analysis_z_grid, z_m_data, width_m_data, left=np.nan, right=np.nan)
                        interp_width_s = np.interp(analysis_z_grid, z_s_current, width_s_current, left=np.nan, right=np.nan)
                        
                        # Actualizar límites globales de Y basados en los datos DENTRO del rango de p3
                        current_subplot_y_values_m = interp_width_m[~np.isnan(interp_width_m)]
                        current_subplot_y_values_s = interp_width_s[~np.isnan(interp_width_s)]
                        
                        if len(current_subplot_y_values_m) > 0:
                            overall_y_min_for_p3_range = np.min([overall_y_min_for_p3_range, np.min(current_subplot_y_values_m)])
                            overall_y_max_for_p3_range = np.max([overall_y_max_for_p3_range, np.max(current_subplot_y_values_m)])
                            valid_y_data_found_for_p3_range = True
                        if len(current_subplot_y_values_s) > 0:
                            overall_y_min_for_p3_range = np.min([overall_y_min_for_p3_range, np.min(current_subplot_y_values_s)])
                            overall_y_max_for_p3_range = np.max([overall_y_max_for_p3_range, np.max(current_subplot_y_values_s)])
                            valid_y_data_found_for_p3_range = True

                        diff_array = interp_width_m - interp_width_s
                        if not np.all(np.isnan(diff_array)):
                            min_diff_val, max_diff_val = np.nanmin(diff_array), np.nanmax(diff_array)
                            idx_min_diff, idx_max_diff = np.nanargmin(diff_array), np.nanargmax(diff_array)
                            z_at_min_diff, z_at_max_diff = analysis_z_grid[idx_min_diff], analysis_z_grid[idx_max_diff]
                            
                            y_offset_text = 0.035 
                            current_calc_z_range = analysis_z_max - analysis_z_min

                            # --- Texto para MAXIMA diferencia --- (lógica de texto como antes)
                            z_coord_max = z_at_max_diff
                            ha_max = 'center'
                            if current_calc_z_range > 1e-6:
                                relative_pos_max = (z_coord_max - analysis_z_min) / current_calc_z_range
                                if relative_pos_max < 0.20: ha_max = 'left'
                                elif relative_pos_max > 0.80: ha_max = 'right'
                            
                            y_base_max_m = interp_width_m[idx_max_diff] if not np.isnan(interp_width_m[idx_max_diff]) else -np.inf
                            y_base_max_s = interp_width_s[idx_max_diff] if not np.isnan(interp_width_s[idx_max_diff]) else -np.inf
                            y_base_max = max(y_base_max_m, y_base_max_s)
                            if y_base_max == -np.inf : y_base_max = (overall_y_min_for_p3_range + overall_y_max_for_p3_range) / 2 if valid_y_data_found_for_p3_range else 0


                            y_pos_max_text = y_base_max + y_offset_text
                            va_max = 'bottom'
                            
                            axes[i].text(z_coord_max, y_pos_max_text, r"Max $\Delta$:"+f"{max_diff_val:+.2f}",
                                        horizontalalignment=ha_max, verticalalignment=va_max,
                                        fontsize=8, color='darkblue', weight='bold',
                                        bbox=dict(facecolor='white', alpha=0.85, pad=0.2, edgecolor='lightgray', boxstyle='round,pad=0.25'))

                            # --- Texto para MINIMA diferencia --- (lógica de texto como antes)
                            z_coord_min = z_at_min_diff
                            ha_min = 'center'
                            if current_calc_z_range > 1e-6:
                                relative_pos_min = (z_coord_min - analysis_z_min) / current_calc_z_range
                                if relative_pos_min < 0.20: ha_min = 'left'
                                elif relative_pos_min > 0.80: ha_min = 'right'

                            y_base_min_m = interp_width_m[idx_min_diff] if not np.isnan(interp_width_m[idx_min_diff]) else -np.inf
                            y_base_min_s = interp_width_s[idx_min_diff] if not np.isnan(interp_width_s[idx_min_diff]) else -np.inf
                            y_base_min = max(y_base_min_m, y_base_min_s)
                            if y_base_min == -np.inf : y_base_min = (overall_y_min_for_p3_range + overall_y_max_for_p3_range) / 2 if valid_y_data_found_for_p3_range else 0


                            y_pos_min_text_initial = y_base_min + y_offset_text
                            va_min_initial = 'bottom'
                            y_pos_min_final, va_min_final = y_pos_min_text_initial, va_min_initial

                            if abs(z_coord_max - z_coord_min) < current_calc_z_range * 0.15: 
                                if abs(y_pos_max_text - y_pos_min_text_initial) < y_offset_text * 1.5:
                                    y_base_min_alt_m = interp_width_m[idx_min_diff] if not np.isnan(interp_width_m[idx_min_diff]) else +np.inf
                                    y_base_min_alt_s = interp_width_s[idx_min_diff] if not np.isnan(interp_width_s[idx_min_diff]) else +np.inf
                                    y_base_min_alt = min(y_base_min_alt_m, y_base_min_alt_s)
                                    if y_base_min_alt == +np.inf : y_base_min_alt = (overall_y_min_for_p3_range + overall_y_max_for_p3_range) / 2 if valid_y_data_found_for_p3_range else 0
                                    y_pos_min_final = y_base_min_alt - y_offset_text
                                    va_min_final = 'top'
                            
                            axes[i].text(z_coord_min, y_pos_min_final, r"Min $\Delta$"+f": {min_diff_val:+.2f}",
                                        horizontalalignment=ha_min, verticalalignment=va_min_final,
                                        fontsize=8, color='indigo', weight='bold',
                                        bbox=dict(facecolor='white', alpha=0.85, pad=0.2, edgecolor='lightgray', boxstyle='round,pad=0.25'))
            
            axes[i].set_title(titles[i])
            axes[i].grid(True, linestyle=':', alpha=0.7)

        except Exception as e:
            print(f"Error procesando el archivo {path_main} para el subgráfico {titles[i]}: {e}")
            axes[i].set_title(f"{titles[i]}\n(Error cargando datos)")
            axes[i].grid(True, linestyle=':', alpha=0.7)

    # --- Leyendas, Etiquetas y Guardado ---
    if limit_x_axis_to_p3_range:
        print(f"Limitando eje X al rango de p3: ({p3_z_min_global:.2f}, {p3_z_max_global:.2f})")
        for ax_k in axes:
            ax_k.set_xlim(p3_z_min_global, p3_z_max_global)

    if valid_y_data_found_for_p3_range and overall_y_max_for_p3_range > overall_y_min_for_p3_range:
        y_margin_percent = 0.05 # 5% de margen
        y_data_range = overall_y_max_for_p3_range - overall_y_min_for_p3_range
        y_margin_abs = y_data_range * y_margin_percent
        
        # Si el rango de datos es muy pequeño o cero, usar un margen absoluto pequeño
        if y_data_range < 1e-5 : 
            y_margin_abs = 0.05 # Margen absoluto pequeño para evitar zoom excesivo

        final_y_lim_min = overall_y_min_for_p3_range - y_margin_abs
        final_y_lim_max = overall_y_max_for_p3_range + y_margin_abs
        
        print(f"Limitando eje Y al rango de datos visibles en p3 Z-range: ({final_y_lim_min:.2f}, {final_y_lim_max:.2f})")
        for ax_k in axes:
            ax_k.set_ylim(final_y_lim_min, final_y_lim_max)
    elif valid_y_data_found_for_p3_range: # Caso donde min_y == max_y
        final_y_lim_min = overall_y_min_for_p3_range - 0.1 # Margen fijo
        final_y_lim_max = overall_y_max_for_p3_range + 0.1 # Margen fijo
        print(f"Datos en Y tienen rango cero. Limitando eje Y a: ({final_y_lim_min:.2f}, {final_y_lim_max:.2f})")
        for ax_k in axes:
            ax_k.set_ylim(final_y_lim_min, final_y_lim_max)


    handles, labels = [], []
    processed_labels = set()
    for ax_idx, ax in enumerate(fig.axes):
        ax_handles, ax_labels = ax.get_legend_handles_labels()
        for h, l in zip(ax_handles, ax_labels):
            if l not in processed_labels:
                handles.append(h)
                labels.append(l)
                processed_labels.add(l)

    if handles:
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=2) # Ajustado y para leyenda superior

    axes[0].set_ylabel(r'Ancho de banda ($k_{sup} - k_{inf}$)')
    fig.text(0.5, -0.05, r'Productividad ($z$)', ha='center', va='center') # Ajustada posición y

    plt.tight_layout(rect=[0, 0.05, 1, 0.90]) # Ajustar rect para leyenda y etiquetas

    if not os.path.exists("./figuras"):
        os.makedirs("./figuras")

    try:
        save_name = f"./figuras/{problem_main_name}_vs_{problem_shadow_name}.pdf"
        fig.savefig(save_name, dpi=300, bbox_inches='tight', pad_inches=0.1)
        print(f"Gráfico guardado en: {save_name}")
    except Exception as e:
        print(f"Error al guardar la figura: {e}")

    plt.show()


def plot_slope(problem_main_name = 'p6', problem_shadow_name = 'p3'):
    titles = ['Caso (i.a)', 'Caso (i.b)', 'Caso (ii.a)', 'Caso (ii.b)']
    num_cases = 4
    output_folder = "./figuras"

    # --- Función Auxiliar para Cargar Datos de Política k*(z) ---
    def load_policy_k_star_data(problem_id, num_expected_files):
        """Carga z y k_star para un problema dado."""
        folder = f'./backup/{problem_id}'
        paths = sorted(glob(os.path.join(folder, '*.mat')))
        
        z_arrays = [None] * num_expected_files
        k_star_arrays = [None] * num_expected_files
        
        actual_files_found = len(paths)
        if actual_files_found != num_expected_files:
            print(f"Advertencia: Para '{problem_id}', se esperaban {num_expected_files} archivos, se encontraron {actual_files_found}.")
        
        for i in range(min(actual_files_found, num_expected_files)):
            try:
                data = loadmat(paths[i])
                z_original = data['z'][:, 0]
                k_star_original = data['k_star'].flatten() # Asegurar que k_star es 1D

                if len(z_original) > 1: # Necesitamos al menos 2 puntos para un ajuste lineal
                    # Ordenar por z y eliminar duplicados para un ajuste robusto
                    sorted_indices = np.argsort(z_original)
                    z_sorted = z_original[sorted_indices]
                    k_star_sorted = k_star_original[sorted_indices]
                    
                    unique_z, unique_indices = np.unique(z_sorted, return_index=True)
                    if len(unique_z) >= 2: # np.polyfit necesita al menos 2 puntos únicos
                        z_arrays[i] = unique_z
                        k_star_arrays[i] = k_star_sorted[unique_indices]
                    # else: print(f"Datos insuficientes (tras unificar z) para ajuste en {paths[i]}")
                # else: print(f"Datos originales insuficientes para ajuste en {paths[i]}")
            except Exception as e:
                print(f"Error procesando el archivo {paths[i]}: {e}")
        return z_arrays, k_star_arrays

    # --- Cargar Datos ---
    z_p6_cases, k_star_p6_cases = load_policy_k_star_data(problem_main_name, num_cases)
    z_p3_cases, k_star_p3_cases = load_policy_k_star_data(problem_shadow_name, num_cases)

    # --- Determinar Rango Global de Z (Unión de p3 y p6) ---
    combined_z_points_for_range = []
    for z_list_case in [z_p3_cases, z_p6_cases]:
        for z_array_item in z_list_case:
            if z_array_item is not None and len(z_array_item) > 0:
                combined_z_points_for_range.append(z_array_item)

    overall_x_min_data, overall_x_max_data = None, None
    if combined_z_points_for_range:
        all_z_combined = np.concatenate(combined_z_points_for_range)
        if len(all_z_combined) > 0:
            overall_x_min_data = np.min(all_z_combined)
            overall_x_max_data = np.max(all_z_combined)
            if not (overall_x_max_data > overall_x_min_data): # si min==max o inválido
                overall_x_min_data, overall_x_max_data = None, None 

    final_x_plot_lim = None
    if overall_x_min_data is not None and overall_x_max_data is not None:
        x_range_data = overall_x_max_data - overall_x_min_data
        x_margin = x_range_data * 0.02 
        if x_range_data < 1e-5 : x_margin = 0.1 # Margen absoluto si el rango es diminuto
        final_x_plot_lim = (overall_x_min_data - x_margin, overall_x_max_data + x_margin)
        print(f"Rango X global para gráficos (con margen): ({final_x_plot_lim[0]:.2f}, {final_x_plot_lim[1]:.2f})")
    else:
        print("No se pudo determinar un rango X global. Los ejes X se autoescalarán.")
        final_x_plot_lim = (0,1) if plt.get_backend() else None # Fallback

    # --- Realizar Ajustes Lineales (sobre datos originales completos) ---
    coeffs_p6_fits = [None] * num_cases
    coeffs_p3_fits = [None] * num_cases

    for i in range(num_cases):
        if z_p6_cases[i] is not None and k_star_p6_cases[i] is not None and len(z_p6_cases[i]) >= 2:
            coeffs_p6_fits[i] = np.polyfit(z_p6_cases[i], k_star_p6_cases[i], 1) # Retorna [pendiente, intercepto]
        if z_p3_cases[i] is not None and k_star_p3_cases[i] is not None and len(z_p3_cases[i]) >= 2:
            coeffs_p3_fits[i] = np.polyfit(z_p3_cases[i], k_star_p3_cases[i], 1)

    # --- Determinar Rango Global de Y (k_star y sus ajustes) dentro del Rango X Final ---
    all_y_values_in_visible_x_range = []
    if final_x_plot_lim is not None:
        x_line_for_y_range = np.array([final_x_plot_lim[0], final_x_plot_lim[1]]) # Puntos para evaluar líneas de ajuste

        for i in range(num_cases):
            # Para k_star p6 y su ajuste
            if z_p6_cases[i] is not None and k_star_p6_cases[i] is not None:
                mask = (z_p6_cases[i] >= final_x_plot_lim[0]) & (z_p6_cases[i] <= final_x_plot_lim[1])
                all_y_values_in_visible_x_range.extend(k_star_p6_cases[i][mask])
            if coeffs_p6_fits[i] is not None:
                m, c = coeffs_p6_fits[i]
                all_y_values_in_visible_x_range.extend(m * x_line_for_y_range + c)

            # Para k_star p3 y su ajuste
            if z_p3_cases[i] is not None and k_star_p3_cases[i] is not None:
                mask = (z_p3_cases[i] >= final_x_plot_lim[0]) & (z_p3_cases[i] <= final_x_plot_lim[1])
                all_y_values_in_visible_x_range.extend(k_star_p3_cases[i][mask])
            if coeffs_p3_fits[i] is not None:
                m, c = coeffs_p3_fits[i]
                all_y_values_in_visible_x_range.extend(m * x_line_for_y_range + c)

    final_y_plot_lim = None
    if all_y_values_in_visible_x_range:
        min_y_visible = np.min(all_y_values_in_visible_x_range)
        max_y_visible = np.max(all_y_values_in_visible_x_range)
        
        if max_y_visible > min_y_visible:
            y_range_data = max_y_visible - min_y_visible
            y_margin = y_range_data * 0.10 
            final_y_plot_lim = (min_y_visible - y_margin, max_y_visible + y_margin)
        else: 
            abs_val = abs(min_y_visible)
            y_margin_fallback = 0.1 * abs_val if abs_val > 1e-5 else 0.1
            final_y_plot_lim = (min_y_visible - y_margin_fallback, max_y_visible + y_margin_fallback)
        print(f"Rango Y global para k*(z) y ajustes (con margen): ({final_y_plot_lim[0]:.2f}, {final_y_plot_lim[1]:.2f})")
    else:
        print("No hay valores k*(z) visibles para determinar el rango Y. El eje Y se autoescalará.")


    # --- Graficar ---
    fig, axes = plt.subplots(1, num_cases, sharex=True, sharey=True, 
                            figsize=(13, 2.5), gridspec_kw={'wspace':0.05}, dpi=100) # Un poco más de altura

    try:
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        label_p3_k = r'$k^*_{(sigma=0.05)}$'
        label_p6_k = r'$k^*_{(sigma=0.15)}$'
        label_p3_fit = r'Ajuste Lin. ($\sigma = 0.05$)'
        label_p6_fit = r'Ajuste Lin. ($\sigma = 0.15$)'
    except RuntimeError:
        print("LaTeX no disponible, usando etiquetas de texto plano.")
        label_p3_k = f'k* (sigma=0.05)'
        label_p6_k = f'k* (sigma=0.15)'
        label_p3_fit = r'Ajuste Lin. ($\sigma = 0.05$)'
        label_p6_fit = r'Ajuste Lin. ($\sigma = 0.15$)'

    for i in range(num_cases):
        ax = axes[i]
        ax.set_title(titles[i] if i < len(titles) else f'Caso {i+1}')

        plot_p6_done, plot_p3_done = False, False
        # Graficar p6 (principal) y su ajuste lineal
        if z_p6_cases[i] is not None and k_star_p6_cases[i] is not None:
            ax.plot(z_p6_cases[i], k_star_p6_cases[i], color='darkblue', linestyle='-', linewidth=2, label=label_p6_k, alpha=1.)
            plot_p6_done = True
            if coeffs_p6_fits[i] is not None and final_x_plot_lim is not None:
                m, c = coeffs_p6_fits[i]
                x_fit_line = np.array(final_x_plot_lim) # Graficar la línea de ajuste en todo el rango X visible
                ax.plot(x_fit_line, m * x_fit_line + c, color='darkblue', linestyle=':', linewidth=1.5, label=label_p6_fit)
                
        # Graficar p3 (referencia) y su ajuste lineal
        if z_p3_cases[i] is not None and k_star_p3_cases[i] is not None:
            ax.plot(z_p3_cases[i], k_star_p3_cases[i], color='darkgreen', linestyle='-', linewidth=2, label=label_p3_k, alpha=1.)
            plot_p3_done = True
            if coeffs_p3_fits[i] is not None and final_x_plot_lim is not None:
                m, c = coeffs_p3_fits[i]
                x_fit_line = np.array(final_x_plot_lim)
                ax.plot(x_fit_line, m * x_fit_line + c, color='darkgreen', linestyle='-.', linewidth=1.5, label=label_p3_fit, alpha=1.)
        
        ax.grid(True, linestyle=':', alpha=0.5)
        if not plot_p6_done and not plot_p3_done :
            ax.text(0.5, 0.5, "Datos no disponibles", transform=ax.transAxes, ha='center', va='center', fontsize=9)

    # --- Ajustes Finales de los Ejes y Leyenda ---
    if final_x_plot_lim:
        axes[0].set_xlim(final_x_plot_lim)
    if final_y_plot_lim:
        axes[0].set_ylim(final_y_plot_lim)

    axes[0].set_ylabel(r'Capital Óptimo $k^*(z)$')
    fig.text(0.5, -0.05, r'Productividad ($z$)', ha='center', va='center')

    handles, plot_labels_unique = [], []
    temp_labels_dict_unique = {} 
    for ax_iter in fig.axes: # Recolectar handles y etiquetas para leyenda única
        h_iter, l_iter = ax_iter.get_legend_handles_labels()
        for new_h, new_l in zip(h_iter,l_iter):
            if new_l not in temp_labels_dict_unique: # Evitar duplicados
                temp_labels_dict_unique[new_l] = new_h
    handles = list(temp_labels_dict_unique.values())
    plot_labels_unique = list(temp_labels_dict_unique.keys())

    if handles:
        fig.legend(handles, plot_labels_unique, loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=4) # ncol=2 o 4

    plt.tight_layout(rect=[0, 0.05, 1, 0.90]) # Ajustar rect para la leyenda y etiquetas

    # --- Guardar Figura ---
    if not os.path.exists(output_folder): os.makedirs(output_folder)
    save_name = os.path.join(output_folder, f"{problem_main_name}_vs_{problem_shadow_name}_slope.pdf")
    try:
        fig.savefig(save_name, dpi=300, bbox_inches='tight', pad_inches=0.05)
        print(f"Gráfico guardado en: {save_name}")
    except Exception as e: print(f"Error al guardar la figura: {e}")

    plt.show()


def sigma_vs_sswidth():
    problem = 'p7'
    folder = './backup/'+problem
    policy_paths_ext = glob(os.path.join(folder, '*', '*.mat'))
    sigma_groups = [policy_paths_ext[i:i+4] for i in range(0, len(policy_paths_ext), 4)]

    # --- Configuración de Estilos de Línea para Estadísticas ---
    # Puedes modificar colores, alphas, estilos de línea y etiquetas aquí
    STAT_CONFIG = {
        'max':    {'color': 'gray',  'alpha': 0.7, 'linestyle': '--', 'linewidth': 1.5, 'label': 'Máximo', 'marker': '^'},
        'min':    {'color': 'gray', 'alpha': 0.7, 'linestyle': ':',  'linewidth': 1.5, 'label': 'Mínimo', 'marker': 's'},
        'median': {'color': 'darkgreen',   'alpha': 1.0, 'linestyle': '-',  'linewidth': 2.0, 'label': 'Mediana', 'marker': 'o', 'markersize': 4},
        'mean':   {'color': 'darkblue', 'alpha': 0.9, 'linestyle': '-.','linewidth': 2.0, 'label': 'Media',   'marker': 'x', 'markersize': 5}
    }
    output_folder_p7 = "./figuras"


    # --- Procesamiento de Datos ---
    parsed_sigmas = []
    case_labels = ['Caso (i.a)', 'Caso (i.b)', 'Caso (ii.a)', 'Caso (ii.b)']
    # Estructura para almacenar las estadísticas: stats_data[estadistica][caso] = [lista de valores]
    stats_data = {stat_name: {case_label: [] for case_label in case_labels} for stat_name in STAT_CONFIG.keys()}

    for group_paths in sigma_groups:
        if not group_paths or len(group_paths) == 0: continue

        normalized_path = group_paths[0].replace('\\', '/')
        match = re.search(r"sigma_(\d+\.\d+)", normalized_path)
        if not match:
            print(f"Advertencia: No se pudo extraer sigma de '{group_paths[0]}'. Saltando grupo.")
            continue
            
        current_sigma = float(match.group(1))
        
        # Almacenadores temporales para este sigma
        temp_stats_for_this_sigma = {
            stat: {label: np.nan for label in case_labels} for stat in STAT_CONFIG.keys()
        }
        any_file_processed_for_sigma = False

        for i, file_path in enumerate(group_paths):
            if i >= len(case_labels): break
            current_case_label = case_labels[i]
            try:
                mat_data = loadmat(file_path)
                k_lower = mat_data['k_lower'].flatten()
                k_upper = mat_data['k_upper'].flatten()

                if len(k_lower) > 0 and len(k_lower) == len(k_upper):
                    band_width_vector = k_upper - k_lower
                    if len(band_width_vector) > 0:
                        temp_stats_for_this_sigma['max'][current_case_label] = np.max(band_width_vector)
                        temp_stats_for_this_sigma['min'][current_case_label] = np.min(band_width_vector)
                        temp_stats_for_this_sigma['median'][current_case_label] = np.median(band_width_vector)
                        temp_stats_for_this_sigma['mean'][current_case_label] = np.mean(band_width_vector)
                        any_file_processed_for_sigma = True
                    else: print(f"Advertencia: Vector de ancho de banda vacío para {file_path}.")
                else: print(f"Advertencia: Vectores k_lower/k_upper vacíos o con longitudes desiguales en {file_path}.")
            except FileNotFoundError: print(f"Advertencia: Archivo no encontrado {file_path}.")
            except Exception as e: print(f"Error procesando el archivo {file_path}: {e}.")
        
        # Solo añadir sigma y sus datos si al menos un archivo se procesó o se intentó
        if any_file_processed_for_sigma or True: # Se añade sigma de todas formas, con NaNs si es necesario
            parsed_sigmas.append(current_sigma)
            for stat_name in STAT_CONFIG.keys():
                for case_label_to_append in case_labels:
                    stats_data[stat_name][case_label_to_append].append(temp_stats_for_this_sigma[stat_name][case_label_to_append])

    # Ordenar datos por sigma
    if parsed_sigmas:
        sort_indices = np.argsort(parsed_sigmas)
        sigmas_np = np.array(parsed_sigmas)[sort_indices]
        for stat_name in STAT_CONFIG.keys():
            for label in case_labels:
                stats_data[stat_name][label] = np.array(stats_data[stat_name][label])[sort_indices]
    else:
        sigmas_np = np.array([])
        print("No se procesaron datos de sigma válidos.")

    # --- Graficar en Subplots ---
    if len(sigmas_np) > 0:
        fig, axes = plt.subplots(1, 4, figsize=(13, 3.5), sharex=False, sharey=True, dpi=100) # sharey=False es más flexible
        axes_flat = axes.flatten()

        # Configuración de LaTeX (opcional)
        try:
            plt.rc('text', usetex=True)
            plt.rc('font', family='serif')
            latex_on = True
            xlabel_text = r'Volatilidad de Productividad ($\sigma$)'
            ylabel_text_common = r'Ancho de Banda de Inacción'
            suptitle_text = r'Estadísticas del Ancho de Banda de Inacción vs. $\sigma$'
        except RuntimeError:
            print("LaTeX no disponible, usando etiquetas de texto plano.")
            latex_on = False
            xlabel_text = 'Volatilidad de Productividad (sigma)'
            ylabel_text_common = 'Ancho de Banda de Inaccion'
            suptitle_text = 'Estadisticas del Ancho de Banda de Inaccion vs. Sigma'

        for i, case_name in enumerate(case_labels):
            ax = axes_flat[i]

            plotted_anything_on_ax = False
            for stat_name, config in STAT_CONFIG.items():
                data_list = stats_data[stat_name][case_name]
                if len(data_list) == len(sigmas_np):
                    valid_indices = ~np.isnan(data_list)
                    if np.any(valid_indices):
                        ax.plot(sigmas_np[valid_indices], data_list[valid_indices],
                                color=config['color'], alpha=config['alpha'],
                                linestyle=config['linestyle'], linewidth=config['linewidth'],
                                label=config['label'], marker=config.get('marker', None), # Añadir marker si está en config
                                markersize=config.get('markersize', None)) 
                        plotted_anything_on_ax = True
                # else: print(f"Inconsistencia de datos para {stat_name}, {case_name}")
            
            if not plotted_anything_on_ax:
                ax.text(0.5, 0.5, "Datos no disponibles\npara este caso", transform=ax.transAxes,
                        horizontalalignment='center', verticalalignment='center', fontsize=10, color='gray')

            ax.grid(True, linestyle=':', alpha=0.7)
            ax.set_xlabel(xlabel_text)
            if i == 0:
                fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=4) # Ajustado y para leyenda superior
            # Auto-ajuste del eje Y para cada subplot
            if plotted_anything_on_ax:
                ax.autoscale(enable=True, axis='y', tight=False)
                current_ylim = ax.get_ylim()
                yrange = current_ylim[1] - current_ylim[0]
                if yrange > 1e-6 : # Evitar margen excesivo si el rango es casi cero
                    ax.set_ylim(current_ylim[0] - yrange*0.05, current_ylim[1] + yrange*0.05)
        axes[0].set_ylabel(ylabel_text_common)
        

        # fig.suptitle(suptitle_text, fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Ajustar para suptitle

        # --- Guardar Figura ---
        if not os.path.exists(output_folder_p7):
            os.makedirs(output_folder_p7)
        save_name = os.path.join(output_folder_p7, "p7_width.pdf")
        try:
            plt.savefig(save_name, dpi=300, bbox_inches='tight')
            print(f"Gráfico guardado en: {save_name}")
        except Exception as e:
            print(f"Error al guardar el gráfico: {e}")
        
        plt.show()
    else:
        print("No hay datos suficientes para generar los gráficos de estadísticas.")


def compare_with_base(problem='p6'):
    # Carga los datos para la política de referencia (p3) que actuará como sombra
    policy_p3_paths = glob(os.path.join('./backup/p3', '*.mat'))

    # Variables para almacenar los datos de la sombra
    z_shadow, k_star_shadow, k_lower_shadow, k_upper_shadow = None, None, None, None
    plot_shadow_flag = False

    if policy_p3_paths:
        # Asumimos que quieres usar el primer archivo .mat encontrado para p3 como sombra
        # Si tienes múltiples archivos para p3 y una lógica específica, deberás ajustarlo.
        shadow_backup_path = policy_p3_paths[0]
        try:
            backup_shadow_data = loadmat(shadow_backup_path)
            z_shadow = backup_shadow_data['z']
            k_star_shadow = backup_shadow_data['k_star'][0]
            k_lower_shadow = backup_shadow_data['k_lower']
            k_upper_shadow = backup_shadow_data['k_upper']
            plot_shadow_flag = True
        except Exception as e:
            print(f"Error {shadow_backup_path}: {e}")


    
    folder = './backup/' + problem
    policy_paths = glob(os.path.join(folder, '*.mat'))
    


    fig, axes = plt.subplots(1, 4, sharex=True, sharey=True, figsize=(13, 2.5), gridspec_kw={'wspace':0.05}, dpi=100)
    titles = ['Caso (i.a)', 'Caso (i.b)', 'Caso (ii.a)', 'Caso (ii.b)']

    if problem == 'p4':
        labelx = r'Banda de inacción cuando $\rho = 0.85$'
        labely = r'$k^*(z)$ ($\rho=0$)'
        labelz = r'Banda de inacción ($\rho=0$)'
        offset = 0.52

    if problem == 'p5':
        labelx = r'Banda de inacción cuando $P_0 = 0.03$ y $F_0 = 0.02$'
        labely = r'$k^*(z)$ ($P_1 = 0.06$ y $F_1 = 0.04$)'
        labelz = r'Banda de inacción ($P_1 = 0.06$ y $F_1 = 0.04$)'
        offset = 1.15

    if problem == 'p6':
        labelx = r'Banda de inacción cuando $\sigma = 0.05$'
        labely = r'$k^*(z)$ ($\sigma = 0.15$)'
        labelz = r'Banda de inacción ($\sigma = 0.15$)'
        offset = 0.5

    if problem == 'p7/sigma_0.15':
        os.makedirs('./figuras/p7', exist_ok=True)
        labelx = r'Banda de inacción cuando $\sigma = 0.05$ y $\delta = 0.$'
        labely = r'$k^*(z)$ ($\sigma = 0.15$ y $\delta = 0.3$)'
        labelz = r'Banda de inacción ($\sigma = 0.15$ y $\delta = 0.3$)'
        offset = 1.

    if problem == 'p7/sigma_0.05':
        os.makedirs('./figuras/p7', exist_ok=True)
        labelx = r'Banda de inacción cuando $\sigma = 0.05$ y $\delta = 0.$'
        labely = r'$k^*(z)$ ($\sigma = 0.05$ y $\delta = 0.3$)'
        labelz = r'Banda de inacción ($\sigma = 0.05$ y $\delta = 0.3$)'
        offset = 1.

    for i, path in enumerate(policy_paths):
        try:
            backup = loadmat(path)
            z_main = backup['z']
            k_star_main = backup['k_star'][0]
            k_lower_main = backup['k_lower']
            k_upper_main = backup['k_upper']

            # 1. Graficar la sombra (datos de p3) si están disponibles
            if plot_shadow_flag and z_shadow is not None:
                # Sombra de k_star
                axes[i].plot(z_shadow, k_star_shadow, color='lightgray', linestyle='-', 
                            linewidth=1, alpha=0.7, zorder=1)
                # Sombra de la banda de inacción (fill)
                axes[i].fill_between(z_shadow[:, 0], k_lower_shadow[:, 0], k_upper_shadow[:, 0], 
                                    color='silver', alpha=0.5, zorder=0, label=labelx)
                
            # 2. Graficar los datos principales (p4)
            axes[i].plot(z_main, k_star_main, color='k', label=labely, linewidth=1, zorder=3)
            axes[i].fill_between(z_main[:, 0], k_lower_main[:, 0], k_upper_main[:, 0], color='lightblue', alpha=0.4, zorder=2)

            axes[i].plot(z_main, k_lower_main, linestyle='--', color='darkblue', linewidth=1, zorder=3)
            axes[i].plot(z_main, k_upper_main, linestyle='--', color='darkblue', linewidth=1, 
                        label=labelz, zorder=3)

            axes[i].set_title(titles[i])
            axes[i].grid(True, linestyle=':', alpha=0.7) # Hice la grilla un poco más sutil

        except Exception as e:
            print(f"Error procesando el archivo {path} para el subgráfico {i}: {e}")
            axes[i].set_title(f"{titles[i]}\n(Error cargando datos)")
            axes[i].grid(True, linestyle=':', alpha=0.7)

    axes[3].legend(ncols=3, bbox_to_anchor=(offset, 1.45))
    axes[0].set_ylabel(r'Capital ($k$)')

    fig.text(0.463, -0.05, r'Productividad ($z$)', va='center', rotation='horizontal')
    plt.tight_layout(rect=[0, 0.05, 1, 0.95]) # Ajusta el layout para dar espacio al fig.text y títulos

    try:
        fig.savefig("./figuras/{}.pdf".format(problem), dpi=300, bbox_inches='tight', pad_inches=0.)
        print(f"Gráfico guardado en: ./figuras/{problem}.pdf")
    except Exception as e:
        print(f"Error al guardar la figura: {e}")

    plt.show()