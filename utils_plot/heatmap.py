import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # Para un heatmap más estético
from scipy.io import loadmat
from glob import glob
import os
import re
import pandas as pd

def generate_p8_heatmap(base_folder, statistic_type='mean', selected_case_name=None,
                        cmap_name='viridis', # Nuevo parámetro para el colormap
                        output_filename="p8_heatmap.pdf"):
    """
    Genera un mapa de calor del ancho de la banda de inacción en función de sigma y delta.
    Ahora sin anotaciones numéricas ni líneas de grilla en el heatmap, y con colormap configurable.

    Args:
        base_folder (str): Carpeta base que contiene los subdirectorios (ej: './backup/p8').
        statistic_type (str): Tipo de estadística a calcular para el ancho de banda.
                              Opciones: 'max', 'min', 'mean', 'median'.
        selected_case_name (str, opcional): Caso específico a graficar ('i.a', 'i.b', 'ii.a', 'ii.b').
                                         Si es None, se promedia la estadística entre los 4 casos.
        cmap_name (str, opcional): Nombre del colormap de Matplotlib a usar (ej: 'viridis', 'plasma', 'coolwarm').
        output_filename (str): Nombre del archivo PDF para guardar el gráfico.
    """

    case_map_to_filename = {
        'i.a': '1_fixed.mat',
        'i.b': '1_proportional.mat',
        'ii.a': '2_fixed.mat',
        'ii.b': '2_proportional.mat'
    }
    all_case_filenames_ordered = list(case_map_to_filename.values())
    data_for_heatmap = []
    sub_dirs = sorted(glob(os.path.join(base_folder, 'sigma_*_delta_*')))

    if not sub_dirs:
        print(f"No se encontraron subdirectorios en '{base_folder}' con el patrón 'sigma_*_delta_*'.")
        return

    parsed_sigmas = set()
    parsed_deltas = set()

    for dir_path in sub_dirs:
        dir_name = os.path.basename(dir_path)
        match = re.match(r"sigma_(\d+\.\d+)_delta_(\d+\.\d{1,2})", dir_name)
        if not match:
            print(f"Advertencia: Formato de directorio no reconocido '{dir_name}'. Saltando.")
            continue
        
        sigma_val = float(match.group(1))
        delta_val = float(match.group(2))
        parsed_sigmas.add(sigma_val)
        parsed_deltas.add(delta_val)
        stats_for_current_sigma_delta = []
        
        files_to_process_paths = []
        if selected_case_name:
            if selected_case_name not in case_map_to_filename:
                print(f"Advertencia: Nombre de caso '{selected_case_name}' no reconocido. Se promediarán los casos.")
                for fname in all_case_filenames_ordered:
                    files_to_process_paths.append(os.path.join(dir_path, fname))
            else:
                files_to_process_paths.append(os.path.join(dir_path, case_map_to_filename[selected_case_name]))
        else:
            for fname in all_case_filenames_ordered:
                files_to_process_paths.append(os.path.join(dir_path, fname))
        
        for mat_file_path in files_to_process_paths:
            if not os.path.exists(mat_file_path):
                continue
            try:
                mat_data = loadmat(mat_file_path)
                k_lower = mat_data['k_lower'].flatten()
                k_upper = mat_data['k_upper'].flatten()
                if len(k_lower) > 0 and len(k_lower) == len(k_upper):
                    band_width_vector = k_upper - k_lower
                    if len(band_width_vector) > 0:
                        stat_val = np.nan
                        if statistic_type == 'max': stat_val = np.max(band_width_vector)
                        elif statistic_type == 'min': stat_val = np.min(band_width_vector)
                        elif statistic_type == 'mean': stat_val = np.mean(band_width_vector)
                        elif statistic_type == 'median': stat_val = np.median(band_width_vector)
                        else: print(f"Advertencia: Tipo de estadística '{statistic_type}' no reconocido.")
                        if not np.isnan(stat_val): stats_for_current_sigma_delta.append(stat_val)
            except Exception as e:
                print(f"Error procesando archivo {mat_file_path}: {e}")
        
        final_value_for_cell = np.nan
        if stats_for_current_sigma_delta:
            if selected_case_name and len(stats_for_current_sigma_delta) == 1 :
                 final_value_for_cell = stats_for_current_sigma_delta[0]
            elif not selected_case_name : 
                 final_value_for_cell = np.nanmean(stats_for_current_sigma_delta)
        if not np.isnan(final_value_for_cell):
            data_for_heatmap.append({'sigma': sigma_val, 'delta': delta_val, 'value': final_value_for_cell})

    if not data_for_heatmap:
        print("No se procesaron datos válidos para generar el heatmap.")
        return

    df = pd.DataFrame(data_for_heatmap)
    try:
        heatmap_data = df.pivot_table(index='sigma', columns='delta', values='value', aggfunc=np.mean)
    except Exception as e:
        print(f"Error al pivotar los datos: {e}. Verifique la consistencia de los datos.")
        return

    heatmap_data.sort_index(axis=0, inplace=True)
    heatmap_data.sort_index(axis=1, inplace=True)
    
    if heatmap_data.empty:
        print("La tabla pivote para el heatmap está vacía. No se puede generar el gráfico.")
        return

    plt.figure(figsize=(5, 5))
    # MODIFICACIONES AQUÍ: annot=False, linewidths=0, y se usa cmap_name
    sns.heatmap(heatmap_data, annot=False, cmap=cmap_name, linewidths=0, cbar=True)

    case_info = f"Caso: {selected_case_name}" if selected_case_name else "Promedio de Casos"
    stat_desc = {"max": "Máximo", "min": "Mínimo", "mean": "Promedio", "median": "Mediana"}
    
    title_text = f"Mapa de Calor: Ancho {stat_desc.get(statistic_type, statistic_type)} de Banda ({case_info})"
    xlabel_text = r'\rightarrow\quad\rightarrow  ($\delta$)'
    ylabel_text = r'\rightarrow\quad\rightarrow ($\sigma$)'
    
    try:
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.title(title_text, fontsize=15)
        plt.xlabel(xlabel_text, fontsize=12)
        plt.ylabel(ylabel_text, fontsize=12)
    except RuntimeError:
        print("LaTeX no disponible, usando etiquetas de texto plano.")
        plain_title = title_text.replace('$', '').replace('\\', '').replace('{', '').replace('}', '')
        plain_xlabel = xlabel_text.replace('$', '').replace('\\', '').replace('{', '').replace('}', '')
        plain_ylabel = ylabel_text.replace('$', '').replace('\\', '').replace('{', '').replace('}', '')
        plt.title(plain_title, fontsize=15)
        plt.xlabel(plain_xlabel, fontsize=12)
        plt.ylabel(plain_ylabel, fontsize=12)

    plt.tight_layout()
    
    output_folder_p8 = os.path.join('./figuras', "p8")
    if not os.path.exists(output_folder_p8):
        os.makedirs(output_folder_p8)
    
    full_save_path = os.path.join(output_folder_p8, output_filename)
    try:
        plt.savefig(full_save_path, dpi=300, bbox_inches='tight')
        print(f"Heatmap guardado en: {full_save_path}")
    except Exception as e:
        print(f"Error al guardar el heatmap: {e}")
    plt.show()



def generate_p8_heatmaps_all_cases_1x4(base_folder, statistic_type='mean',
                                       cmap_name='viridis',
                                       figsize=(22, 5), # Ajustado para 1x4
                                       output_filename="p8_all_cases_heatmaps_1x4.pdf"):
    """
    Genera 4 heatmaps (uno por caso) en una disposición 1x4.
    Cada heatmap muestra una estadística del ancho de banda en función de sigma y delta.
    """

    case_file_map = {
        'Caso (i.a)': '1_fixed.mat',
        'Caso (i.b)': '1_proportional.mat',
        'Caso (ii.a)': '2_fixed.mat',
        'Caso (ii.b)': '2_proportional.mat'
    }
    case_labels_ordered = list(case_file_map.keys())
    all_stats_for_df = []

    sub_dirs = sorted(glob(os.path.join(base_folder, 'sigma_*_delta_*')))
    if not sub_dirs:
        print(f"No se encontraron subdirectorios en '{base_folder}' con el patrón 'sigma_*_delta_*'.")
        return

    for dir_path in sub_dirs:
        dir_name = os.path.basename(dir_path)
        match = re.match(r"sigma_(\d+\.\d+)_delta_(\d+\.\d{1,2})", dir_name)
        if not match:
            # print(f"Advertencia: Formato de directorio no reconocido '{dir_name}'. Saltando.") # Puede ser verboso
            continue
        sigma_val = float(match.group(1))
        delta_val = float(match.group(2))

        for case_label, case_filename in case_file_map.items():
            mat_file_path = os.path.join(dir_path, case_filename)
            stat_val = np.nan
            if os.path.exists(mat_file_path):
                try:
                    mat_data = loadmat(mat_file_path)
                    k_lower = mat_data['k_lower'].flatten()
                    k_upper = mat_data['k_upper'].flatten()
                    if len(k_lower) > 0 and len(k_lower) == len(k_upper):
                        band_width_vector = k_upper - k_lower
                        if len(band_width_vector) > 0:
                            if statistic_type == 'max': stat_val = np.max(band_width_vector)
                            elif statistic_type == 'min': stat_val = np.min(band_width_vector)
                            elif statistic_type == 'mean': stat_val = np.mean(band_width_vector)
                            elif statistic_type == 'median': stat_val = np.median(band_width_vector)
                except Exception as e:
                    print(f"Error procesando archivo {mat_file_path}: {e}")
            all_stats_for_df.append({
                'sigma': sigma_val, 'delta': delta_val,
                'case': case_label, 'value': stat_val
            })

    if not all_stats_for_df:
        print("No se procesaron datos válidos para generar los heatmaps.")
        return

    df_all_cases = pd.DataFrame(all_stats_for_df)
    valid_values = df_all_cases['value'].dropna()
    if valid_values.empty:
        print("No hay valores válidos de la estadística para determinar la escala de color.")
        global_vmin, global_vmax = 0, 1
    else:
        global_vmin, global_vmax = valid_values.min(), valid_values.max()
        if global_vmin == global_vmax:
            global_vmin = global_vmin - 0.1 * abs(global_vmin) if abs(global_vmin) > 1e-6 else global_vmin - 0.1
            global_vmax = global_vmax + 0.1 * abs(global_vmax) if abs(global_vmax) > 1e-6 else global_vmax + 0.1
            if global_vmin >= global_vmax:
                global_vmin -= 0.1 if global_vmin != 0 else 0.1 # Asegurar un rango
                global_vmax += 0.1 if global_vmax != 0 else 0.1

    # --- Código anterior para cargar y procesar datos ---
    # ... (todo tu código hasta la sección de graficar) ...

    # --- Graficar 4 Subplots en formato 1x4 ---
    fig, axes = plt.subplots(1, 4, figsize=figsize, sharex=True, sharey=True, dpi=100)
    if isinstance(axes, plt.Axes): 
        axes = np.array([axes])
    axes_flat = axes.flatten()
        
    try:
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        latex_on = True
    except RuntimeError:
        print("LaTeX no disponible, usando etiquetas de texto plano.")
        latex_on = False

    stat_desc_map = {"max": "Máximo", "min": "Mínimo", "mean": "Promedio", "median": "Mediana"}
    stat_title_part = stat_desc_map.get(statistic_type, statistic_type.capitalize())
    mappable_for_colorbar = None

    for i, case_label_to_plot in enumerate(case_labels_ordered):
        ax = axes_flat[i]
        df_current_case = df_all_cases[df_all_cases['case'] == case_label_to_plot]
        heatmap_data_current_case = pd.DataFrame()
        if not df_current_case.empty:
            try:
                heatmap_data_current_case = df_current_case.pivot_table(
                    index='sigma', columns='delta', values='value', dropna=False
                )
                # Asegurar que el índice (sigma) y las columnas (delta) estén ordenados
                # para que la inversión del eje Y tenga el efecto visual esperado.
                # Si all_unique_sigmas/deltas se usaron para reindexar, ya deberían estar ordenados.
                # Si no, es buena idea ordenarlos:
                heatmap_data_current_case.sort_index(axis=0, ascending=True, inplace=True) # Sigma de menor a mayor
                heatmap_data_current_case.sort_index(axis=1, ascending=True, inplace=True) # Delta de menor a mayor

            except Exception as e:
                print(f"Error al pivotar datos para {case_label_to_plot}: {e}")
        
        if heatmap_data_current_case.empty or heatmap_data_current_case.isnull().all().all():
            ax.text(0.5, 0.5, "Datos no\ndisponibles", transform=ax.transAxes, ha='center', va='center')
            ax.set_title(case_label_to_plot)
        else:
            current_mappable = sns.heatmap(heatmap_data_current_case, ax=ax, annot=False, cmap=cmap_name, 
                                        linewidths=0, cbar=False,
                                        vmin=global_vmin, vmax=global_vmax)
            if mappable_for_colorbar is None and not heatmap_data_current_case.empty:
                mappable_for_colorbar = current_mappable
            # ax.set_title(case_label_to_plot) # El título ya se estableció antes en tu código.
                                            # Lo moví dentro del else para que solo se ponga si hay heatmap.
                                            # O mejor, ponerlo siempre fuera del else:
        ax.set_title(case_label_to_plot)


        # Configurar etiquetas de ejes X
        # (Tu código para set_xlabel y ticks X ya está aquí, lo mantengo)
        ax.set_xlabel(r'$\rightarrow\quad\rightarrow$ ($\delta$)' if latex_on else 'Delta')
        # Aquí deberías añadir tu lógica para los X ticks (MultipleLocator, etc.) si es necesario para cada 'ax'
        # Ejemplo (si lo tenías en el bucle y no después):
        # ax.xaxis.set_major_locator(MultipleLocator(0.2))
        # ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        # plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")


        if i == 0: # Solo para el primer subplot (más a la izquierda)
            ax.set_ylabel(r'$\leftarrow\quad\leftarrow$ ($\sigma$)' if latex_on else 'Sigma')
            # La configuración de Y ticks (MultipleLocator, FormatStrFormatter, y rotación de etiquetas)
            # se hará DESPUÉS del bucle para el primer eje, y se propagará si sharey=True.
        else:
            ax.set_ylabel('')


    # --- Ajustes FINALES de los Ejes (DESPUÉS del bucle de subplots) ---

    # Invertir el eje Y (Sigma) para que los valores bajos estén abajo.
    # Como sharey=True, aplicar al primer eje debería invertir todos los ejes Y compartidos.
    if len(axes_flat) > 0:
        axes_flat[0].invert_yaxis() # <--- LÍNEA CLAVE PARA INVERTIR EL EJE Y

    # Configuración de Ticks Y (Sigma) para el primer subplot (se propaga si sharey=True)
    # Asumiendo que all_unique_sigmas está disponible y ordenado
    # all_unique_sigmas = sorted(df_all_cases['sigma'].dropna().unique()) # Calcular si no está global
    if len(axes_flat) > 0 and 'all_unique_sigmas' in locals() and all_unique_sigmas:
        axes_flat[0].yaxis.set_major_locator(MultipleLocator(0.02))
        axes_flat[0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        # La rotación de las yticklabels del primer eje.
        plt.setp(axes_flat[0].get_yticklabels(), rotation=0, va='center')
    else: # Fallback si all_unique_sigmas no está definido o está vacío
        if len(axes_flat) > 0 : plt.setp(axes_flat[0].get_yticklabels(), rotation=0, va='center')


    # Configuración de Ticks X (Delta) - aplicar al primer subplot (se propaga si sharex=True)
    # Asumiendo que all_unique_deltas está disponible y ordenado
    # all_unique_deltas = sorted(df_all_cases['delta'].dropna().unique()) # Calcular si no está global
    if len(axes_flat) > 0 and 'all_unique_deltas' in locals() and all_unique_deltas:
        axes_flat[0].xaxis.set_major_locator(MultipleLocator(0.2))
        axes_flat[0].xaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    # Asegurar que las etiquetas X rotadas se apliquen a todos los subplots visibles
    # (Esto es mejor hacerlo en el bucle si las etiquetas X no se comparten bien,
    #  o si cada subplot tiene diferentes xticks/xticklabels debido a datos faltantes
    #  aunque el reindex debería haberlo solucionado)
    for ax_k in axes_flat: # Es más seguro aplicar la rotación a todos
        plt.setp(ax_k.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")


    # --- Código posterior para colorbar, suptitle, tight_layout, guardar y show ---
    # ... (tu código existente) ...
    # Añadir una única colorbar para toda la figura
    if mappable_for_colorbar is not None:
        cbar_ax = fig.add_axes([0.92, 0.32, 0.015, 0.5]) 
        cbar = fig.colorbar(mappable_for_colorbar.get_children()[0], cax=cbar_ax, orientation='vertical')
        cbar_label = f'Ancho de Banda ({stat_title_part})'
        if latex_on:
            cbar.set_label(cbar_label, rotation=270, labelpad=18)
        else:
            cbar.set_label(cbar_label.replace('$', '').replace('\\', '').replace('{', '').replace('}', ''), rotation=270, labelpad=18)

    # fig_title_text = f'Mapas de Calor: Ancho {stat_title_part} de Banda por Caso'
    # if latex_on:
    #     fig.suptitle(fig_title_text, fontsize=16, y=0.98) 
    # else:
    #     fig.suptitle(fig_title_text.replace('$', '').replace('\\', '').replace('{', '').replace('}', ''), fontsize=16, y=0.98)

    plt.tight_layout(rect=[0.05, 0.15, 0.91, 0.93]) # Ajustado rect: [left, bottom, right, top] para etiquetas Y rotadas

    # --- Guardar Figura ---
    output_folder_p8 = os.path.join("./figuras", "p8") # Nombre de carpeta más específico
    if not os.path.exists(output_folder_p8):
        os.makedirs(output_folder_p8)

    full_save_path = os.path.join(output_folder_p8, 'heatmap.pdf')
    try:
        plt.savefig(full_save_path, dpi=300) 
        print(f"Heatmaps guardados en: {full_save_path}")
    except Exception as e:
        print(f"Error al guardar los heatmaps: {e}")
    plt.show()