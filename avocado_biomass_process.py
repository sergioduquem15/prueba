import os
import time
import cv2
import numpy as np
import warnings
import pandas as pd
import sys
from tqdm.contrib import itertools

from yolov5_obb.detect_v9 import YOLOv5_OBB
from yolov5_obb.dataframe_analysis import Analysis
from yolov5_obb.tools_analysis import corr_cood, get_image_parameters, image_representation, clean_df, create_shape

def get_images_folder(path_data):
    folder_rgb = os.listdir(f"{path_data}rgb/")
    folder_dsm = os.listdir(f"{path_data}dsm/")
    print(folder_rgb)
    return folder_rgb, folder_dsm

def dsm_scale(rgb, dsm):
    image_rgb = cv2.imread(rgb, 1)
    X_height, X_width, _ = image_rgb.shape

    if dsm == None:
        return image_rgb, X_height, X_width
    else:
        image_dsm = cv2.imread(dsm, -1)
        second_min = np.min(image_dsm[image_dsm != np.min(image_dsm)]) # get second min becasue nan default in qgis is -32767.0
        image_dsm[image_dsm<second_min]=second_min - 1e-3 # making different from the last one
        image_dsm = cv2.resize(image_dsm, (X_width, X_height))

        return image_rgb,image_dsm,X_height,X_width
    
def process(image_rgb, image_dsm, X_height, X_width, obj):
    COLUMNS = ['x1','y1','x2','y2','x3','y3','x4','y4','p']
    df = pd.DataFrame(columns=COLUMNS)

    size_crop = 640
    stride = size_crop//2
    i_range = np.append(np.arange(0, X_height - size_crop, stride), X_height - size_crop)
    j_range = np.append(np.arange(0, X_width - size_crop, stride), X_width - size_crop)

    for i, j in itertools.product(i_range, j_range, desc="result"):
        crop_rgb = image_rgb[i : i + size_crop, j : j + size_crop, :]
        try: crop_dsm = image_dsm[i : i + size_crop, j : j + size_crop]
        except: pass

        if type(image_dsm) != list: img_with = crop_rgb
        else: img_with = image_representation(crop_rgb, crop_dsm)

        if np.mean(img_with==0.0) > 0.8:
            continue
        pred = obj.run(img_with) # prediccion
        pred_c = corr_cood(pred,j,i) # correccion a prediccion
        df = pd.concat([df,pd.DataFrame(pred_c, columns=COLUMNS)], ignore_index=True)
    
    return df

def clean(path_rgb,df,conf):
    df_ = df.copy()
    an = Analysis(df_)
    df_full = an.pixels2coords(*get_image_parameters(path_rgb))
    df_full.sort_values('p', ascending=False)
    df_full_clean = clean_df(df_full,p=conf)
    return df_full_clean



def make_dataframes(isLD : bool = True, conf_th : list = [0.5,0.63], dev : str = 'cpu', path : str = '.'):
    if isLD: weight = f'./yolov5_obb/runs/train/exp/weights/best_n.pt'
    else: weight = f'./yolov5_obb/runs/train/exp/weights/best_m5.pt'
        
    folder_rgb, folder_dsm = get_images_folder(path)
    
    df_list = []
    path_list = []
    name_list = []
    namef_list = []
    for c in conf_th:
        dfol = []
        prfol = []
        nafol = []
        nrafol = []
        for n_process, path_rgb in enumerate(folder_rgb):
            prgb = f"{path}rgb/{path_rgb}"
            path_dsm = path_rgb[:-4]+"_dsm.tif"
            #print(path_dsm)
            name = path_rgb.split("/")[-1]
            name = name.split(".")[0]
            name = name.split("_")[1:]
            name = "_".join(map(str,name))
            
            model = YOLOv5_OBB(weights=weight,
                               augment=True,
                               source=path_rgb,
                               imgsz=[640,640],
                               conf_thres=c,
                               device=dev,
                               agnostic_nms=True,
                               save_txt=True)
            
            if path_dsm in folder_dsm:
                print(n_process+1,"-", name,"(Tiene archivo DSM), umbral de confianza:",c)
                print("GENERANDO PREDICCIONES...")
                   
                image_rgb,image_dsm,X_height,X_width = dsm_scale(prgb, f"{path}dsm/{path_dsm}")
                df = process(image_rgb, image_dsm, X_height, X_width, model)
                
            else:
                print(n_process+1,"-", name,"(No tiene archivo DSM), umbral de confianza:",c)
                print("GENERANDO PREDICCIONES...")
                
                image_rgb, X_height, X_width = dsm_scale(prgb, None)
                df = process(image_rgb, None, X_height, X_width, model)
            
            try: os.mkdir(f"./resultados/dataframes/")
            except: pass
            
            try: os.mkdir(f"./resultados/dataframes/{name}/")
            except: pass
            
            pathr = f"./resultados/dataframes/{name}/"
            namef = f"{name}_{int(c*100)}"
            df.to_csv(f"./resultados/dataframes/{name}/{namef}.csv",index=False)
            print("Se ha guardado el dataframe en la ruta:", pathr,"\n")
            
            dfol.append(df)
            prfol.append(prgb)
            nafol.append(name)
            nrafol.append(namef)
            
            print(f"Numero de predicciones resultantes para {name} con umbral de confianza {c}:",len(df),"\n")
        df_list.append(dfol)
        path_list.append(prfol)
        name_list.append(nafol)
        namef_list.append(nrafol)

    return df_list, path_list, name_list, namef_list

def main(path,conf : list = [0.5,0.63]):
    print("Inicio del programa...\n")
    inicio = time.time()
    
    all = []
    df_list, prgb, name, namef = make_dataframes(path="./data/",conf_th=conf)
    for path, df, n, nf in zip(prgb, df_list, name, namef):
        for path_e, df_e, n_e, nf_e in zip(path, df, n, nf):
            all.append([path_e, df_e, n_e, nf_e])

    for x in range(int(len(all)/2)):
        all_50 = all[x]
        all_63 = all[x+int(len(all)/2)]

        #all_50[0] = b
        #all_63[0] = b

        df50_60 = clean(all_50[0],all_50[1],0.6)
        df63_50 = clean(all_63[0],all_63[1],0.5)

        df_full = pd.concat([df50_60, df63_50], ignore_index=True)
        df_full.sort_values('p', ascending=False)
        df_full_clean = clean_df(df_full, p=0.5)

        try: os.mkdir(f"resultados/shapes/")
        except: pass
        try: os.mkdir(f"resultados/shapes/{all_50[2]}/")
        except: pass

        shape = create_shape(df_full_clean, f'resultados/shapes/{all_50[2]}/{all_50[2]}.shp', 'mean', 'EPSG:3116', is_cross=True)

        print("\n",shape.head(5),"\n")
        print(f"Numero de predicciones para el shape generado de {all_50[2]}:",len(shape),"\n")

    print("Fin del programa...\n")
    final = time.time()
    print("Tiempo de ejecución:", final - inicio)
    
    #return shape
