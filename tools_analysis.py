from skimage.morphology import square, erosion
from skimage.filters import meijering, sato, frangi, hessian
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.neighbors import KDTree
from shapely.geometry import Polygon, LineString, Point
from tqdm.contrib import itertools
import matplotlib.pyplot as plt
from collections import Counter
from pathlib import Path
import geopandas as gpd
from tqdm import tqdm
from skimage import data
from skimage import filters
import pandas as pd
import numpy as np
import rasterio
import torch
import math
import cv2

from yolov5_obb.equations import hass_avocado


def corr_cood(pred_c,x_off,y_off):
    pred_c[:,[0,2,4,6]] = pred_c[:,[0,2,4,6]] + x_off # x
    pred_c[:,[1,3,5,7]] = pred_c[:,[1,3,5,7]] + y_off # y
    return pred_c

def get_image_parameters(rgb_file):
    src = rasterio.open(rgb_file)
    xoff = src.transform.xoff
    yoff = src.transform.yoff
    a = src.transform.a
    b = src.transform.b
    d = src.transform.d
    e = src.transform.e

    res_sqr = (a**2+b**2)**0.5 
    return xoff, a, b, yoff, d, e, res_sqr

def image_representation(rgb, dsm, kernel_erosion=11):
    X_height, X_width, _ = rgb.shape
    # Paso 1: Leer la imagen y aplicar skimage.filters.sato
    sato_img = filters.sato(dsm)

    # Paso 2: Normalizar los valores entre 0 y 1
    sato_norm = np.interp(sato_img, (sato_img.min(), sato_img.max()), (0, 1))

    # Paso 3: Invertir los valores
    sato_inv = 1 - sato_norm

    # Paso 4: Aplicar erosión
    sato_inv_eroded = 1 - erosion(sato_norm, square(kernel_erosion))

    # Generar la nueva imagen con filtro del dsm para cada canal
    img_with = rgb.copy()
    img_with[:,:,0] = sato_inv_eroded * img_with[:,:,0]
    img_with[:,:,1] = sato_inv_eroded * img_with[:,:,1]
    img_with[:,:,2] = sato_inv_eroded * img_with[:,:,2]

    return img_with 

def clean_df(df2clean, ignore_index=True, p=0.85):
    # Crear estructura df salida
    COLUMNS = df2clean.columns
    df_cleanned = pd.DataFrame(columns=COLUMNS)
    # Seleccionar columnas de interés
    x_y = list(df2clean.columns).index('x_c_pxl'), list(df2clean.columns).index('y_c_pxl')
    # Ordenar según la probabilidad
    X = df2clean.sort_values('p', ascending=False).to_numpy()
    tree = KDTree(X[:, x_y]) #  only X and Y pox of COLUMNS
    index = list(range(len(X))) # index ordered from 0 to N 34172586
    while index:
        index_1st = index.pop(0)
        # Seleccionar columnas de interés
        axis = list(df2clean.columns).index('m_a_pxl'), list(df2clean.columns).index('M_a_pxl')
        # calcular el promedio del eje mayor y menor
        r = X[index_1st:index_1st + 1, axis].mean() * p
        ind = tree.query_radius(X[index_1st:index_1st + 1, x_y], r=r)[0]
        row = X[ind[0]]
        # Eliminar indices de la lista 'index'  cuando está dentro de un arbol seleccionado
        for i in ind:
            if i == index_1st:
                continue
            elif i in index:
                index.pop(index.index(i))
        # Agregar registro en df salida
        df_cleanned.loc[len(df_cleanned)] = row.round(3)
    return df_cleanned


def cross(row):
    x1 = row['x1']
    y1 = row['y1']
    x2 = row['x2']
    y2 = row['y2']
    x3 = row['x3']
    y3 = row['y3']
    x4 = row['x4']
    y4 = row['y4']

    # Definir los puntos del polígono
    p1 = Point(x1, y1)
    p2 = Point(x2, y2)
    p3 = Point(x3, y3)
    p4 = Point(x4, y4)

    # Crear el polígono
    poly = Polygon([p1, p2, p3, p4])

    # Encontrar los puntos medios entre los pares de puntos opuestos
    mid1 = LineString([p1, p4]).interpolate(0.5, normalized=True)
    mid2 = LineString([p2, p3]).interpolate(0.5, normalized=True)
    mid3 = LineString([p1, p2]).interpolate(0.5, normalized=True)
    mid4 = LineString([p3, p4]).interpolate(0.5, normalized=True)

    # Crear las líneas que forman la cruz
    line1 = LineString([mid1, mid2])
    line2 = LineString([mid3, mid4])
    return line1.union(line2)

def bbox(row):
    coords = [
        (row['x1'], row['y1']), 
        (row['x2'], row['y2']), 
        (row['x3'], row['y3']), 
        (row['x4'], row['y4'])
        ]
    poly = Polygon(coords)
    return poly


# FUNCIÖN PARA CREAR EL SHAPE
def create_shape(data, path2save, diameter='mean', crs=None, is_cross=True):
    DRIVER = 'ESRI Shapefile'
    BIOMASS_COL = 'BioM(kg)'
    CARBONO_COL = 'Carbono'
    CO2eq_COL = 'CO2eq'
    COLUMNS = ['x_center', 'y_center', 'minor_axis', 'mayor_axis', 'p', BIOMASS_COL, CARBONO_COL, CO2eq_COL]

    # Calculamos la biomasa y variables
    data[BIOMASS_COL] = hass_avocado(data, diameter)
    data[CARBONO_COL] = data[BIOMASS_COL]*0.43
    data[CO2eq_COL] = data[CARBONO_COL]*3.67

    # Convertimos las columnas en una lista de polígonos
    geom_c = []
    geom_b = []
    for i, row in data.iterrows():
        geom_c.append(cross(row))
        geom_b.append(bbox(row))
    
    if is_cross: geom = geom_c
    else: geom = geom_b
    
    # Creación de df de geometría
    gdf = gpd.GeoDataFrame(geometry=geom)
    gdf = gdf.merge(data[COLUMNS], left_index=True, right_index=True).round(3)   
    # Definimos el sistema de coordenadas
    if crs:
        # Exportamos el GeoDataFrame a un archivo .shp con el sistema de coordenadas definido
        gdf.to_file(path2save, driver=DRIVER, crs=crs)
    else:
        gdf.to_file(path2save, driver=DRIVER)

    return gdf
