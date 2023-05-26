import pandas as pd
import os
from yolov5_obb.dataframe_analysis import Analysis
from yolov5_obb.tools_analysis import corr_cood, get_image_parameters, image_representation, clean_df, create_shape

def fail(name='',cl=[0.5]):
	archivo = name
	df = pd.read_csv(f"./resultados/dataframes/{archivo[:-3]}/{archivo}.csv")
	path_rgb = f"./data/rgb/Recorte_{name[:-3]}.tif"
	p_clean = cl
	try: os.mkdir(f"./resultados/dataframes/{archivo[:-3]}/d_{archivo[:-3]}/")
	except: pass

	for i in p_clean:
		folder = f"{archivo}_{int(i*100)}"
		df_ = df.copy()
		an = Analysis(df_)
		df_full = an.pixels2coords(*get_image_parameters(path_rgb))
		df_full.sort_values('p', ascending=False)
		df_full_clean = clean_df(df_full,p=i)
		df_full_clean.to_csv(f"./resultados/dataframes/{archivo[:-3]}/d_{archivo[:-3]}/{folder}.csv",index=False)

def fail_shape(name='',cl =[50,63], p=0.5):
	archivo = name
	df1 = pd.read_csv(f"./resultados/dataframes/{archivo}/d_{archivo}/{archivo}_{cl[0]}_60.csv")
	df2 = pd.read_csv(f"./resultados/dataframes/{archivo}/d_{archivo}/{archivo}_{cl[1]}_50.csv")

	df_concat = pd.concat([df1, df2], axis=0)
	df_concat.sort_values('p', ascending=False)
	df_full_clean = clean_df(df_concat, p=p)
	

	try: os.mkdir(f"./resultados/shapes/")
	except: pass

	try: os.mkdir(f"./resultados/shapes/{archivo}/")
	except: pass
	
	df_full_clean.to_csv(f"./resultados/shapes/{archivo}/{archivo}.csv",index=False)
	shape = create_shape(df_full_clean, f'./resultados/shapes/{archivo}/{archivo}.shp', 'mean', 'EPSG:3116', is_cross=True)
	print("\n",shape.head(5),"\n")
	print(f"Numero de predicciones para el shape generado de {archivo}:",len(shape),"\n")
	print("Fin del programa...\n")

def nclean(name=''):
	archivo = name
	df = pd.read_csv(f"./resultados/dataframes/{archivo[:-3]}/d_{archivo[:-3]}/{archivo}_50.csv")
	df.sort_values('p', ascending=False)
	df_clean = clean_df(df, p=0.6)


	try: os.mkdir(f"./resultados/shapes_clean/")
	except: pass
	try: os.mkdir(f"./resultados/shapes_clean/{archivo}/")
	except: pass
	
	shape = create_shape(df_clean, f'./resultados/shapes_clean/{archivo}/{archivo}.shp', 'mean', 'EPSG:3116', is_cross=True)
	print("\n",shape.head(5),"\n")
	print(f"Numero de predicciones para el shape generado de {archivo}:",len(shape),"\n")
	print("Fin del programa...\n")


	