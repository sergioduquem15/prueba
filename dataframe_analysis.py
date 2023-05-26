import pandas as pd
import numpy as np
#from osgeo import ogr, osr, gdal

class Analysis:
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.data_len = len(self.dataframe)

    def center_b2p(self,box):
        box = box.to_numpy()[:,:-1].reshape(self.data_len,4,2)
        M_all = []
        for elem in box:
            dot = []
            for ld in range(4):
                ld = ld+1
                if ld == 4: n = 4
                else: n = 0
                X_1, Y_1 = elem[ld-1][0], elem[ld-1][1]
                X_2, Y_2 = elem[ld-n][0], elem[ld-n][1]

                M = [(X_2 + X_1)/2 , (Y_2 + Y_1)/2]
                dot.append(M)
            M_all.append(dot)
        return M_all
    
    def rect_center(self,c):
        x11,y11 = c[0]
        x12,y12 = c[2]
        x21,y21 = c[1]
        x22,y22 = c[3]
        if (x11-x12 == 0) or (x21 - x22 == 0):
            x = x21
            y = y11
        else:
            m1 = (y11-y12)/(x11-x12)
            m2 = (y21-y22)/(x21-x22)
            x = ((y11-y21) + (m2*x21 - m1*x11))/(m2 - m1)
            y = y21 + m2*( (y11 - y21 + m1*(x21 - x11)) / ( m2 - m1 ) )
        return [x,y]
    
    def distancia(self,box):
        box = box.to_numpy()[:,:-1].reshape(self.data_len,4,2)
        dtotal = []
        for elem in box:
            d_all = []
            for ld in range(4):
                ld = ld+1
                if ld == 4: n = 4
                else: n = 0
                X_1, Y_1 = elem[ld-1][0], elem[ld-1][1]
                X_2, Y_2 = elem[ld-n][0], elem[ld-n][1]
                d = np.sqrt( (X_2 - X_1)**2 + (Y_2 - Y_1)**2 )
                d_all.append(d)
            r = [min(d_all),max(d_all)]
            dtotal.append(r)
        return dtotal
    
    def process(self):
        data = self.dataframe
        lcenter = self.center_b2p(data)
        dlist = self.distancia(data)
        center_x = []
        center_y = []
        eje_menor = []
        eje_mayor = []
        for elem in lcenter:
            try:
                center = self.rect_center(elem)
                center_x.append(center[0])
                center_y.append(center[1])
            except RuntimeError:
                center_x.append(None)
                center_y.append(None)

        for elem in dlist:
            eje_menor.append(elem[0])
            eje_mayor.append(elem[1])        

        data["x_center"] = center_x
        data["y_center"] = center_y
        data["eje_menor"] = eje_menor
        data["eje_mayor"] = eje_mayor

        return data

    def pixels2coords(self, xoff, a, b, yoff, d, e, res_sqr): 
        """Returns global coordinates from coordinates x, y of the pixel"""
        data = self.process()
        x1 = a * data['x1'] + b * data['y1'] + xoff 
        x2 = a * data['x2'] + b * data['y2'] + xoff 
        x3 = a * data['x3'] + b * data['y3'] + xoff 
        x4 = a * data['x4'] + b * data['y4'] + xoff 

        y1 = d * data['x1'] + e * data['y1'] + yoff 
        y2 = d * data['x2'] + e * data['y2'] + yoff 
        y3 = d * data['x3'] + e * data['y3'] + yoff 
        y4 = d * data['x4'] + e * data['y4'] + yoff 

        p = data['p'] 
        x_center = a * data['x_center'] + b * data['y_center'] + xoff
        y_center = d * data['x_center'] + e * data['y_center'] + yoff

        minor_axs = data['eje_menor'] * res_sqr
        mayor_axs = data['eje_mayor'] * res_sqr

        serie = pd.DataFrame( { 
                            'x1_pxl' : data['x1'], 'y1_pxl' : data['y1'],
                            'x1': x1,'y1': y1,
                            'x2_pxl' : data['x2'], 'y2_pxl' : data['y2'],
                            'x2': x2,'y2': y2,
                            'x3_pxl' : data['x3'], 'y3_pxl' : data['y3'],
                            'x3': x3,'y3': y3,
                            'x4_pxl' : data['x4'], 'y4_pxl' : data['y4'],
                            'x4': x4,'y4': y4,
                            'p': p,
                            'x_c_pxl' : data['x_center'], 'y_c_pxl' : data['y_center'],
                            'x_center' : x_center, 'y_center' : y_center,
                            'm_a_pxl' : data['eje_menor'], 'M_a_pxl' : data['eje_mayor'],
                            'minor_axis' : minor_axs, 'mayor_axis' : mayor_axs } )
        return serie 