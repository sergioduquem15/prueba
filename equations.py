
"""Este módulo contiene las ecuaciones alométricas para hacer el cálculo de la biomasa.
"""

import math
import pandas as pd
import numpy as np  


def hass_avocado(df_metrics: pd.DataFrame, col_diameter='mean') -> float:
    """Modelo alométrico para calcular la biomasa aérea (kg).
    
    Ecuación para aguacate Hass obtenida por Forestry.

    Parameters
    ----------
    df_metrics : pd.DataFrame
        Marco de datos con las características 
        geométricas de cada árbol para extraer
        el diámetro de copa.

    Returns
    -------
    float
        Biomasa aérea estimada del árbol (kg/árbol).
    """
    if col_diameter == 'mean':
        diameter = (df_metrics['minor_axis'] + df_metrics['mayor_axis']) / 2
    else:
        diameter = df_metrics[col_diameter]
    biomass = 0.8865 * (diameter ** 2.9265)
    return biomass