# VAR_VECM_MEXICO_MODULO_CACHE.py

import streamlit as st
import requests
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import VAR, VECM
from statsmodels.tsa.vector_ar.vecm import coint_johansen
#from statsmodels.graphics.tsaplots import plot_acf
#import matplotlib.pyplot as plt

# --- FUNCIÓN AUXILIAR PARA OBTENER DATOS ---
def obtener_serie_banxico(id_serie, token, fecha_inicio):
    fecha_fin = pd.Timestamp.now().strftime('%Y-%m-%d')
    url = f"https://www.banxico.org.mx/SieAPIRest/service/v1/series/{id_serie}/datos/{fecha_inicio}/{fecha_fin}"
    headers = {"Bmx-Token": token}
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        data = response.json()
        serie_data = data['bmx']['series'][0]['datos']
        df_temp = pd.DataFrame(serie_data)
        df_temp['fecha'] = pd.to_datetime(df_temp['fecha'], format='%d/%m/%Y')
        df_temp.set_index('fecha', inplace=True)
        df_temp['dato'] = pd.to_numeric(df_temp['dato'], errors='coerce')
        return df_temp['dato']
    except Exception as e:
        st.error(f"Error al obtener la serie {id_serie}: {e}")
        return None

# --- FUNCIÓN PRINCIPAL (CON CACHÉ Y LÓGICA CORREGIDA) ---
@st.cache_data 
def generar_proyeccion_mexico(token, series_ids, start_date, anos_proyeccion, params_escenarios):
    """
    Función completa que ejecuta el análisis y devuelve los resultados.
    """
    # --- 1. Carga de Datos ---
    # CORRECCIÓN: Usa los parámetros de la función (token, start_date), no los por defecto.
    datos_api = {nombre: obtener_serie_banxico(id_serie, token, start_date) for nombre, id_serie in series_ids.items()}
    if not all(serie is not None for serie in datos_api.values()):
        return None

    tasa_mensual = datos_api["tasa_interes"].resample('MS').mean()
    tipo_cambio_mensual = datos_api["tipo_cambio"].resample('MS').mean()
    df = pd.concat([datos_api["inflacion"], tasa_mensual, np.log(tipo_cambio_mensual)], axis=1)
    df.columns = ['inflacion', 'tasa_interes', 'tipo_cambio']
    df.dropna(inplace=True)

    # --- 2. Análisis y Selección de Modelo ---
    variables_a_probar = ['tasa_interes', 'tipo_cambio']
    series_no_estacionarias = [col for col in variables_a_probar if adfuller(df[col].dropna())[1] >= 0.05]
    if len(series_no_estacionarias) < 2:
        usar_vecm, reconstruir_niveles = False, True
        df_modelo = df.copy()
        for col in series_no_estacionarias: df_modelo[col] = df_modelo[col].diff()
        df_modelo.dropna(inplace=True)
    else:
        num_relaciones_coint = sum(coint_johansen(df[series_no_estacionarias], 0, 1).lr1 > coint_johansen(df[series_no_estacionarias], 0, 1).cvt[:, 1])
        if num_relaciones_coint > 0:
            usar_vecm, reconstruir_niveles = True, False
            df_modelo = df.copy()
        else:
            usar_vecm, reconstruir_niveles = False, True
            df_modelo = df.copy()
            for col in series_no_estacionarias: df_modelo[col] = df_modelo[col].diff()
            df_modelo.dropna(inplace=True)
            
    # --- 3. Entrenamiento y Proyección ---
    n_periodos = anos_proyeccion * 12 # CORRECCIÓN: Usa el parámetro de años de proyección
    if usar_vecm:
        p = VAR(df_modelo).select_order(maxlags=12).aic
        resultados_modelo = VECM(df_modelo, k_ar_diff=p-1, coint_rank=num_relaciones_coint, deterministic='ci').fit()
        punto_proy, lim_inferior, lim_superior = resultados_modelo.predict(steps=n_periodos, alpha=0.05)
    else:
        resultados_modelo = VAR(df_modelo).fit(maxlags=12, ic='aic')
        y_input = df_modelo.values[-resultados_modelo.k_ar:]
        punto_proy, lim_inferior, lim_superior = resultados_modelo.forecast_interval(y=y_input, steps=n_periodos, alpha=0.05)
    
    fechas_futuras = pd.date_range(start=df.index[-1] + pd.DateOffset(months=1), periods=n_periodos, freq="MS")
    df_proy_punto = pd.DataFrame(punto_proy, index=fechas_futuras, columns=df.columns)
    df_proy_inferior = pd.DataFrame(lim_inferior, index=fechas_futuras, columns=df.columns)
    df_proy_superior = pd.DataFrame(lim_superior, index=fechas_futuras, columns=df.columns)
    
    if reconstruir_niveles:
        for col in series_no_estacionarias:
            for proj_df in [df_proy_punto, df_proy_inferior, df_proy_superior]:
                proj_df[col] = df[col].iloc[-1] + proj_df[col].cumsum()
    
    df_proy_nivel = pd.DataFrame({
        'inflacion': df_proy_punto['inflacion'],
        'inflacion_lim_inf': df_proy_inferior['inflacion'],
        'inflacion_lim_sup': df_proy_superior['inflacion']
    })

    # --- 4. Creación de Escenarios ---
    escenario_base = df_proy_nivel['inflacion'].copy()
    escenario_positivo = df_proy_nivel['inflacion_lim_inf'].copy()
    escenario_negativo = df_proy_nivel['inflacion_lim_sup'].copy()
    
    anos_modelo, meta_central, meta_baja, meta_alta, theta_central, theta_baja, theta_alta = params_escenarios['anos_modelo'], params_escenarios['meta_central'], params_escenarios['meta_baja'], params_escenarios['meta_alta'], params_escenarios['theta_central'], params_escenarios['theta_baja'], params_escenarios['theta_alta']

    fecha_transicion = escenario_base.index[anos_modelo * 12]
    
    for t in escenario_base.loc[fecha_transicion:].index:
        escenario_base.loc[t] = escenario_base.loc[t - pd.DateOffset(months=1)] + theta_central * (meta_central - escenario_base.loc[t - pd.DateOffset(months=1)])

    for t in escenario_positivo.loc[fecha_transicion:].index:
        escenario_positivo.loc[t] = escenario_positivo.loc[t - pd.DateOffset(months=1)] + theta_baja * (meta_baja - escenario_positivo.loc[t - pd.DateOffset(months=1)])

    for t in escenario_negativo.loc[fecha_transicion:].index:
        escenario_negativo.loc[t] = escenario_negativo.loc[t - pd.DateOffset(months=1)] + theta_alta * (meta_alta - escenario_negativo.loc[t - pd.DateOffset(months=1)])
        

    # --- Analisis de residuos ---

    # Extraer los residuos de la variable 'inflacion' de forma robusta
    if usar_vecm:
        residuos_inflacion = resultados_modelo.resid[:, 0]
    else:
        if isinstance(resultados_modelo.resid, pd.DataFrame):
            residuos_inflacion = resultados_modelo.resid['inflacion']
        else:
            residuos_inflacion = resultados_modelo.resid[:, 0]


    # --- Resumen estadístico ---

    resumen_texto = str(resultados_modelo.summary())

    # --- 5. Preparación de Resultados Finales ---
    promedios = { "Base": escenario_base.mean(), 
                 "Positivo": escenario_positivo.mean(), 
                 "Negativo": escenario_negativo.mean() 
                }
  
    df_resumen_escenarios = pd.DataFrame({
        'Promedio (%)': [promedios['Base'], promedios['Positivo'], promedios['Negativo']],
        'Volatilidad (Desv. Est.)': [escenario_base.std(), escenario_positivo.std(), escenario_negativo.std()],
        'Máximo (%)': [escenario_base.max(), escenario_positivo.max(), escenario_negativo.max()],
        'Mínimo (%)': [escenario_base.min(), escenario_positivo.min(), escenario_negativo.min()]
    }, index=['Base', 'Positivo', 'Negativo'])

    df_resumen_escenarios = df_resumen_escenarios.round(3)

    resultados = {
    "df_historico": df,
    "escenario_base": escenario_base,
    "escenario_positivo": escenario_positivo,
    "escenario_negativo": escenario_negativo,
    "promedios": promedios,
    "tabla_escenarios": df_resumen_escenarios,
    "modelo_usado": "VECM" if usar_vecm else "VAR",
    "anos_proyectados": anos_proyeccion,
    "series_no_estacionarias": series_no_estacionarias,
    "relaciones_coint": num_relaciones_coint if 'num_relaciones_coint' in locals() else 0,
    #"fig_acf": fig_acf,
    #"fig_hist": fig_hist,
    "resumen_texto": resumen_texto,
    "residuos": pd.Series(residuos_inflacion)
    }
    
    return resultados