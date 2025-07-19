# dashboard.py

# --- 0. LIBRERÍAS ---
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
from statsmodels.tsa.stattools import acf
import numpy as np

# --- 1. IMPORTAR MÓDULOS DE ANÁLISIS ---

from VAR_VECM_MEXICO_MODULO_CACHE2 import generar_proyeccion_mexico
from VAR_VECM_USA_MODULO_CACHE import generar_proyeccion_usa

# --- 2. CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Dashboard de Proyecciones", layout="wide", page_icon="📊")


# --- 3. BARRA LATERAL CON MENÚ DE NAVEGACIÓN ---
with st.sidebar:
    st.title("Análisis Económico")
    st.info("Selecciona el análisis y ajusta los parámetros aquí.")
    
    pagina_seleccionada = option_menu(
        menu_title="Menú Principal",
        options=["Bienvenida", "Variables Económicas"],
        icons=["house", "graph-up-arrow"],
        menu_icon="cast",
        default_index=0
    )


# --- 4. CONTENIDO DE CADA PÁGINA ---

# --- PÁGINA DE BIENVENIDA ---
if pagina_seleccionada == "Bienvenida":
    st.header("Bienvenido al Dashboard de Proyecciones")
    st.markdown("""
    Esta aplicación interactiva proporciona proyecciones económicas y financieras utilizando modelos econométricos y simulaciones de Monte Carlo.
    **Selecciona un análisis en el menú de la izquierda para comenzar.**
    """)


# --- PÁGINA DE INFLACIÓN MÉXICO ---
if pagina_seleccionada == "Variables Económicas":
    st.title("Análisis de Variables Económicas")

    # Creamos las pestañas secundarias para cada análisis específico
    analisis_seleccionado = option_menu(
        menu_title=None,
        options=["Inflación México", "Inflación Estados Unidos", "S&P 500", "EMBI", "Beta Desampalancada", "Deuda Largo Plazo", "Bonos 20 años"],
        icons=['bank', 'flag-fill', 'graph-up-arrow', 'shop', 'bank2', 'piggy-bank', 'currency-dollar'], # Iconos de Bootstrap
        orientation="horizontal",
        styles={
            "container": {"padding": "0!important", "background-color": "#fafafa"},
            "icon": {"color": "#607D8B", "font-size": "20px"}, 
            "nav-link": {"font-size": "16px", "text-align": "center", "margin":"0px", "--hover-color": "#eee"},
            "nav-link-selected": {"background-color": "#0a3161"},
        }
    )    

    # --- Contenido de la Pestaña de Inflación México ---
    if analisis_seleccionado == "Inflación México":
        st.header("Proyección de Inflación para México")

        # Pestañas anidadas 
        tab_metodologia, tab_proyeccion, tab_diagnosticos, tab_descarga = st.tabs([
            "📄 Metodología", 
            "📈 Proyección", 
            "🩺 Diagnósticos", 
            "📥 Descarga de Datos"
        ])

    # --- Contenido de la Pestaña de Metodología ---
        with tab_metodologia:
            st.subheader("Metodología del Análisis Econométrico")


            st.markdown("A continuación se describe el proceso para generar las proyecciones de inflación.")
    

            col1, col2 = st.columns(2, gap="large")

            with col1:
                with st.container(border=True):
                    st.subheader("Modelo Econométrico")
                    st.markdown("""
                    El núcleo del análisis es un modelo econométrico que captura la interdependencia entre las variables clave de la política monetaria.

                    - **Selección Automática:** El script elige entre un modelo **VAR** (Vector Autorregresivo) o **VECM** (Vector de Corrección de Errores) basándose en pruebas estadísticas formales:
                        1.  **Prueba de Estacionariedad (ADF):** Para identificar si las series tienen una tendencia persistente.
                        2.  **Prueba de Cointegración (Johansen):** Para determinar si las series no estacionarias mantienen una relación de equilibrio a largo plazo.
                    """)

                with st.container(border=True):
                    st.subheader("Lógica de Escenarios")
                    st.markdown("""
                    Las proyecciones a 30 años se construyen en dos etapas para asegurar su realismo:

                    1.  **Pronóstico del Modelo (Primeros 5 Años):** Se utiliza el pronóstico directo del modelo VAR/VECM, incluyendo sus intervalos de confianza para definir los escenarios optimista y pesimista.
                    2.  **Convergencia Suave a la Meta:** Después de los primeros 5 años, los escenarios son guiados gradualmente hacia una meta de largo plazo (ej. 3% para Banxico), reflejando la expectativa de que la política monetaria eventualmente anclará la inflación.
                    """)

            with col2:
                with st.container(border=True):
                    st.subheader("Variables Utilizadas:")
                    st. markdown(""" 
                    * **Inflación (Variable a Proyectar):** Se usa la variación anual del Índice Nacional de Precios al Consumidor (INPC). Es la medida oficial de la inflación en el país.
                        - *ID de Serie (Banxico):* `SP30578`
                                
                    * **Tasa de Interés:** Se utiliza la Tasa de Interés Interbancaria de Equilibrio (TIIE) a 28 días. Refleja la postura del Banco de México y el costo del crédito a corto plazo.
                        - *ID de Serie (Banxico):* `SF43783`
                                
                    * **Tipo de Cambio:** Se emplea el Tipo de Cambio FIX (USD/MXN). Es el principal canal de transmisión de shocks externos a la economía mexicana y afecta los precios de los bienes importados. Se utiliza su logaritmo para estabilizar la varianza.
                        - *ID de Serie (Banxico):* `SF43718`
                                
                    **Para hacer uso del modelo es necesario que el usuario cuente con un token de consulta. Si el usuario aún no cuenta con uno puede descargarlo en la siguiente dirección: https://www.banxico.org.mx/SieAPIRest/service/v1/token**""")
                 

    # --- Contenido de la Pestaña de Proyección ---
        with tab_proyeccion:
            st.subheader("Generar Proyección")  
            st.divider()

            # Dividimos la página en una columna principal y una lateral para el formulario
            col_espacio1, col_analisis, col_espacio2, col_formulario = st.columns([0.5, 5.0, 0.5, 1])

            with col_formulario:
                st.subheader("Parámetros")
                with st.form(key="form_mex"):
                    token_banxico = st.secrets["TOKEN_BANXICO"]
                    start_date_mex = st.date_input("Fecha de Inicio", pd.to_datetime("2002-01-01"))
                    anos_proyeccion_mex = st.number_input("Años a Proyectar", 5, 50, 30)

                    col_metas1, col_metas2 = st.columns(2)
                    with col_metas1:
                        meta_central = st.number_input("Meta Central (%)", value=3.0, step=0.1)
                        meta_baja = st.number_input("Meta Baja (%)", value=3.0, step=0.1)
                    with col_metas2:
                        meta_alta = st.number_input("Meta Alta (%)", value=5.5, step=0.1)
                        
                    submit_button = st.form_submit_button(label="Generar Proyección")

            with col_analisis:

                if submit_button:
                    if not token_banxico:
                        st.warning("Por favor, ingresa un Token de Banxico válido.")
                    else:
                        with st.spinner("Ejecutando modelo econométrico..."):
                            # Parámetros para los escenarios (puedes moverlos a la barra lateral después)
                            params_escenarios = {
                                'anos_modelo': 5, 'meta_central': meta_central, 'meta_baja': meta_baja, 'meta_alta': meta_alta,
                                'theta_central': 0.030, 'theta_baja': 0.015, 'theta_alta': 0.050
                            }
                            series_ids = {
                                "inflacion": "SP30578", "tasa_interes": "SF43783", "tipo_cambio": "SF43718"
                            }
                            
                            # Llamamos a la función del módulo
                            resultados = generar_proyeccion_mexico(
                                token=token_banxico,
                                series_ids=series_ids,
                                start_date=start_date_mex.strftime("%Y-%m-%d"),
                                anos_proyeccion=anos_proyeccion_mex,
                                params_escenarios=params_escenarios
                            )

                        if resultados:
                            st.session_state['resultados'] = resultados # Guarda los resultados obtenidos en la pestaña proyeccion para visualizar diagnostico
                            st.success("Proyección generada exitosamente.")
                            
                            # Diccionarios con los resultados
                            promedios = resultados["promedios"]
                            df_historico = resultados["df_historico"]
                            escenario_base = resultados["escenario_base"]
                            escenario_positivo = resultados["escenario_positivo"]
                            escenario_negativo = resultados["escenario_negativo"]
                            
                            # Mostrar KPIs (1/3)
                            with st.container():
                                st.subheader("Promedios Proyectados")
                                col0, col1, col2, col3, col4 = st.columns([3, 3, 3, 3, 3])
                                col1.metric("Escenario Base", f"{promedios['Base']:.2f}%")
                                col2.metric("Escenario Positivo", f"{promedios['Positivo']:.2f}%")
                                col3.metric("Escenario Negativo", f"{promedios['Negativo']:.2f}%")

                            st.divider()

                            # --- Creación de la Gráfica Interactiva con Plotly ---
                            st.subheader("Gráfica de Proyección")

                            df_historico_serie = resultados["df_historico"]['inflacion']
                            ultimo_punto_historico = df_historico_serie.iloc[-1:]
                            base_para_graficar = pd.concat([ultimo_punto_historico, resultados['escenario_base']])
                            positivo_para_graficar = pd.concat([ultimo_punto_historico, resultados['escenario_positivo']])
                            negativo_para_graficar = pd.concat([ultimo_punto_historico, resultados['escenario_negativo']])

                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=df_historico_serie.index, y=df_historico_serie, mode='lines', name='Histórico', line=dict(color='#BBBBBB', width=3)))
                            fig.add_trace(go.Scatter(x=base_para_graficar.index, y=base_para_graficar, mode='lines', name=f"Base (Prom: {promedios['Base']:.2f}%)", line=dict(color='#003366', width=4)))
                            fig.add_trace(go.Scatter(x=positivo_para_graficar.index, y=positivo_para_graficar, mode='lines', name=f"Positivo (Prom: {promedios['Positivo']:.2f}%)", line=dict(color='#6699CC', dash='dash')))
                            fig.add_trace(go.Scatter(x=negativo_para_graficar.index, y=negativo_para_graficar, mode='lines', name=f"Negativo (Prom: {promedios['Negativo']:.2f}%)", line=dict(color='#666666', dash='dash')))
                            
                            fig.add_hline(y=3.0, line_dash="dot", line_color="black", annotation_text="Meta Banxico (3%)", annotation_position="bottom right")

                            # Configurar el diseño de la gráfica
                            fig.update_layout(
                                title_text=f"Proyección de inflación en México a {anos_proyeccion_mex} años",
                                xaxis_title="Fecha",
                                yaxis_title="Inflación Anualizada (%)",
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                                template="plotly_white",
                                font=dict(
                                    family="Arial, sans-serif",
                                    size=12,
                                    color="black"
                                ),
                                height=600,
                                xaxis=dict(gridcolor='#EAEAEA'), # Color de la cuadrícula
                                yaxis=dict(gridcolor='#EAEAEA')
                            )
                            
                            # Mostrar la gráfica de Plotly (2/3)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            st.divider()

                            # Tabla comparativa de escenarios (3/3)

                            with st.container():
                                st.subheader("Comparativa de Escenarios")

                                col0, col1, col2 = st.columns([1, 2, 1])

                                with col1:      
                                    st.dataframe(resultados['tabla_escenarios'], use_container_width=True)   

                        else:
                            st.error("Ocurrió un error al generar la proyección.")
                else:
                    st.info("Ingresa los parámetros en el formulario y haz clic en 'Generar Proyección' para ver los resultados.")
   
    # --- Contenido de la Pestaña de Diagnósticos ---
        with tab_diagnosticos:
            st.subheader("Diagnósticos del Modelo")
            st.divider() # Crea una linea divisoria

            # Verificamos si ya se generó una proyección QUIZAS TENER CUIDADO CON LA PALABRA RESULTADOS, puede ser resultados_mex revisar codigo gemini con eso
            if 'resultados' in st.session_state:
                resultados = st.session_state['resultados']

                with st.container():

                    # Pruebas hechas (1/3)
                    st.markdown("#### Pruebas hechas para decidir entre VAR/VECM")
                    st.info(f"El modelo seleccionado automáticamente para esta proyección fue un **{resultados['modelo_usado']}**.")

                    col_espacio1, col_prueba1, col_espacio2, col_prueba2, col_espacio3 = st.columns([2, 3, 2, 3, 2])

                    with col_prueba1:
                        st.markdown("**Prueba de Estacionariedad (Dickey-Fuller)**")

                        nombres_mapa = {
                                    'inflacion': 'inflación',
                                    'tasa_interes': 'tasa de interés',
                                    'tipo_cambio': 'tipo de cambio'
                        }

                        variables_no_est = resultados['series_no_estacionarias']

                        if variables_no_est:
                            nombres_legibles = [nombres_mapa.get(var, var) for var in variables_no_est]
                            lista_series = ", ".join(nombres_legibles)
                        else:
                            lista_series = "Ninguna"

                        st.markdown(f"Resultado de la prueba de estacionariedad se concluyo que las siguientes variables no son estacionarias: **{lista_series}**.")

                    with col_prueba2:
                        st.markdown("**Prueba de Cointegración (Johansen)**")
                        # Mostramos el número de relaciones de cointegración encontradas
                        num_rel = resultados['relaciones_coint']
                        st.markdown(f"Resultado de la prueba de cointegración se concluyo que el siguiente número de variables tienen relación: **{num_rel}**")
                        if num_rel > 0:
                            st.markdown("_Esto justifica el uso de un modelo VECM._")
                        else:
                            st.markdown("_Al no encontrar cointegración en los datos, se decidío utilizar un modelo VAR aplicando diferencias._")

                st.divider()

                with st.container():

                    # Sección de Análisis de Residuos (2/3)
                    st.subheader("Análisis de Residuos")
                    st.write("Un buen modelo debe tener errores (residuos) que se comporten como ruido aleatorio, sin patrones.")
                
                    # Usamos columnas para mostrar las gráficas lado a lado
                    col_espacio1, col_acf, col_espacio2, col_hist, col_espacio3 = st.columns([1, 4, 1, 4, 1])

                    with col_acf:
                        st.markdown("**Autocorrelación (ACF)**")

                        # Calcular ACF y los intervalos de confianza
                        acf_values, confint = acf(resultados['residuos'], nlags=24, alpha=0.05)
                        
                        fig_acf = go.Figure()
                        
                        # Banda de confianza (sombreado azul)
                        conf_upper = confint[1:, 1] - acf_values[1:]
                        conf_lower = confint[1:, 0] - acf_values[1:]
                        x_axis = np.arange(1, 25)
                        fig_acf.add_trace(go.Scatter(x=np.concatenate([x_axis-1, 1+x_axis[::-1]]), y=np.concatenate([conf_upper, conf_lower[::-1]]), fill='toself', fillcolor='rgba(173, 216, 230, 0.5)', line=dict(color='rgba(255,255,255,0)'), showlegend=False))
                        fig_acf.add_trace(go.Bar(x=x_axis, y=acf_values[1:], name='ACF', width=0.2)) # 'width' hace las barras más finas          
                        fig_acf.update_layout(template="plotly_white", height=400, title_text="Autocorrelación de Residuos")
                        st.plotly_chart(fig_acf, use_container_width=True)

                        st.markdown("**Interpretación:** Para que el modelo sea válido, la mayoría de las barras deben estar **dentro del área sombreada**.")

                    with col_hist:
                        st.markdown("**Histograma**")
                        fig_hist = go.Figure()
                        fig_hist.add_trace(go.Histogram(x=resultados['residuos'], nbinsx=25, xbins=dict(size=0.1), name='Frecuencia',marker=dict(color='#6699CC',line=dict(color='#003366', width=1) )))
                        fig_hist.update_layout(template="plotly_white", height=400, title_text="Histograma de Residuos", bargap=0.05)
                        st.plotly_chart(fig_hist, use_container_width=True)

                        st.markdown("**Interpretación:** La distribución de los errores debe parecerse a una **campana (distribución normal)**. Esto sugiere que los errores del modelo son aleatorios y no están sesgados.")
            
                st.divider()

                # Resumen estadístico (3/3)
                with st.expander("Ver Resumen Estadístico Completo"):
                    st.text(resultados['resumen_texto'])

            else:
                st.warning("Debes generar una proyección en la pestaña '📈 Proyección' para ver los diagnósticos del modelo.")

    # --- Contenido de la Pestaña de Descarga ---
        with tab_descarga:
            st.subheader("Descargar Datos de la Proyección")
            st.divider()
            
            # 1. Verificar si los resultados existen en la memoria de la sesión
            if 'resultados' in st.session_state:
                resultados = st.session_state['resultados']
                
                # 2. Preparar un DataFrame uniendo los tres escenarios proyectados
                df_para_descarga = pd.DataFrame({
                    'Base': resultados['escenario_base'],
                    'Positivo': resultados['escenario_positivo'],
                    'Negativo': resultados['escenario_negativo']
                })
                
                # Formatear los números a 2 decimales para el CSV
                df_para_descarga = df_para_descarga.round(2)
                
                # 3. Convertir el DataFrame a formato CSV en memoria
                #    .to_csv() lo convierte a texto
                #    .encode('utf-8') lo convierte a bytes, que es lo que el botón necesita
                csv_data = df_para_descarga.to_csv(index=True, encoding='utf-8')
                
                # 4. Crear el botón de descarga
                st.download_button(
                label="Descargar Proyección (CSV)", # Texto del botón
                data=csv_data,                      # Los datos a descargar
                file_name='proyeccion_inflacion_mexico.csv', # Nombre del archivo
                mime='text/csv',                      # Tipo de archivo
                )
            else:
                # Mensaje si aún no se ha corrido el análisis
                st.warning("Debes generar una proyección en la pestaña '📈 Proyección' para poder descargar los datos.")


















    # --- Contenido de la Pestaña de Inflación EE.UU. ---
    if analisis_seleccionado == "Inflación Estados Unidos":
        st.header("Proyección de Inflación para EE.UU.")
 
        tab_metodologia, tab_proyeccion, tab_diagnosticos, tab_descarga = st.tabs([
            "📄 Metodología", 
            "📈 Proyección", 
            "🩺 Diagnósticos", 
            "📥 Descarga de Datos"
        ])

    # --- Contenido de la Pestaña de Metodología ---
        with tab_metodologia:
            st.subheader("Metodología del Análisis Econométrico")


            st.markdown("A continuación se describe el proceso para generar las proyecciones de inflación.")
    

            col1, col2 = st.columns(2, gap="large")

            with col1:
                with st.container(border=True):
                    st.subheader("Modelo Econométrico")
                    st.markdown("""
                    El núcleo del análisis es un modelo econométrico que captura la interdependencia entre las variables clave de la política monetaria.

                    - **Selección Automática:** El script elige entre un modelo **VAR** (Vector Autorregresivo) o **VECM** (Vector de Corrección de Errores) basándose en pruebas estadísticas formales:
                        1.  **Prueba de Estacionariedad (ADF):** Para identificar si las series tienen una tendencia persistente.
                        2.  **Prueba de Cointegración (Johansen):** Para determinar si las series no estacionarias mantienen una relación de equilibrio a largo plazo.
                    """)

                with st.container(border=True):
                    st.subheader("Lógica de Escenarios")
                    st.markdown("""
                    Las proyecciones a 30 años se construyen en dos etapas para asegurar su realismo:

                    1.  **Pronóstico del Modelo (Primeros 5 Años):** Se utiliza el pronóstico directo del modelo VAR/VECM, incluyendo sus intervalos de confianza para definir los escenarios optimista y pesimista.
                    2.  **Convergencia Suave a la Meta:** Después de los primeros 5 años, los escenarios son guiados gradualmente hacia una meta de largo plazo (ej. 3% para Banxico), reflejando la expectativa de que la política monetaria eventualmente anclará la inflación.
                    """)

            with col2:
                with st.container(border=True):
                    st.subheader("Variables Utilizadas:")
                    st. markdown("""
                    * **Inflación (Variable a Proyectar):** Se usa la variación anual del Índice de Precios al Consumidor para todos los Consumidores Urbanos (CPI-U). Es la medida de inflación más utilizada en el país.
                        * *ID de Serie (FRED):* `CPIAUCSL` (se usa el índice para luego calcular la variación anual).

                    * **Tasa de Interés:** Se utiliza la Tasa de Fondos Federales Efectiva (Effective Federal Funds Rate). Es la tasa de interés principal que la Reserva Federal (Fed) utiliza para dirigir su política monetaria.
                        * *ID de Serie (FRED):* `EFFR`

                    * **Tipo de Cambio:** Se emplea el Índice Ponderado del Dólar contra una canasta de monedas de economías extranjeras avanzadas. Mide la fortaleza del dólar a nivel internacional. Se utiliza su logaritmo para estabilizar la varianza.
                        * *ID de Serie (FRED):* `DTWEXAFEGS`

                    **Para hacer uso del modelo es necesario que el usuario cuente con una clave de API. Si el usuario aún no cuenta con una, puede obtenerla en la siguiente dirección: https://fred.stlouisfed.org/docs/api/api_key.html**
                    """)

        with tab_proyeccion:
            st.subheader("Generar Proyección")  
            st.divider()

            # Dividimos la página en una columna principal y una lateral para el formulario
            col_espacio1, col_analisis, col_espacio2, col_formulario = st.columns([0.5, 5.0, 0.5, 1])

            with col_formulario:
                st.subheader("Parámetros")
                with st.form(key="form_usa"):
                    fred_api_key = st.secrets["FRED_API_KEY"]
                    start_date_usa = st.date_input("Fecha de Inicio", pd.to_datetime("2005-01-01"))
                    anos_proyeccion_usa = st.number_input("Años a Proyectar", 5, 50, 30)

                    col_metas1, col_metas2 = st.columns(2)
                    with col_metas1:
                        meta_central = st.number_input("Meta Central (%)", value=2.0, step=0.1)
                        meta_baja = st.number_input("Meta Baja (%)", value=2.0, step=0.1)
                    with col_metas2:
                        meta_alta = st.number_input("Meta Alta (%)", value=3.5, step=0.1)
                        
                    submit_button = st.form_submit_button(label="Generar Proyección")

            with col_analisis:
                if submit_button:
                    if not fred_api_key:
                        st.warning("Por favor, ingresa un Token de FRED válido.")
                    else:
                        with st.spinner("Ejecutando modelo econométrico..."):
                            # Parámetros para los escenarios (puedes moverlos a la barra lateral después)
                            params_escenarios_usa = {
                                'anos_modelo': 5, 'meta_central': meta_central, 'meta_baja': meta_baja, 'meta_alta': meta_alta,
                                'theta_central': 0.030, 'theta_baja': 0.050, 'theta_alta': 0.015
                            }
                            series_ids_usa = {
                                "cpi_index": "CPIAUCSL", "tasa_interes": "EFFR", "tipo_cambio": "DTWEXAFEGS"#"inflacion": "SP30578", "tasa_interes": "SF43783", "tipo_cambio": "SF43718"
                            }
                                
                            # Llamamos a la función del módulo
                            resultados_usa = generar_proyeccion_usa(
                                api_key=fred_api_key,
                                series_ids=series_ids_usa,
                                start_date=start_date_usa.strftime("%Y-%m-%d"),
                                anos_proyeccion=anos_proyeccion_usa,
                                params_escenarios=params_escenarios_usa
                            )

                        if resultados_usa:
                            st.session_state['resultados'] = resultados_usa # Guarda los resultados obtenidos en la pestaña proyeccion para visualizar diagnostico
                            st.success("Proyección generada exitosamente.")
                            
                            # Diccionarios con los resultados
                            promedios = resultados_usa["promedios"]
                            df_historico = resultados_usa["df_historico"]
                            escenario_base = resultados_usa["escenario_base"]
                            escenario_positivo = resultados_usa["escenario_positivo"]
                            escenario_negativo = resultados_usa["escenario_negativo"]
                            
                            # Mostrar KPIs (1/3)
                            with st.container():
                                st.subheader("Promedios Proyectados")
                                col0, col1, col2, col3, col4 = st.columns([3, 3, 3, 3, 3])
                                col1.metric("Escenario Base", f"{promedios['Base']:.2f}%")
                                col2.metric("Escenario Positivo", f"{promedios['Positivo']:.2f}%")
                                col3.metric("Escenario Negativo", f"{promedios['Negativo']:.2f}%")

                            st.divider()

                            # --- Creación de la Gráfica Interactiva con Plotly ---
                            st.subheader("Gráfica de Proyección")

                            df_historico_serie = resultados_usa["df_historico"]['inflacion']
                            ultimo_punto_historico = df_historico_serie.iloc[-1:]
                            base_para_graficar = pd.concat([ultimo_punto_historico, resultados_usa['escenario_base']])
                            positivo_para_graficar = pd.concat([ultimo_punto_historico, resultados_usa['escenario_positivo']])
                            negativo_para_graficar = pd.concat([ultimo_punto_historico, resultados_usa['escenario_negativo']])

                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=df_historico_serie.index, y=df_historico_serie, mode='lines', name='Histórico', line=dict(color='#BBBBBB', width=3)))
                            fig.add_trace(go.Scatter(x=base_para_graficar.index, y=base_para_graficar, mode='lines', name=f"Base (Prom: {promedios['Base']:.2f}%)", line=dict(color='#003366', width=4)))
                            fig.add_trace(go.Scatter(x=positivo_para_graficar.index, y=positivo_para_graficar, mode='lines', name=f"Positivo (Prom: {promedios['Positivo']:.2f}%)", line=dict(color='#6699CC', dash='dash')))
                            fig.add_trace(go.Scatter(x=negativo_para_graficar.index, y=negativo_para_graficar, mode='lines', name=f"Negativo (Prom: {promedios['Negativo']:.2f}%)", line=dict(color='#666666', dash='dash')))
                            
                            fig.add_hline(y=2.0, line_dash="dot", line_color="black", annotation_text="Meta FRED (2%)", annotation_position="bottom right")

                            # Configurar el diseño de la gráfica
                            fig.update_layout(
                                title_text=f"Proyección de inflación en Estados Unidos a {anos_proyeccion_usa} años",
                                xaxis_title="Fecha",
                                yaxis_title="Inflación Anualizada (%)",
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                                template="plotly_white",
                                font=dict(
                                    family="Arial, sans-serif",
                                    size=12,
                                    color="black"
                                ),
                                height=600,
                                xaxis=dict(gridcolor='#EAEAEA'), # Color de la cuadrícula
                                yaxis=dict(gridcolor='#EAEAEA')
                            )
                            
                            # Mostrar la gráfica de Plotly (2/3)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            st.divider()

                            # Tabla comparativa de escenarios (3/3)

                            with st.container():
                                st.subheader("Comparativa de Escenarios")

                                col0, col1, col2 = st.columns([1, 2, 1])

                                with col1:      
                                    st.dataframe(resultados_usa['tabla_escenarios'], use_container_width=True)   

                        else:
                            st.error("Ocurrió un error al generar la proyección.")
                else:
                    st.info("Ingresa los parámetros en el formulario y haz clic en 'Generar Proyección' para ver los resultados.")


    # --- Contenido de la Pestaña de Diagnósticos ---
        with tab_diagnosticos:
            st.subheader("Diagnósticos del Modelo")
            st.divider() # Crea una linea divisoria

            # Verificamos si ya se generó una proyección QUIZAS TENER CUIDADO CON LA PALABRA RESULTADOS, puede ser resultados_mex revisar codigo gemini con eso
            if 'resultados' in st.session_state:
                resultados_usa = st.session_state['resultados']

                with st.container():

                    # Pruebas hechas (1/3)
                    st.markdown("#### Pruebas hechas para decidir entre VAR/VECM")
                    st.info(f"El modelo seleccionado automáticamente para esta proyección fue un **{resultados_usa['modelo_usado']}**.")

                    col_espacio1, col_prueba1, col_espacio2, col_prueba2, col_espacio3 = st.columns([2, 3, 2, 3, 2])

                    with col_prueba1:
                        st.markdown("**Prueba de Estacionariedad (Dickey-Fuller)**")

                        nombres_mapa = {
                                    'inflacion': 'inflación',
                                    'tasa_interes': 'tasa de interés',
                                    'tipo_cambio': 'tipo de cambio'
                        }

                        variables_no_est = resultados_usa['series_no_estacionarias']

                        if variables_no_est:
                            nombres_legibles = [nombres_mapa.get(var, var) for var in variables_no_est]
                            lista_series = ", ".join(nombres_legibles)
                        else:
                            lista_series = "Ninguna"

                        st.markdown(f"Resultado de la prueba de estacionariedad se concluyo que las siguientes variables no son estacionarias: **{lista_series}**.")

                    with col_prueba2:
                        st.markdown("**Prueba de Cointegración (Johansen)**")
                        # Mostramos el número de relaciones de cointegración encontradas
                        num_rel = resultados_usa['relaciones_coint']
                        st.markdown(f"Resultado de la prueba de cointegración se concluyo que el siguiente número de variables tienen relación: **{num_rel}**")
                        if num_rel > 0:
                            st.markdown("_Esto justifica el uso de un modelo VECM._")
                        else:
                            st.markdown("_Al no encontrar cointegración en los datos, se decidío utilizar un modelo VAR aplicando diferencias._")

                st.divider()

                with st.container():

                    # Sección de Análisis de Residuos (2/3)
                    st.subheader("Análisis de Residuos")
                    st.write("Un buen modelo debe tener errores (residuos) que se comporten como ruido aleatorio, sin patrones.")
                
                    # Usamos columnas para mostrar las gráficas lado a lado
                    col_espacio1, col_acf, col_espacio2, col_hist, col_espacio3 = st.columns([1, 4, 1, 4, 1])

                    with col_acf:
                        st.markdown("**Autocorrelación (ACF)**")

                        # Calcular ACF y los intervalos de confianza
                        acf_values, confint = acf(resultados_usa['residuos'], nlags=24, alpha=0.05)
                        
                        fig_acf = go.Figure()
                        
                        # Banda de confianza (sombreado azul)
                        conf_upper = confint[1:, 1] - acf_values[1:]
                        conf_lower = confint[1:, 0] - acf_values[1:]
                        x_axis = np.arange(1, 25)
                        fig_acf.add_trace(go.Scatter(x=np.concatenate([x_axis-1, 1+x_axis[::-1]]), y=np.concatenate([conf_upper, conf_lower[::-1]]), fill='toself', fillcolor='rgba(173, 216, 230, 0.5)', line=dict(color='rgba(255,255,255,0)'), showlegend=False))
                        fig_acf.add_trace(go.Bar(x=x_axis, y=acf_values[1:], name='ACF', width=0.2)) # 'width' hace las barras más finas          
                        fig_acf.update_layout(template="plotly_white", height=400, title_text="Autocorrelación de Residuos")
                        st.plotly_chart(fig_acf, use_container_width=True)

                        st.markdown("**Interpretación:** Para que el modelo sea válido, la mayoría de las barras deben estar **dentro del área sombreada**.")

                    with col_hist:
                        st.markdown("**Histograma**")
                        fig_hist = go.Figure()
                        fig_hist.add_trace(go.Histogram(x=resultados_usa['residuos'], nbinsx=25, xbins=dict(size=0.1), name='Frecuencia',marker=dict(color='#6699CC',line=dict(color='#003366', width=1) )))
                        fig_hist.update_layout(template="plotly_white", height=400, title_text="Histograma de Residuos", bargap=0.05)
                        st.plotly_chart(fig_hist, use_container_width=True)

                        st.markdown("**Interpretación:** La distribución de los errores debe parecerse a una **campana (distribución normal)**. Esto sugiere que los errores del modelo son aleatorios y no están sesgados.")
            
                st.divider()

                # Resumen estadístico (3/3)
                with st.expander("Ver Resumen Estadístico Completo"):
                    st.text(resultados_usa['resumen_texto'])

            else:
                st.warning("Debes generar una proyección en la pestaña '📈 Proyección' para ver los diagnósticos del modelo.")

    # --- Contenido de la Pestaña de Descarga ---
        with tab_descarga:
            st.subheader("Descargar Datos de la Proyección")
            st.divider()
            
            # 1. Verificar si los resultados existen en la memoria de la sesión
            if 'resultados' in st.session_state:
                resultados_usa = st.session_state['resultados']
                
                # 2. Preparar un DataFrame uniendo los tres escenarios proyectados
                df_para_descarga = pd.DataFrame({
                    'Base': resultados_usa['escenario_base'],
                    'Positivo': resultados_usa['escenario_positivo'],
                    'Negativo': resultados_usa['escenario_negativo']
                })
                
                # Formatear los números a 2 decimales para el CSV
                df_para_descarga = df_para_descarga.round(2)
                
                # 3. Convertir el DataFrame a formato CSV en memoria
                #    .to_csv() lo convierte a texto
                #    .encode('utf-8') lo convierte a bytes, que es lo que el botón necesita
                csv_data = df_para_descarga.to_csv(index=True, encoding='utf-8')
                
                # 4. Crear el botón de descarga
                st.download_button(
                label="Descargar Proyección (CSV)", # Texto del botón
                data=csv_data,                      # Los datos a descargar
                file_name='proyeccion_inflacion_mexico.csv', # Nombre del archivo
                mime='text/csv',                      # Tipo de archivo
                )
            else:
                # Mensaje si aún no se ha corrido el análisis
                st.warning("Debes generar una proyección en la pestaña '📈 Proyección' para poder descargar los datos.")












    # --- Contenido de la Pestaña del S&P 500 ---
    if analisis_seleccionado == "S&P 500":
        st.header("Proyección de Rendimiento del S&P 500")
        st.info("La funcionalidad para este análisis se añadirá aquí.")
        # Aquí iría el formulario y la lógica para el análisis del S&P 500.