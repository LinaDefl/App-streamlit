import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import folium
from streamlit_folium import st_folium
import joblib
import streamlit as st
from folium.plugins import HeatMap
# Configuración de página
st.set_page_config(page_title="Predicción Aeropuerto", layout="wide")

# Usando una URL directa del GIF (ej. de Giphy)

# ==================== CONFIGURACIÓN VISUAL (AERO TECH) ====================
st.markdown("""
<div style="
    width:100%;
    height:160px;
    overflow:hidden;
    margin-top:10px;
    margin-bottom:20px;
">
    <img src="https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExanl1eWpyMjJkcGgyZDB0MHI1M3Eyb3E0MDd0cG5rYmRzM3d2MTFsbSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/l3q2Ev2aYYSIf7DYA/giphy.gif"
         style="
         width:100%;
         object-fit:cover;
         opacity:0.25;
         ">
</div>
""", unsafe_allow_html=True)

# Paleta de colores definida en variables para usar en Python
COLORES = {
    "azul_corp": "#0078D7",    # Check-in / Principal
    "verde_tech": "#00C49F",   # Pasaporte / Positivo
    "naranja_alert": "#FF8042",# Seguridad / Alerta
    "gris_texto": "#0078D7",   # Texto principal
    "fondo_suave": "#F0F2F6",  # Fondos secundarios
    "blanco": "#darkblue"
}
# CSS Personalizado
st.markdown(f"""
    <style>
    /* Fondo general limpio */
    .stApp {{
        background-color: {COLORES['blanco']};
    }}
     
    /* Títulos */
    h1, h2, h3 {{
        color: {COLORES['gris_texto']} !important;
        font-family: 'Segoe UI', sans-serif;
    }}
    
    /* Estilo de Tarjetas para los KPIs (Metrics) */
    div[data-testid="metric-container"] {{
        background-color: {COLORES['fondo_suave']};
        border-left: 5px solid {COLORES['azul_corp']};
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        colo: {COLORES['gris_texto']}
    }}
    
    div[data-testid="metric-container"] label {{
        color: #555 !important; /* Etiqueta gris medio */
        font-size: 14px;
    }}
    
    div[data-testid="metric-container"] [data-testid="stMetricValue"] {{
        color: {COLORES['azul_corp']} !important;
        font-size: 26px !important;
        font-weight: 700;
    }}
    
    /* Pestañas (Tabs) */
    .stTabs [data-baseweb="tab-list"] {{
        background-color: transparent;
        border-bottom: 2px solid #ddd;
    }}
    .stTabs [data-baseweb="tab"] {{
        color: #777 !important;
    }}
    .stTabs [data-baseweb="tab"][aria-selected="true"] {{
        color: {COLORES['azul_corp']} !important;
        border-top: 3px solid {COLORES['azul_corp']} !important;
        background-color: rgba(0, 120, 215, 0.05);
    }}
    </style>
    """, unsafe_allow_html=True)


st.markdown("<h1 style='text-align: center; color: #0078D7;'>✈️ Predicción de Afluencia en el Aeropuerto de BCN- T2 </h1>", unsafe_allow_html=True)



# ==================== CARGA DE MODELOS ====================
@st.cache_resource
def cargar_modelos():
    """Carga los 3 modelos guardados"""
    try:
        modelo_checkin = joblib.load('forecasterDIC.joblib') 
        modelo_seguridad = joblib.load('forecasterSEC_DIC.joblib')
        modelo_pasaporte = joblib.load('forecasterPAS_DIC.joblib')
        return modelo_checkin, modelo_seguridad, modelo_pasaporte
    except FileNotFoundError:
        st.warning("⚠️ Modelos no encontrados. Modo demo activado.")
        return None, None, None

modelo_checkin, modelo_seguridad, modelo_pasaporte = cargar_modelos()

# ==================== CARGA DE DATOS ====================
@st.cache_data
def cargar_datos():
    """Carga datos históricos"""
    try:
        df = pd.read_csv('aflu_correct_modelar2.csv', parse_dates=['datetime'])
        
        # Crear columna de total de pasajeros
        if 'tot_pax' not in df.columns:
            df['tot_pax'] = df['real_pax_seguridad']
        
        # Crear momento del día
        df['hora'] = pd.to_datetime(df['datetime']).dt.hour
        def clasificar_momento(hora):
            if 4 <= hora <= 11:
                return 'Mañana'
            elif 12 <= hora <= 19:
                return 'Tarde'
            else:
                return 'Noche'
        df['momento_dia'] = df['hora'].apply(clasificar_momento)
        
        # Tipo de día
        if 'tipo_dia' not in df.columns:
            df['dia_semana'] = pd.to_datetime(df['datetime']).dt.dayofweek
            df['tipo_dia'] = df['dia_semana'].apply(lambda x: 'Fin de semana' if x >= 5 else 'Laborable')
        
        # Temporada
        if 'temporada' not in df.columns:
            mes = pd.to_datetime(df['datetime']).dt.month
            df['temporada'] = mes.apply(lambda m: 'Alta' if m in [6,7,8,12] else 'Media' if m in [4,5,9,10] else 'Baja')
        
        return df
        
    except Exception as e:
        st.error(f"Error cargando datos: {e}")
        return None

df_historico = cargar_datos()

if df_historico is None:
    st.stop()
# ======================= CARGAR TABLA MODELOS 
@st.cache_data
def cargar_tablas_modelo():
    df_ci = pd.read_csv("Tabla_modelo_checkin.csv", parse_dates=["datetime"])
    df_sec = pd.read_csv("Tabla_modelo_seguridad.csv", parse_dates=["datetime"])
    df_pp = pd.read_csv("Tabla_modelo_pasaporte.csv", parse_dates=["datetime"])

    for df in [df_ci, df_sec, df_pp]:
        df.sort_values("datetime", inplace=True)
        df.set_index("datetime", inplace=True)

    return df_ci, df_sec, df_pp


df_modelo_ci, df_modelo_sec, df_modelo_pp = cargar_tablas_modelo()

# ==================== FUNCIONES AUXILIARES ====================
# ==================== CARGAR FESTIVIDADES Y CLIMA ====================
@st.cache_data
def cargar_festividades():
    """Carga el CSV de festividades de Barcelona 2025"""
    try:
        df_fest = pd.read_csv("festividades_bcn.csv", parse_dates=['Date_holiday'])
        # Convertir a set de fechas para búsqueda rápida
        festivos = set(df_fest['Date_holiday'].dt.date)
        return festivos
    except Exception as e:
        st.warning(f"⚠️ No se pudo cargar festividades: {e}. Usando lista por defecto.")
        # Fallback: festivos principales de Catalunya 2025
        return {
            datetime(2025, 1, 1).date(),   # Año Nuevo
            datetime(2025, 1, 6).date(),   # Reyes
            datetime(2025, 4, 18).date(),  # Viernes Santo
            datetime(2025, 4, 21).date(),  # Lunes de Pascua
            datetime(2025, 5, 1).date(),   # Día del Trabajo
            datetime(2025, 6, 24).date(),  # Sant Joan
            datetime(2025, 8, 15).date(),  # Asunción
            datetime(2025, 9, 11).date(),  # Diada
            datetime(2025, 9, 24).date(),  # La Mercè
            datetime(2025, 10, 12).date(), # Hispanidad
            datetime(2025, 11, 1).date(),  # Todos los Santos
            datetime(2025, 12, 6).date(),  # Constitución
            datetime(2025, 12, 8).date(),  # Inmaculada
            datetime(2025, 12, 25).date(), # Navidad
            datetime(2025, 12, 26).date()  # San Esteban
        }

@st.cache_data
def cargar_clima():
    """Carga el CSV de clima actualizado"""
    try:
        df_clima = pd.read_csv("clima.csv", parse_dates=['fecha_scrapeo'])
        df_clima.set_index('fecha_scrapeo', inplace=True)
        return df_clima
    except Exception as e:
        st.warning(f"⚠️ No se pudo cargar clima: {e}. Usando valores por defecto.")
        return None

# Cargar datos
festivos_bcn = cargar_festividades()
df_clima = cargar_clima()

# ==================== FUNCIÓN PARA CREAR EXÓGENAS BASE ====================
# ==================== FUNCIONES PARA EXÓGENAS ====================

def crear_exogenas_base(fechas, festivos_set, df_clima):
    """
    Crea las variables exógenas básicas comunes a todos los escenarios
    """
    exog = pd.DataFrame(index=fechas)
    
    h = exog.index.hour
    weekday = exog.index.weekday
    
    # Variables temporales
    exog["h"] = h
    exog["weekday_num"] = weekday
    exog["is_weekend"] = pd.Series(weekday).isin([5, 6]).astype(int).values
    exog["is_weekday"] = (exog["is_weekend"] == 0).astype(int)
    
    # Momentos del día
    exog["is_morning"] = ((h >= 4) & (h <= 11)).astype(int)
    exog["is_evening"] = ((h >= 12) & (h <= 18)).astype(int)
    exog["is_night"] = ((h >= 19) | (h <= 3)).astype(int)
    
    # Festividades
    exog["festivo"] = [1 if d.date() in festivos_set else 0 for d in exog.index]
    exog["Visp_fest"] = [
        1 if (d + timedelta(days=1)).date() in festivos_set else 0
        for d in exog.index
    ]
    
    # Clima (desde CSV si existe)
    if df_clima is not None:
        exog["temperatura"] = np.nan
        exog["v_viento"] = np.nan
        
        for fecha in exog.index:
            if fecha in df_clima.index:
                exog.loc[fecha, "temperatura"] = df_clima.loc[fecha, "temperatura"]
                exog.loc[fecha, "v_viento"] = df_clima.loc[fecha, "v_viento"]
        
        # Rellenar faltantes
        exog["temperatura"] = exog["temperatura"].fillna(method='ffill').fillna(15.0)
        exog["v_viento"] = exog["v_viento"].fillna(method='ffill').fillna(10.0)
    else:
        exog["temperatura"] = 15.0
        exog["v_viento"] = 10.0
    
    return exog


def crear_exogenas_prediccion_real(df_modelo, cutoff, steps, festivos_set, df_clima):
    """
    Crea exógenas para PREDICCIÓN REAL (con continuidad de momentum/lags)
    """
    fechas = pd.date_range(
        start=cutoff + timedelta(hours=1),
        periods=steps,
        freq="h"
    )
    
    # Empezar con las variables base
    exog = crear_exogenas_base(fechas, festivos_set, df_clima)
    
    # Aeropuerto: usar patrón histórico real
    if "no_vuelos" in df_modelo.columns:
        patron = df_modelo.groupby(df_modelo.index.hour)["no_vuelos"].mean()
        exog["no_vuelos"] = exog["h"].map(patron).fillna(0).round().astype(int)
    else:
        exog["no_vuelos"] = ((exog["h"] >= 0) & (exog["h"] <= 3)).astype(int)
    
    # ✅ MOMENTUM: Basado en la HORA, no heredado ciegamente
    for col in df_modelo.columns:
        if any(k in col for k in ["momentum", "roll"]):
            # Calcular momentum promedio por hora del día desde el histórico
            patron_momentum = df_modelo.groupby(df_modelo.index.hour)[col].mean()
            exog[col] = exog["h"].map(patron_momentum).fillna(0)
        elif "lag" in col:
            # Los lags sí pueden heredarse del último valor (son datos reales pasados)
            exog[col] = df_modelo[col].iloc[-1]
    
    # Otras variables persistentes
    for col in df_modelo.columns:
        if col not in exog.columns:
            exog[col] = df_modelo[col].iloc[-1]
    
    return exog


def crear_exogenas_simulacion(df_modelo, cutoff, steps, festivos_set, df_clima, 
                               hora_sim=None, temp_sim=None, viento_sim=None,
                               es_festivo=False, es_vispera=False, aeropuerto_cerrado=False):
    """
    Crea exógenas para SIMULACIÓN (valores personalizados por el usuario)
    """
    fechas = pd.date_range(
        start=cutoff + timedelta(hours=1),
        periods=steps,
        freq="h"
    )
    
    # Empezar con las variables base
    exog = crear_exogenas_base(fechas, festivos_set, df_clima)
    
    # ✅ SOBRESCRIBIR ÚLTIMA FILA con valores de simulación
    idx = exog.index[-1]
    
    if hora_sim is not None:
        exog.loc[idx, 'h'] = hora_sim
        exog.loc[idx, 'is_morning'] = 1 if 4 <= hora_sim <= 11 else 0
        exog.loc[idx, 'is_evening'] = 1 if 12 <= hora_sim <= 18 else 0
        exog.loc[idx, 'is_night'] = 1 if (hora_sim >= 19 or hora_sim <= 3) else 0
    
    if temp_sim is not None:
        exog.loc[idx, 'temperatura'] = temp_sim
    
    if viento_sim is not None:
        exog.loc[idx, 'v_viento'] = viento_sim
    
    exog.loc[idx, 'festivo'] = 1 if es_festivo else 0
    exog.loc[idx, 'Visp_fest'] = 1 if es_vispera else 0
    
    # Aeropuerto cerrado
    if aeropuerto_cerrado:
        if "no_vuelos" in exog.columns:
            exog.loc[idx, 'no_vuelos'] = 1
        # ⚠️ momentum y rolling = 0
        for col in df_modelo.columns:
            if any(k in col for k in ["lag", "momentum", "roll"]):
                if col in exog.columns:
                    exog.loc[idx, col] = 0
                else:
                    exog[col] = 0
                    exog.loc[idx, col] = 0
    else:
        # Patrón normal de vuelos
        if "no_vuelos" in df_modelo.columns:
            patron = df_modelo.groupby(df_modelo.index.hour)["no_vuelos"].mean()
            exog["no_vuelos"] = exog["h"].map(patron).fillna(0).round().astype(int)
        
        # Heredar momentum del histórico
        for col in df_modelo.columns:
            if any(k in col for k in ["lag", "momentum", "roll"]):
                if col not in exog.columns:
                    exog[col] = df_modelo[col].iloc[-1]
    
    # Otras variables persistentes
    for col in df_modelo.columns:
        if col not in exog.columns:
            exog[col] = df_modelo[col].iloc[-1]
    
    return exog
#=================================================================
def color_por_pax(pax):
    """Determina color según cantidad de pasajeros""" # SEGUN ZONA SERA ALGO DETERMINADO COMO ALTO O NO EN EL CONTROL DE SEGURIDAD HAY MAS PAX NO ES LO MISMO A C.PASAPORTE 
    if pax < 600:
        return "green"
    elif pax < 900:
        return "orange"
    else:
        return "red"

def radio_por_pax(pax):
    """Determina tamaño del círculo según pasajeros"""
    return max(10, min(40, 10 + (pax / 40)))

def nivel_afluencia(zona, pax):
    """
    Determina el nivel de afluencia según la zona y número de pasajeros
    Retorna: (nivel, color)
    """
    # Umbrales por zona (ajusta según tus datos)
    umbrales = {
        'Check-in': {'bajo': 600, 'medio':1000},
       'Seguridad': {'bajo': 750, 'medio': 1100},
       'Pasaporte': {'bajo': 180, 'medio': 300}
    }
    
    umbral = umbrales.get(zona, {'bajo': 400, 'medio': 700})
    
    if pax < umbral['bajo']:
        return 'Baja', 'green'
    elif pax < umbral['medio']:
        return 'Media', 'orange'
    else:
        return 'Alta', 'red'

def radio_por_pax(zona, pax):
    """
    Determina el tamaño del círculo en el mapa según pasajeros
    """
    # Ajustar el radio según la zona (diferentes escalas)
    escalas = {
        'Check-in': {'min': 15, 'max': 40, 'divisor': 30},
        'Seguridad': {'min': 15, 'max': 40, 'divisor': 35},
        'Pasaporte': {'min': 15, 'max': 40, 'divisor': 25}
    }
    
    escala = escalas.get(zona, {'min': 15, 'max': 40, 'divisor': 30})
    
    radio = escala['min'] + (pax / escala['divisor'])
    return max(escala['min'], min(escala['max'], radio))

# ==================== COORDENADAS DE LAS 3 ZONAS ====================
ZONAS = {
    'Check-in': {'lat': 41.30217150841842, 'lon': 2.076006729313788},
    'Seguridad': {'lat': 41.30259511824617, 'lon': 2.075516325702159},
    'Pasaporte': {'lat': 41.30217526799761, 'lon': 2.0743422666825895}
}


# ==================== SIDEBAR: FILTROS GLOBALES ====================
st.sidebar.markdown("## 🎛️ Filtros Temporales")

# Filtro de tipo de día  
tipo_dia_filter = st.sidebar.multiselect(
    "📆 Tipo de día",
    options=['Laborable', 'Fin de semana'],
    default=['Laborable', 'Fin de semana']
)

# Filtro de momento del día
momento_dia_filter = st.sidebar.multiselect(
    "🕐 Momento del día",
    options=['Mañana', 'Tarde', 'Noche'],
    default=['Mañana', 'Tarde', 'Noche']
)
# Rango de fechas
st.sidebar.markdown("### 📊 Rango de datos")
fecha_min = pd.to_datetime(df_historico['datetime']).min().date()
fecha_max = pd.to_datetime(df_historico['datetime']).max().date()

fecha_inicio = st.sidebar.date_input("Desde", fecha_min, min_value=fecha_min, max_value=fecha_max)
fecha_fin = st.sidebar.date_input("Hasta", fecha_max, min_value=fecha_min, max_value=fecha_max)

# Aplicar filtros
df_filtrado = df_historico.copy()

#if temporada_filter != 'Todas':
 #   df_filtrado = #df_filtrado[df_filtrado['temporada'] == #temporada_filter]

df_filtrado = df_filtrado[
    (df_filtrado['tipo_dia'].isin(tipo_dia_filter)) &
    (df_filtrado['momento_dia'].isin(momento_dia_filter)) &
    (pd.to_datetime(df_filtrado['datetime']).dt.date >= fecha_inicio) &
    (pd.to_datetime(df_filtrado['datetime']).dt.date <= fecha_fin)
]


# ==================== KPIs GLOBALES ====================
st.markdown("### 📊 KPIs Generales")
# PRIMERA FILA DE MÉTRICAS
col1, col2, col3 = st.columns(3)

with col1:  # TOTAL PASAJEROS 
    total_pax = int(df_filtrado['tot_pax'].sum())
    st.metric("👥 Total Pasajeros", f"{total_pax:,}")

with col2:  # TOTAL VUELOS
    if 'new_flt_h' in df_filtrado.columns:
        total_vuelos = int(df_filtrado['new_flt_h'].sum())
        st.metric("✈️ Total Vuelos", f"{total_vuelos:,}")
    else:
        st.metric("✈️ Total Vuelos", "N/A")

with col3:  # TASA DE OCUPACIÓN
    if 'tot_pax' in df_filtrado.columns and 'tot_capacity_h' in df_filtrado.columns:
        df_ocup = (
            df_filtrado
            .groupby(pd.to_datetime(df_filtrado['datetime']).dt.date)
            .agg({
                'tot_pax': 'sum',
                'tot_capacity_h': 'sum'
            })
        )
        df_ocup['tasa'] = np.where(
            df_ocup['tot_capacity_h'] > 0,
            df_ocup['tot_pax'] / df_ocup['tot_capacity_h'],
            0
        )
        tasa_ocupacion = int(df_ocup['tasa'].mean() * 100)
        st.metric("📊 Tasa de Ocupación", f"{tasa_ocupacion}%")
    else:
        st.metric("📊 Tasa de Ocupación", "N/A")

# SEGUNDA FILA DE MÉTRICAS
col4, col5, col6 = st.columns(3)

with col4:  # MEDIA CHECK-IN
    media_ci_dia = (
        df_filtrado
        .groupby(pd.to_datetime(df_filtrado['datetime']).dt.date)['real_pax_checkin_adj']
        .sum()
        .mean()
    )
    st.metric("🟦 Media en Check-in/día", f"{int(media_ci_dia):,}")

with col5:  # MEDIA SEGURIDAD
    media_sec_dia = (
        df_filtrado
        .groupby(pd.to_datetime(df_filtrado['datetime']).dt.date)['real_pax_seguridad']
        .sum()
        .mean()
    )
    st.metric("👮🏼 Media en c.Seguridad/día", f"{int(media_sec_dia):,}")

with col6:  # MEDIA PASAPORTE
    media_pp_dia = (
        df_filtrado
        .groupby(pd.to_datetime(df_filtrado['datetime']).dt.date)['real_pax_passport']
        .sum()
        .mean()
    )
    st.metric("🛂 Media en c.pasaportes/día", f"{int(media_pp_dia):,}")
#col1, col2, col3 = st.columns(3)

#with col1:
 #   st.metric("🟦 Media diaria Check-in", #f"{int(media_ci_dia):,}")

#with col2:
#    st.metric("👮🏼 Media diaria Seguridad", #f"{int(media_sec_dia):,}")

#with col3:
#    st.metric("🛂 Media diaria Pasaportes", #f"{int(media_pp_dia):,}")


st.markdown("---")

# ==================== PESTAÑAS ====================
tab_muestra, tab_graficos, tab_pred,  = st.tabs([
    "🕵️‍♂️ Análisis de la muestra", 
    "📊 Análisis Histórico", 
    #"🗺️ Mapa Interactivo" , tab_simulacion,
    "🗺️ Mapa Predicción" 
    #"🔮 Predicciones Futuras"
])
# tab_predicciones
# ==================== TAB 1: MUESTRA ====================
with tab_muestra:
    st.subheader("🔎 Análisis de la muestra de datos recogidos")
    
    col_con1, col_con2= st.columns([2, 1])
    with col_con1:
        Variables = st.selectbox(
            "Visualizar vuelos:",
            ["Por compañías", "Por destino", "Por día de la semana"])
    
    # Cargar datos
    @st.cache_data
    def cargar_vuelos():
        """Carga datos vuelos"""
        try:
            df_v = pd.read_csv("Tabla_total_muestra", parse_dates=['fecha_scrapeo'])
            return df_v
        except Exception as e:
            st.error(f"Error al cargar datos: {e}")
            return None
    
    df_v = cargar_vuelos()
    
    # Usar datos completos directamente
    df_plot = df_v.copy()
    
    # GRÁFICO
    if Variables == "Por compañías":
        # Contar vuelos por compañía
        conteo = df_plot['Name_company'].value_counts().reset_index()
        conteo.columns = ['Compañía', 'Cantidad']
        conteo = conteo.sort_values('Cantidad', ascending=True) 
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=conteo['Cantidad'],
            y=conteo['Compañía'],
            orientation='h', 
            name='Vuelos por compañía',
            marker=dict(color='#3498db')
        ))
        
        fig.update_layout(
            title='Cantidad de vuelos por compañía',
            xaxis_title='Cantidad de vuelos',
            yaxis_title='Compañía',
            showlegend=True
        )

    elif Variables == "Por destino":
        # Contar vuelos por destino
        conteo = df_plot['Name_Destiny'].value_counts().reset_index().head(20)
        conteo.columns = ['Destino', 'Cantidad']
        conteo = conteo.sort_values('Cantidad', ascending=True) 
            
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=conteo['Cantidad'],
            y=conteo['Destino'],
            orientation='h', 
            name='Vuelos por destino',
            marker=dict(color='#3498db')
        ))
            
        fig.update_layout(
            title='Destinos más frecuentes en el aeropuerto de BCN-T2',
            xaxis_title='Cantidad de vuelos',
            yaxis_title='Destino',
            yaxis=dict(
            type='category', 
            tickmode='linear', # Fuerza a mostrar cada etiqueta
            automargin=True    # Evita que los nombres largos se corten
        ),
        height=700, #  showlegend=True
        showlegend=True
    )
        
    elif Variables == "Por día de la semana": # week_day
        # 1. Agrupar primero por fecha exacta para contar vuelos diarios reales
        # Asumiendo que tienes una columna 'Fecha' o similar
        vuelos_por_fecha = df_plot.groupby(['fecha_scrapeo',
                                            'week_day']).size().reset_index(name='Vuelos_Día')
    
        # 2. Ahora sí, agrupar por día de la semana para sacar la MEDIA de esos conteos
        conteo = vuelos_por_fecha.groupby('week_day')['Vuelos_Día'].mean().reset_index()
        conteo.columns = ['Dia_semana', 'Media_vuelos']
        
        # Ordenar días de la semana
        orden_dias = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
        conteo['Dia_semana'] = pd.Categorical(conteo['Dia_semana'], categories=orden_dias,
                                              ordered=True)
        conteo = conteo.sort_values('Dia_semana')
        
        # 3. La media general (la línea roja) será el promedio de esos promedios
        media_general = conteo['Media_vuelos'].mean()
        
        fig = go.Figure()
        
        # Barras con los valores reales
        fig.add_trace(go.Bar(
            x=conteo['Dia_semana'],
            y=conteo['Media_vuelos'],  # Valores reales, no la media
            name='Media vuelos por día',
            marker=dict(color='#3498db'),
            hovertemplate='Día: %{x}<br>Media: %{y:.2f}<extra></extra>'
        ))
        
        # Línea horizontal con la media
        fig.add_trace(go.Scatter(
            x=conteo['Dia_semana'],
            y=[media_general] * len(conteo),
            mode='lines',
            name=f'Media Total ({media_general:.1f})',
            line=dict(color='red', dash='dash', width=2)
        ))
            
        fig.update_layout(
            title='Promedio de vuelos según día de la semana',
            xaxis_title='Día de la semana',
            yaxis_title='Promedio de vuelos',
            showlegend=True
        )
    
    st.plotly_chart(fig, use_container_width=True)
# ==================== TAB 2: GRÁFICOS ====================
with tab_graficos:
    st.subheader("📈 Evolución Histórica del flujo")
    
    col_config1, col_config2, col_config3 = st.columns([2, 1, 1])
    
    with col_config1:
        zona_grafico = st.selectbox(
            "Zona a visualizar:",
            ["Todas las zonas", "Check-in", "Seguridad",
             "Pasaporte", "Por Compañías", "Por Tipo de ruta", "Por capacidad total", "Por estado de avión"]
        )
    
    with col_config2:
        agrupacion = st.radio("Agrupar por:", 
                              ["Horas", "Días", "Semanas"],
                              horizontal=True)
    
   # with col_config3:
    #    mostrar_vuelos = st.checkbox("Mostrar #vuelos activos", value=False)
    
    # Preparar datos según agrupación
    
# ================= PREPARAR DATOS SEGÚN AGRUPACIÓN =================
    
    if agrupacion == "Horas":
        df_plot = df_filtrado.copy()
    
    elif agrupacion == "Días":
        df_plot = (
            df_filtrado       .groupby(pd.to_datetime(df_filtrado['datetime']).dt.date)
            .agg({
                'real_pax_checkin_adj': 'sum',
                'real_pax_seguridad': 'sum',
                'real_pax_passport': 'sum',
                'tot_pax': 'sum',
                'active_flt_2h': 'sum',
                'Group_Ryanair': 'sum',
                'Group_wizz': 'sum',
                'Group_easy': 'sum',
                'other_companies': 'sum',
                'EU_flt': 'sum',
                'INT_flt': 'sum',
                'NAC_flt': 'sum',
                'tot_capacity_h': 'sum',
                'Ontime_flt': 'sum',
                'Delayed_flt': 'sum'
            })
            .reset_index()
        )
        df_plot.rename(columns={'index': 'datetime'}, inplace=True)
    
    elif agrupacion == "Semanas":
        df_temp = df_filtrado.copy()
        df_temp["year_week"] = pd.to_datetime(df_temp["datetime"]).dt.isocalendar().week
    
        df_plot = (
            df_temp
            .groupby("year_week")
            .agg({
                'real_pax_checkin_adj': 'sum',
                'real_pax_seguridad': 'sum',
                'real_pax_passport': 'sum',
                'tot_pax': 'sum',
                'active_flt_2h': 'sum',
                'Group_Ryanair': 'sum',
                'Group_wizz': 'sum',
                'Group_easy': 'sum',
                'other_companies': 'sum',
                'EU_flt': 'sum',
                'INT_flt': 'sum',
                'NAC_flt': 'sum',
                'tot_capacity_h': 'sum',
                'Ontime_flt': 'sum',
                'Delayed_flt': 'sum'
            })
            .reset_index()
        )
    
        df_plot.rename(columns={'year_week': 'datetime'}, inplace=True)

# ================= Crear gráfico con colores vibrantes y buen contraste
    fig = go.Figure()
    
    if zona_grafico == "Todas las zonas":
        fig.add_trace(go.Scatter(
            x=df_plot['datetime'],
            y=df_plot['real_pax_checkin_adj'],
            name='Check-in',
            line=dict(color='#3498db', width=3),
            mode='lines'
        ))
        fig.add_trace(go.Scatter(
            x=df_plot['datetime'],
            y=df_plot['real_pax_seguridad'],
            name='Seguridad',
            line=dict(color='#e74c3c', width=3),
            mode='lines'
        ))
        fig.add_trace(go.Scatter(
            x=df_plot['datetime'],
            y=df_plot['real_pax_passport'],
            name='Pasaporte',
            line=dict(color='#2ecc71', width=3),
            mode='lines'
        ))
    elif zona_grafico == "Por Compañías":
        fig.add_trace(go.Scatter(
            x=df_plot['datetime'],
            y=df_plot['Group_Ryanair'],
            name='Ryanair',
            line=dict(color='#3498db', width=3),
            mode='lines'
        ))
        fig.add_trace(go.Scatter(
            x=df_plot['datetime'],
            y=df_plot['Group_wizz'],
            name='Wizz Air',
            line=dict(color='#9b59b6', width=3),
            mode='lines'
        ))
        fig.add_trace(go.Scatter(
            x=df_plot['datetime'],
            y=df_plot['Group_easy'],
            name='EasyJet',
            line=dict(color='#f39c12', width=3),
            mode='lines'
        ))
        fig.add_trace(go.Scatter(
            x=df_plot['datetime'],
            y=df_plot['other_companies'],
            name='Otras',
            line=dict(color='#1abc9c', width=3),
            mode='lines'
        ))
    elif zona_grafico == "Por Tipo de ruta" :
        fig.add_trace(go.Scatter(
            x=df_plot['datetime'],
            y=df_plot['EU_flt'],
            name='Europa',
            line=dict(color='#3498db', width=3),
            mode='lines'
        ))
        fig.add_trace(go.Scatter(
            x=df_plot['datetime'],
            y=df_plot['INT_flt'],
            name='Internacional',
            line=dict(color='#9b59b6', width=3),
            mode='lines'
        ))
        fig.add_trace(go.Scatter(
            x=df_plot['datetime'],
            y=df_plot['NAC_flt'],
            name='Nacional',
            line=dict(color='#f39c12', width=3),
            mode='lines'
        ))
    elif zona_grafico == "Por capacidad total" :
        fig.add_trace(go.Scatter(
            x=df_plot['datetime'],
            y=df_plot['tot_capacity_h'],
            name='Capacidad total de los vuelos activos',
            line=dict(color='#3498db', width=3),
            mode='lines'
        ))
        fig.add_trace(go.Scatter(
            x=df_plot['datetime'],
            y=df_plot['tot_pax'],
            name='Pasajeros totales estimados',
            line=dict(color='#9b59b6', width=3),
            mode='lines'
        ))
    elif zona_grafico == "Por estado de avión" :
        fig.add_trace(go.Scatter(
            x=df_plot['datetime'],
            y=df_plot['Ontime_flt'],
            name='Vuelos On-Time',
            line=dict(color='#3498db', width=3),
            mode='lines'
        ))
        fig.add_trace(go.Scatter(
            x=df_plot['datetime'],
            y=df_plot['Delayed_flt'],
            name='Vuelos Delayed',
            line=dict(color='#9b59b6', width=3),
            mode='lines'
        ))
    else:
        col_map = {
            'Check-in': 'real_pax_checkin_adj',
            'Seguridad': 'real_pax_seguridad',
            'Pasaporte': 'real_pax_passport'
        }
        color_map = {
            'Check-in': '#3498db',
            'Seguridad': '#e74c3c',
            'Pasaporte': '#2ecc71'
        }
        fig.add_trace(go.Scatter(
            x=df_plot['datetime'],
            y=df_plot[col_map[zona_grafico]],
            name=zona_grafico,
            line=dict(color=color_map[zona_grafico], width=3),
            fill='tozeroy',
            fillcolor=f'rgba({int(color_map[zona_grafico][1:3], 16)}, {int(color_map[zona_grafico][3:5], 16)}, {int(color_map[zona_grafico][5:7], 16)}, 0.3)'
        ))
    
    #if mostrar_vuelos and 'active_flt_2h' in #df_plot.columns:
 #       fig.add_trace(go.Scatter(
  #          x=df_plot['datetime'],
   #         y=df_plot['active_flt_2h'],
    #        name='Vuelos activos',
     #       line=dict(color='#f1c40f', width=2.5, #dash='dash'),
 #           yaxis='y2'
  #      ))
    
    fig.update_layout(
        xaxis_title = (
            "Fecha y Hora" if agrupacion == "Horas"
            else "Fecha" if agrupacion == "Días"
            else "Semana"),
        yaxis_title="Número de Pasajeros",
        yaxis2=dict(
            title=dict(text="Número de Vuelos",
                       font=dict(color='#f1c40f')),
            overlaying='y',
            side='right',
            showgrid=False,
            tickfont=dict(color='#f1c40f')
        ),
        yaxis=dict(
            gridcolor='rgba(189, 195, 199, 0.2)',
            color='#ecf0f1'
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            bgcolor='rgba(52, 73, 94, 0.8)',
            bordercolor='#3498db',
            borderwidth=2,
            font=dict(color='#ecf0f1', size=11)
        ),
        margin=dict(t=80)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Comparativas
    st.markdown("### 📊 Comparativas de Afluencia")
    col1, col2 = st.columns(2)
    
    with col1:
        # Por momento del día
        df_momento = df_filtrado.groupby('momento_dia')['tot_pax'].mean().reset_index()
        df_momento['momento_dia'] = pd.Categorical(
            df_momento['momento_dia'], 
            categories=['Mañana', 'Tarde', 'Noche'], 
            ordered=True
        )
        df_momento = df_momento.sort_values('momento_dia')
        
        fig_momento = go.Figure(data=[
            go.Bar(
                x=df_momento['momento_dia'], 
                y=df_momento['tot_pax'],
                marker_color=['#f39c12', '#e74c3c', '#2c3e50'],
                text=df_momento['tot_pax'].round(0),
                textposition='outside',
                texttemplate='%{text:.0f}',
                textfont=dict(color='#ecf0f1')
            )
        ])
        fig_momento.update_layout(
            title="Promedio por Momento del Día",
            xaxis_title="Momento del Día",
            yaxis_title="Pasajeros Promedio",
            plot_bgcolor='rgba(236, 240, 241, 0.05)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#ecf0f1'),
            xaxis=dict(gridcolor='rgba(189, 195, 199, 0.2)', color='#ecf0f1'),
            yaxis=dict(gridcolor='rgba(189, 195, 199, 0.2)', color='#ecf0f1'),
            height=400
        )
        st.plotly_chart(fig_momento, use_container_width=True)
    
    with col2:
        # Por tipo de día
        df_tipo = df_filtrado.groupby('tipo_dia')['tot_pax'].mean().reset_index()
        fig_tipo = go.Figure(data=[
            go.Bar(
                x=df_tipo['tipo_dia'], 
                y=df_tipo['tot_pax'],
                marker_color=['#3498db', '#9b59b6'],
                text=df_tipo['tot_pax'].round(0),
                textposition='outside',
                texttemplate='%{text:.0f}',
                textfont=dict(color='#ecf0f1')
            )
        ])
        fig_tipo.update_layout(
            title="Promedio por Tipo de Día",
            xaxis_title="Tipo de Día",
            yaxis_title="Pasajeros Promedio",
            plot_bgcolor='rgba(236, 240, 241, 0.05)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#ecf0f1'),
            xaxis=dict(gridcolor='rgba(189, 195, 199, 0.2)', color='#ecf0f1'),
            yaxis=dict(gridcolor='rgba(189, 195, 199, 0.2)', color='#ecf0f1'),
            height=400
        )
        st.plotly_chart(fig_tipo, use_container_width=True)
        
# ==================== TAB MAPA: predicc ====================
#modelo.predict(fh=[datetime])
def obtener_datos_historicos_hora(df, fecha, hora):
    """
    Devuelve los pasajeros reales por zona para una fecha y hora concretas
    """
    fecha_hora = datetime.combine(fecha, datetime.min.time()) + timedelta(hours=hora)

    df_h = df[df["datetime"] ==fecha_hora]
    if df_h.empty:
        return None
    
    return {
        "Check-in":
    int(df_h["real_pax_checkin_adj"].iloc[0]),
        "Seguridad": int(df_h["real_pax_seguridad"].iloc[0]),
        "Pasaporte": int(df_h["real_pax_passport"].iloc[0]),}


# ==================== TAB MAPA ====================
#---------------------------
#st.write("Lags modelo CI:",modelo_checkin.lags)
#st.write("Exógenas esperadas CI:",modelo_checkin.exog_names)
# ==================== TAB SIMULACIÓN: JUGAR CON VARIABLES ====================
# ==================== TAB SIMULACIÓN ====================
     
# ==================== TAB MAPA: PREDICCIONES ====================
# ==================== TAB MAPA: PREDICCIÓN REAL ====================
with tab_pred:
    st.subheader("🗺️ Predicción de Afluencia por Zonas")
    st.markdown("**Predicción basada en la continuidad natural del aeropuerto**")

    # ==================== OPCIONES AVANZADAS ====================
    with st.expander("⚙️ Configuración avanzada del modelo"):
        usar_momentum = st.checkbox(
            "Usar variables de momentum/rolling", 
            value=True,
            help="Desmarca para ignorar momentum y rolling means"
        )
    
    # ... resto del código ...
    
    # Luego en la creación de exógenas:
    if not usar_momentum:
        # Poner momentum y rolling a 0
        for col in exog_ci.columns:
            if any(k in col for k in ["momentum", "roll"]):
                exog_ci[col] = 0
        for col in exog_sec.columns:
            if any(k in col for k in ["momentum", "roll"]):
                exog_sec[col] = 0
        for col in exog_pp.columns:
            if any(k in col for k in ["momentum", "roll"]):
                exog_pp[col] = 0

    
    MAX_STEPS_ML = 38
    cutoff = min(
        df_modelo_ci.index.max(),
        df_modelo_sec.index.max(),
        df_modelo_pp.index.max()
    )
    
    # ==================== CONTROLES ====================
    col1, col2 = st.columns(2)
    
    with col1:
        steps_pred = st.slider("📊 Horas a predecir", 1, MAX_STEPS_ML, 24)
    
    with col2:
        alerta_act = st.checkbox("🚨 Activar alertas automáticas", value=True)
    
    fecha_objetivo = cutoff + timedelta(hours=steps_pred)
    st.info(f"📅 Prediciendo para: **{fecha_objetivo.strftime('%d/%m/%Y %H:%M')}** ({steps_pred} horas adelante)")
    
    # ==================== VERIFICAR MODELOS ====================
    if modelo_checkin is None or modelo_seguridad is None or modelo_pasaporte is None:
        st.error("❌ Modelos no cargados.")
        st.stop()
    
    # ==================== HACER PREDICCIÓN REAL ====================
    try:
        # Listas de exógenas necesarias
        EXOGENAS_CI = ['pax_momentum', 'pax_roll_mean_6h', 'Visp_fest',
                       'temperatura', 'v_viento', 'weekday_num','no_vuelos',
                       'is_morning', 'is_evening', 'is_night', 'festivo']
        
        EXOGENAS_SEC = ['festivo', 'no_vuelos', 'pax_momentum_sec', 'Visp_fest',
                        'temperatura', 'v_viento', 'weekday_num',
                        'is_morning', 'is_evening', 'is_night']
        
        EXOGENAS_PP = ['pax_momentum_pas', 'festivo', 'no_vuelos', 'Visp_fest',
                       'temperatura', 'v_viento', 'weekday_num',
                       'is_morning', 'is_evening', 'is_night']
        
        # ✅ Usar predicción REAL (con continuidad)
        exog_ci_full = crear_exogenas_prediccion_real(df_modelo_ci, cutoff, steps_pred, festivos_bcn, df_clima)
        exog_sec_full = crear_exogenas_prediccion_real(df_modelo_sec, cutoff, steps_pred, festivos_bcn, df_clima)
        exog_pp_full = crear_exogenas_prediccion_real(df_modelo_pp, cutoff, steps_pred, festivos_bcn, df_clima)
        
        # Seleccionar columnas
        exog_ci = exog_ci_full[EXOGENAS_CI]
        exog_sec = exog_sec_full[EXOGENAS_SEC]
        exog_pp = exog_pp_full[EXOGENAS_PP]

        # ==================== REGLA OPERATIVA: AEROPUERTO CERRADO ====================
        
        # Predicciones
        pred_checkin = modelo_checkin.predict(steps=steps_pred, exog=exog_ci)
        pred_seguridad = modelo_seguridad.predict(steps=steps_pred, exog=exog_sec)
        pred_pasaporte = modelo_pasaporte.predict(steps=steps_pred, exog=exog_pp)

        predicciones = {
            "Check-in": int(pred_checkin.iloc[-1]),
            "Seguridad": int(pred_seguridad.iloc[-1]),
            "Pasaporte": int(pred_pasaporte.iloc[-1])
        }
        
        # DEBUG
        with st.expander("🔍 Ver detalles de predicción"):
            st.write(f"**Fecha cutoff:** {cutoff}")
            st.write(f"**Festivo:** {'Sí' if fecha_objetivo.date() in festivos_bcn else 'No'}")
            st.write(f"**Hora:** {fecha_objetivo.hour}h")
            st.write(f"**Temperatura:** {exog_ci['temperatura'].iloc[-1]:.1f}°C")
            st.write(f"**Aeropuerto:** {'Cerrado' if exog_sec['no_vuelos'].iloc[-1] == 1 else 'Abierto'}")
            #if 'pax_momentum' in exog_ci.columns:
             #   st.write(f"**Momentum CI:** {exog_ci['pax_momentum'].iloc[-1]:.2f}")
        
    except Exception as e:
        st.error(f"❌ Error: {e}")
        import traceback
        st.code(traceback.format_exc())
        st.stop()
    if exog_ci["no_vuelos"].iloc[-1] == 1:
        predicciones["Check-in"] = 0
    # ==================== KPIs ====================
    col1, col2, col3 = st.columns(3)
    col1.metric("🟦 Check-in", f"{predicciones['Check-in']:,}")
    col2.metric("🟠 Seguridad", f"{predicciones['Seguridad']:,}")
    col3.metric("🟢 Pasaporte", f"{predicciones['Pasaporte']:,}")
   # col4.metric("👥 Total", f"{sum(predicciones.values()):,}")
    
    # ==================== MAPA ====================
    mapa = folium.Map(
        location=[41.30324, 2.07700],
        zoom_start=17,
        tiles="https://{s}.tile.openstreetmap.fr/osmfr/{z}/{x}/{y}.png",
        attr="OSM France"
    )
    
    for zona, coords in ZONAS.items():
        pax = predicciones[zona]
        nivel, color = nivel_afluencia(zona, pax)
        
        folium.CircleMarker(
            location=[coords["lat"], coords["lon"]],
            radius=radio_por_pax(zona, pax),
            fill=True,
            fill_color=color,
            fill_opacity=0.9,
            color=None,
            popup=f"<b>{zona}</b><br>Pax: {pax:,}<br>Nivel: {nivel}",
            tooltip=f"Zona: {zona}"
        ).add_to(mapa)
    
    st_folium(mapa, width=1200, height=500)

    # ==================== ALERTAS ====================
    if alerta_act:
        for zona, pax in predicciones.items():
            nivel, _ = nivel_afluencia(zona, pax)
            if nivel == "Alta":
                st.error(f"🚨 ALTA AFLUENCIA en {zona}: {pax:,} pasajeros")
    
    # ==================== GRÁFICO ====================
    st.markdown("### 📈 Evolución de la Predicción")
    
    fechas_graf = [cutoff + timedelta(hours=i) for i in range(1, steps_pred + 1)]
    df_pred = pd.DataFrame({
        'Fecha': fechas_graf,
        'Check-in': pred_checkin.values,
        'Seguridad': pred_seguridad.values,
        'Pasaporte': pred_pasaporte.values
    })
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_pred['Fecha'], y=df_pred['Check-in'], name='Check-in', 
                             line=dict(color='#3498db', width=3), mode='lines+markers'))
    fig.add_trace(go.Scatter(x=df_pred['Fecha'], y=df_pred['Seguridad'], name='Seguridad', 
                             line=dict(color='#e74c3c', width=3), mode='lines+markers'))
    fig.add_trace(go.Scatter(x=df_pred['Fecha'], y=df_pred['Pasaporte'], name='Pasaporte', 
                             line=dict(color='#2ecc71', width=3), mode='lines+markers'))
    
    fig.update_layout(xaxis_title="Fecha", yaxis_title="Pasajeros", height=500, hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("📋 Ver tabla detallada"):
        st.dataframe(df_pred.style.format({
            'Check-in': '{:,.0f}',
            'Seguridad': '{:,.0f}',
            'Pasaporte': '{:,.0f}'
        }))
# ==================== COMPARACIÓN CON HISTÓRICO ====================
    st.markdown("### 📈 Comparación con Histórico")
    
    # Definir nombres de días
    dias_semana = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
    
    # Buscar datos históricos de la misma hora/día de semana
    hora_obj = fecha_objetivo.hour
    weekday_obj = fecha_objetivo.weekday()
    
    # Filtrar cada tabla modelo por hora y día de semana
    df_ci_filtrado = df_modelo_ci[
        (df_modelo_ci.index.hour == hora_obj) &
        (df_modelo_ci['weekday_num'] == weekday_obj)
    ]
    
    df_sec_filtrado = df_modelo_sec[
        (df_modelo_sec.index.hour == hora_obj) &
        (df_modelo_sec['weekday_num'] == weekday_obj)
    ]
    
    df_pp_filtrado = df_modelo_pp[
        (df_modelo_pp.index.hour == hora_obj) &
        (df_modelo_pp['weekday_num'] == weekday_obj)
    ]
    
    # Verificar que hay datos
    if len(df_ci_filtrado) > 0 and len(df_sec_filtrado) > 0 and len(df_pp_filtrado) > 0:
        # Calcular medias históricas desde cada tabla
        media_hist = {
            'Check-in': int(df_ci_filtrado['real_pax_checkin_adj'].mean()) if 'real_pax_checkin_adj' in df_ci_filtrado.columns else 0,
            'Seguridad': int(df_sec_filtrado['real_pax_seguridad'].mean()) if 'real_pax_seguridad' in df_sec_filtrado.columns else 0,
            'Pasaporte': int(df_pp_filtrado['real_pax_passport'].mean()) if 'real_pax_passport' in df_pp_filtrado.columns else 0
        }
        
        df_comp = pd.DataFrame({
            'Zona': ['Check-in', 'Seguridad', 'Pasaporte'],
            'Media Histórica': [media_hist['Check-in'], media_hist['Seguridad'], media_hist['Pasaporte']],
            'Predicción': [predicciones['Check-in'], predicciones['Seguridad'], predicciones['Pasaporte']]
        })
        
        df_comp['Diferencia'] = df_comp['Predicción'] - df_comp['Media Histórica']
        df_comp['% Cambio'] = ((df_comp['Predicción'] / df_comp['Media Histórica']) - 1) * 100
        
        # Gráfico
        fig_comp = go.Figure()
        fig_comp.add_trace(go.Bar(
            name='Media Histórica',
            x=df_comp['Zona'],
            y=df_comp['Media Histórica'],
            marker_color='#95a5a6',
            text=df_comp['Media Histórica'],
            textposition='outside'
        ))
        fig_comp.add_trace(go.Bar(
            name='Predicción',
            x=df_comp['Zona'],
            y=df_comp['Predicción'],
            marker_color='#3498db',
            text=df_comp['Predicción'],
            textposition='outside'
        ))
        
        fig_comp.update_layout(
            barmode='group',
            title=f'Predicción vs Media Histórica ({dias_semana[weekday_obj]} a las {hora_obj}:00h)',
            height=400
        )
        
        st.plotly_chart(fig_comp, use_container_width=True)
        
        st.dataframe(df_comp.style.format({
            'Media Histórica': '{:,.0f}',
            'Predicción': '{:,.0f}',
            'Diferencia': '{:+,.0f}',
            '% Cambio': '{:+.1f}%'
        }).background_gradient(subset=['Diferencia'], cmap='RdYlGn', vmin=-500, vmax=500))
        
        # Información adicional
        col_info1, col_info2, col_info3 = st.columns(3)
        with col_info1:
            st.metric("📊 Registros Check-in", len(df_ci_filtrado))
        with col_info2:
            st.metric("📊 Registros Seguridad", len(df_sec_filtrado))
        with col_info3:
            st.metric("📊 Registros Pasaporte", len(df_pp_filtrado))
    else:
        st.info("📊 No hay suficientes datos históricos para esta combinación de día/hora")

#st.write("Estado inicial CI:", df_modelo_ci['real_pax_checkin_adj'].iloc[-1])
# ==============================================
#with tab_predicciones:
 #   st.warning("🚧 Esta sección está en fase de desarrollo.") 
    

# ==================== FOOTER ====================
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: #bdc3c7;'>
    <p style='font-size: 12px;'>
        Predicción de Afluencia Aeroportuaria | 
        {len(df_filtrado):,} registros filtrados | 
        Actualizado: {datetime.now().strftime('%d/%m/%Y %H:%M')}
    </p>
</div>
""", unsafe_allow_html=True)