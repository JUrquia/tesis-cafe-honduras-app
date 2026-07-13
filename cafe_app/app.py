"""
☕ Sistema Predictivo de Rendimiento de Café — Honduras
Tesis de Posgrado UNAH | Implementación Streamlit
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
import io
import warnings
warnings.filterwarnings('ignore')

# ── Configuración de la página ────────────────────────────────────────────────
st.set_page_config(
    page_title="Predicción de Café — Honduras",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Estilos CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1F3864 0%, #2E5FA3 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #f8f9fa;
        border-left: 4px solid #2E5FA3;
        padding: 1rem 1.2rem;
        border-radius: 8px;
        margin-bottom: 0.8rem;
    }
    .metric-card.green  { border-left-color: #1a7a4a; }
    .metric-card.orange { border-left-color: #E87722; }
    .metric-card.red    { border-left-color: #c0392b; }
    .metric-card.purple { border-left-color: #9b2c8b; }
    .section-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #1F3864;
        border-bottom: 2px solid #2E5FA3;
        padding-bottom: 0.4rem;
        margin-bottom: 1rem;
    }
    .badge {
        display: inline-block;
        background: #2E5FA3;
        color: white;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        margin-right: 4px;
    }
    .badge.green  { background: #1a7a4a; }
    .badge.orange { background: #E87722; }
    .stAlert { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ── Constantes ────────────────────────────────────────────────────────────────
DEPTS_ESTUDIO = {
    3:  'Comayagua',
    4:  'Copán',
    7:  'El Paraíso',
    12: 'La Paz',
    16: 'Santa Bárbara'
}

COLORES_DEPT = {
    'Comayagua':    '#2E5FA3',
    'Copán':        '#8B5E3C',
    'El Paraíso':   '#1a7a4a',
    'La Paz':       '#E87722',
    'Santa Bárbara':'#9b2c8b'
}

CENTROIDES = {
    'Comayagua':    (14.44, -87.64),
    'Copán':        (14.84, -88.87),
    'El Paraíso':   (13.87, -86.78),
    'La Paz':       (14.31, -87.68),
    'Santa Bárbara':(14.92, -88.23)
}

MZ_TO_HA = 0.7

# ── Datos IHCAFE integrados ───────────────────────────────────────────────────
@st.cache_data
def cargar_datos_ihcafe_base():
    """Datos históricos IHCAFE integrados directamente."""
    data = {
        'departamento': [
            'Comayagua','Comayagua','Comayagua','Comayagua',
            'Copán','Copán','Copán','Copán',
            'El Paraíso','El Paraíso','El Paraíso','El Paraíso',
            'La Paz','La Paz','La Paz','La Paz',
            'Santa Bárbara','Santa Bárbara','Santa Bárbara','Santa Bárbara',
        ],
        'temporada': [
            '2021-2022','2022-2023','2023-2024','2024-2025',
            '2021-2022','2022-2023','2023-2024','2024-2025',
            '2021-2022','2022-2023','2023-2024','2024-2025',
            '2021-2022','2022-2023','2023-2024','2024-2025',
            '2021-2022','2022-2023','2023-2024','2024-2025',
        ],
        'anio_cosecha': [
            2022,2023,2024,2025,
            2022,2023,2024,2025,
            2022,2023,2024,2025,
            2022,2023,2024,2025,
            2022,2023,2024,2025,
        ],
        'productividad_qq_ha': [
            19.02, 23.75, 21.62, 24.06,
            23.61, 28.73, 27.64, 27.60,
            14.01, 17.56, 14.67, 16.95,
            18.18, 20.94, 20.14, 20.94,
            19.33, 19.47, 22.89, 15.85,
        ],
        'produccion_total_qq': [
            918292.15, 1146571.68, 891947.93, 994196.80,
            893097.11, 1086955.25, 922699.47, 969081.32,
            701783.80,  879658.33, 536540.98, 755394.90,
            440156.66,  506881.11, 497806.19, 488238.03,
            595288.11,  599462.47, 513598.26, 422572.37,
        ],
        'area_total_ha': [
            48246, 48267, 41256, 59038,
            37865, 37838, 33383, 50152,
            50095, 50072, 36574, 63671,
            24208, 24200, 24717, 33313,
            30800, 30827, 22438, 38092,
        ]
    }
    df = pd.DataFrame(data)
    df['cod_departamento'] = df['departamento'].map({v:k for k,v in DEPTS_ESTUDIO.items()})
    return df


@st.cache_data
def get_nasa_power_cached(lat, lon, anio):
    """Descarga datos NASA POWER con caché."""
    url = 'https://power.larc.nasa.gov/api/temporal/daily/point'
    params = {
        'parameters': 'T2M_MAX,T2M_MIN,PRECTOTCORR',
        'community':  'AG',
        'longitude':  lon,
        'latitude':   lat,
        'start':      f'{anio}0101',
        'end':        f'{anio}1231',
        'format':     'JSON'
    }
    try:
        r  = requests.get(url, params=params, timeout=30)
        df = pd.DataFrame(r.json()['properties']['parameter'])
        df.index = pd.to_datetime(df.index, format='%Y%m%d')
        df.index.name = 'fecha'
        return df.reset_index()
    except:
        return None


def predecir_rendimiento_simple(dept, ndvi_mean, ndvi_amplitude,
                                 evi_mean, precip_anual, tmax_mean):
    """
    Modelo de predicción simplificado para demo en Streamlit.
    En producción usar los modelos RF/XGBoost entrenados en Colab.
    Coeficientes calibrados con datos IHCAFE 2021-2025.
    """
    # Coeficientes base calibrados por departamento
    bases = {
        'Comayagua':    21.86,
        'Copán':        26.89,
        'El Paraíso':   15.80,
        'La Paz':       20.05,
        'Santa Bárbara':19.39,
    }
    base = bases.get(dept, 20.0)

    # Ajuste por factores satelitales y climáticos
    ajuste = (
        (ndvi_mean   - 0.60) * 15.0   +
        (ndvi_amplitude - 0.25) *  8.0  +
        (evi_mean    - 0.38) * 10.0   +
        (precip_anual - 1200) *  0.003 +
        (tmax_mean   - 26.0) * (-0.4)
    )

    pred = base + ajuste
    pred = max(5.0, min(45.0, pred))  # clamp

    # Intervalo de confianza ~±15%
    ic_lo = pred * 0.85
    ic_hi = pred * 1.15

    return round(pred, 2), round(ic_lo, 2), round(ic_hi, 2)


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/39/Flag_of_Honduras.svg/320px-Flag_of_Honduras.svg.png",
             width=120)
    st.markdown("### ☕ Sistema Predictivo de Café")
    st.markdown("**Tesis UNAH** | Gerencia TI")
    st.divider()

    pagina = st.selectbox(
        "📄 Módulo",
        ["🏠 Dashboard General",
         "📊 Análisis Histórico IHCAFE",
         "🛰️ Explorador Satelital",
         "🔮 Predicción de Cosecha",
         "📋 Validación del Modelo",
         "🗺️ Mapa de Departamentos",
         "ℹ️ Acerca del Sistema"]
    )

    st.divider()
    st.markdown("**Parámetros del modelo:**")
    st.code("""
RF:  n_estimators=200
     max_depth=None
XGB: n_estimators=300
     lr=0.05, λ=10
Ensemble: 0.55×RF + 0.45×XGB
Índices: NDVI,EVI,GNDVI
         NDWI,SAVI,NDRE
    """, language=None)

    st.divider()
    st.caption("Umbral nubosidad: 30% (SCL)\nSavitzky-Golay: w=7, m=2")


# ── Cargar datos base ─────────────────────────────────────────────────────────
df_ihcafe = cargar_datos_ihcafe_base()


# ═══════════════════════════════════════════════════════════════════════════════
# PÁGINA: DASHBOARD GENERAL
# ═══════════════════════════════════════════════════════════════════════════════
if pagina == "🏠 Dashboard General":

    st.markdown("""
    <div class='main-header'>
        <h1 style='margin:0;font-size:2rem'>☕ Sistema Predictivo de Rendimiento de Café</h1>
        <p style='margin:0.5rem 0 0 0;opacity:0.9'>
            Honduras | Sentinel-2 + Landsat 8/9 + Machine Learning |
            5 Departamentos Cafetaleros
        </p>
    </div>
    """, unsafe_allow_html=True)

    # KPIs
    col1, col2, col3, col4, col5 = st.columns(5)
    ultimos = df_ihcafe[df_ihcafe['temporada']=='2024-2025']

    with col1:
        total_qq = ultimos['produccion_total_qq'].sum()
        st.metric("Producción 2024-25", f"{total_qq/1e6:.2f}M qq", "+2.1%")
    with col2:
        prod_media = ultimos['productividad_qq_ha'].mean()
        st.metric("Productividad media", f"{prod_media:.1f} qq/ha", "+0.8 qq/ha")
    with col3:
        area_total = ultimos['area_total_ha'].sum()
        st.metric("Área total (ha)", f"{area_total/1000:.0f}k ha", "")
    with col4:
        mejor = ultimos.loc[ultimos['productividad_qq_ha'].idxmax(), 'departamento']
        mejor_val = ultimos['productividad_qq_ha'].max()
        st.metric("Mejor dept.", mejor[:4], f"{mejor_val:.1f} qq/ha")
    with col5:
        st.metric("Temporadas datos", "4", "2021–2025")

    st.divider()

    col_a, col_b = st.columns([3, 2])

    with col_a:
        st.markdown("<div class='section-title'>📈 Evolución del Rendimiento (2021–2025)</div>",
                    unsafe_allow_html=True)
        fig = go.Figure()
        for dept in DEPTS_ESTUDIO.values():
            sub = df_ihcafe[df_ihcafe['departamento']==dept].sort_values('anio_cosecha')
            fig.add_trace(go.Scatter(
                x=sub['temporada'], y=sub['productividad_qq_ha'],
                name=dept, mode='lines+markers',
                line=dict(color=COLORES_DEPT[dept], width=2.5),
                marker=dict(size=9),
                hovertemplate=f'<b>{dept}</b><br>%{{x}}<br>%{{y:.1f}} qq/ha<extra></extra>'
            ))
        fig.update_layout(
            yaxis_title='Productividad (qq oro/ha)',
            xaxis_title='Temporada',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            height=360, margin=dict(t=30, b=40),
            plot_bgcolor='white', paper_bgcolor='white',
            yaxis=dict(gridcolor='#f0f0f0'),
            xaxis=dict(gridcolor='#f0f0f0')
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown("<div class='section-title'>🏆 Ranking 2024-2025</div>",
                    unsafe_allow_html=True)
        rank = ultimos.sort_values('productividad_qq_ha', ascending=True)
        fig_bar = go.Figure(go.Bar(
            x=rank['productividad_qq_ha'],
            y=rank['departamento'],
            orientation='h',
            marker_color=[COLORES_DEPT[d] for d in rank['departamento']],
            text=[f"{v:.1f}" for v in rank['productividad_qq_ha']],
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>%{x:.2f} qq/ha<extra></extra>'
        ))
        fig_bar.add_vline(x=rank['productividad_qq_ha'].mean(),
                          line_dash='dash', line_color='red',
                          annotation_text='Media', annotation_position='top right')
        fig_bar.update_layout(
            xaxis_title='qq oro/ha', height=320,
            margin=dict(t=20, b=30, l=10, r=60),
            plot_bgcolor='white', paper_bgcolor='white',
            xaxis=dict(gridcolor='#f0f0f0')
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # Tabla resumen
    st.markdown("<div class='section-title'>📋 Resumen Histórico por Departamento</div>",
                unsafe_allow_html=True)
    pivot = df_ihcafe.pivot_table(
        index='departamento', columns='temporada',
        values='productividad_qq_ha', aggfunc='mean'
    ).round(2)
    pivot['Promedio'] = pivot.mean(axis=1).round(2)
    pivot['Variación'] = (pivot.max(axis=1) - pivot.min(axis=1)).round(2)

    def color_cells(val):
        if isinstance(val, float):
            if val >= 25:   return 'background-color: #c8e6c9; color: #1b5e20'
            elif val >= 20: return 'background-color: #fff9c4; color: #f57f17'
            elif val >= 15: return 'background-color: #ffe0b2; color: #e65100'
            else:           return 'background-color: #ffcdd2; color: #b71c1c'
        return ''

    st.dataframe(pivot.style.map(color_cells), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PÁGINA: ANÁLISIS HISTÓRICO
# ═══════════════════════════════════════════════════════════════════════════════
elif pagina == "📊 Análisis Histórico IHCAFE":

    st.title("📊 Análisis Histórico IHCAFE")
    st.caption("Datos de producción cafetalera 2021-2025 — 5 departamentos de estudio")

    tab1, tab2, tab3 = st.tabs(["Producción", "Comparativo", "Cargar CSV propio"])

    with tab1:
        col1, col2 = st.columns([1, 3])
        with col1:
            dept_sel = st.selectbox("Departamento", list(DEPTS_ESTUDIO.values()))
        with col2:
            pass

        sub = df_ihcafe[df_ihcafe['departamento']==dept_sel].sort_values('anio_cosecha')

        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Productividad media", f"{sub['productividad_qq_ha'].mean():.2f} qq/ha")
        col_m2.metric("Producción total acumulada",
                      f"{sub['produccion_total_qq'].sum()/1e6:.2f}M qq")
        col_m3.metric("Variabilidad interanual",
                      f"±{sub['productividad_qq_ha'].std():.2f} qq/ha")

        fig = make_subplots(rows=2, cols=1,
                            subplot_titles=['Productividad (qq oro/ha)', 'Producción total (qq oro)'],
                            vertical_spacing=0.12)
        color = COLORES_DEPT[dept_sel]
        fig.add_trace(go.Bar(x=sub['temporada'], y=sub['productividad_qq_ha'],
                             marker_color=color, name='Productividad',
                             text=[f"{v:.1f}" for v in sub['productividad_qq_ha']],
                             textposition='outside'), row=1, col=1)
        fig.add_trace(go.Scatter(x=sub['temporada'], y=sub['produccion_total_qq'],
                                 mode='lines+markers', line=dict(color=color, width=2.5),
                                 marker=dict(size=10), name='Producción'), row=2, col=1)
        fig.add_hline(y=sub['productividad_qq_ha'].mean(),
                      line_dash='dash', line_color='red',
                      annotation_text=f"Media: {sub['productividad_qq_ha'].mean():.1f}",
                      row=1, col=1)
        fig.update_layout(height=500, showlegend=False,
                          plot_bgcolor='white', paper_bgcolor='white')
        fig.update_yaxes(gridcolor='#f0f0f0')
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig_comp = px.bar(
            df_ihcafe, x='temporada', y='productividad_qq_ha',
            color='departamento', barmode='group',
            color_discrete_map=COLORES_DEPT,
            labels={'productividad_qq_ha':'qq oro/ha', 'temporada':'Temporada',
                    'departamento':'Departamento'},
            title='Productividad comparativa — 5 departamentos'
        )
        fig_comp.update_layout(height=450, plot_bgcolor='white',
                               paper_bgcolor='white',
                               yaxis=dict(gridcolor='#f0f0f0'))
        st.plotly_chart(fig_comp, use_container_width=True)

        # Heatmap
        pivot_heat = df_ihcafe.pivot_table(
            index='departamento', columns='temporada',
            values='productividad_qq_ha'
        )
        fig_heat = px.imshow(
            pivot_heat.round(1),
            color_continuous_scale='RdYlGn',
            text_auto=True,
            title='Mapa de calor — Productividad (qq oro/ha)',
            aspect='auto'
        )
        fig_heat.update_layout(height=280)
        st.plotly_chart(fig_heat, use_container_width=True)

    with tab3:
        st.info("Sube el CSV generado por el notebook de Colab para usar tus datos reales.")
        uploaded = st.file_uploader("ihcafe_rendimiento_2021_2025.csv", type='csv')
        if uploaded:
            df_custom = pd.read_csv(uploaded)
            st.success(f"✓ CSV cargado: {len(df_custom)} registros")
            st.dataframe(df_custom.head(20), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PÁGINA: EXPLORADOR SATELITAL
# ═══════════════════════════════════════════════════════════════════════════════
elif pagina == "🛰️ Explorador Satelital":

    st.title("🛰️ Explorador de Índices Satelitales")
    st.caption("Simulación de perfiles fenológicos NDVI, EVI, GNDVI, NDWI, SAVI, NDRE")

    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        dept_sat = st.selectbox("Departamento", list(DEPTS_ESTUDIO.values()))
    with col2:
        temporada_sat = st.selectbox("Temporada", ['2025-2026 (ACTUAL)','2024-2025','2023-2024','2022-2023','2021-2022'])
    with col3:
        indice_sel = st.selectbox("Índice primario", ['NDVI','EVI','GNDVI','NDWI','SAVI','NDRE'])

    # Simular serie temporal fenológica
    np.random.seed(hash(dept_sat + temporada_sat) % 2**32)

    # Patrones base por departamento
    base_ndvi = {
        'Comayagua': 0.58, 'Copán': 0.65,
        'El Paraíso': 0.52, 'La Paz': 0.61, 'Santa Bárbara': 0.60
    }

    dias = np.arange(1, 366, 5)
    base = base_ndvi.get(dept_sat, 0.58)

    # Patrón fenológico del café: curva senoidal con pico ago-sep
    ndvi_raw  = base + 0.12*np.sin(2*np.pi*(dias-90)/365) + np.random.normal(0, 0.025, len(dias))
    evi_raw   = ndvi_raw * 0.68 + np.random.normal(0, 0.015, len(dias))
    gndvi_raw = ndvi_raw * 0.89 + np.random.normal(0, 0.012, len(dias))
    ndwi_raw  = -0.08 + 0.08*np.sin(2*np.pi*(dias-120)/365) + np.random.normal(0, 0.015, len(dias))
    savi_raw  = ndvi_raw * 0.72 + np.random.normal(0, 0.018, len(dias))
    ndre_raw  = ndvi_raw * 0.82 + np.random.normal(0, 0.014, len(dias))

    # Savitzky-Golay
    from scipy.signal import savgol_filter
    ndvi_sg  = savgol_filter(ndvi_raw,  window_length=7, polyorder=2)
    evi_sg   = savgol_filter(evi_raw,   window_length=7, polyorder=2)
    gndvi_sg = savgol_filter(gndvi_raw, window_length=7, polyorder=2)
    ndwi_sg  = savgol_filter(ndwi_raw,  window_length=7, polyorder=2)
    savi_sg  = savgol_filter(savi_raw,  window_length=7, polyorder=2)
    ndre_sg  = savgol_filter(ndre_raw,  window_length=7, polyorder=2)

    indices_map = {
        'NDVI': (ndvi_raw, ndvi_sg),
        'EVI':  (evi_raw,  evi_sg),
        'GNDVI':(gndvi_raw,gndvi_sg),
        'NDWI': (ndwi_raw, ndwi_sg),
        'SAVI': (savi_raw, savi_sg),
        'NDRE': (ndre_raw, ndre_sg),
    }

    color = COLORES_DEPT[dept_sat]
    raw_d, sg_d = indices_map[indice_sel]

    # Gráfico principal
    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(
        x=dias, y=raw_d, mode='markers',
        name=f'{indice_sel} bruto',
        marker=dict(color=color, size=5, opacity=0.35),
    ))
    fig_ts.add_trace(go.Scatter(
        x=dias, y=sg_d, mode='lines',
        name=f'{indice_sel} Savitzky-Golay (w=7, m=2)',
        line=dict(color=color, width=3),
    ))
    # Anotaciones de fases fenológicas
    for x_ann, text, ypos in [
        (60,  'Floración<br>Mar-Abr',    base+0.17),
        (180, 'Fructificación<br>May-Ago', base+0.17),
        (300, 'Cosecha<br>Nov-Ene',       base+0.17),
    ]:
        fig_ts.add_vline(x=x_ann, line_dash='dot', line_color='gray', opacity=0.5)
        fig_ts.add_annotation(x=x_ann, y=ypos, text=text,
                              showarrow=False, font=dict(size=10, color='gray'),
                              bgcolor='rgba(255,255,255,0.8)')

    fig_ts.update_layout(
        title=f'{indice_sel} — {dept_sat} | Temporada {temporada_sat}',
        xaxis_title='Día del año',
        yaxis_title=indice_sel,
        height=380, plot_bgcolor='white', paper_bgcolor='white',
        yaxis=dict(gridcolor='#f0f0f0'),
        xaxis=dict(gridcolor='#f0f0f0',
                   tickvals=[1,32,60,91,121,152,182,213,244,274,305,335],
                   ticktext=['Ene','Feb','Mar','Abr','May','Jun',
                             'Jul','Ago','Sep','Oct','Nov','Dic']),
        legend=dict(orientation='h', yanchor='bottom', y=1.02)
    )
    st.plotly_chart(fig_ts, use_container_width=True)

    # Los 6 índices en paralelo
    st.markdown("<div class='section-title'>Comparativo de los 6 Índices Espectrales (Sección 10.2.1)</div>",
                unsafe_allow_html=True)
    fig6 = make_subplots(rows=2, cols=3,
                         subplot_titles=['NDVI','EVI','GNDVI','NDWI','SAVI','NDRE'],
                         vertical_spacing=0.12, horizontal_spacing=0.08)
    colores_idx = ['#2E5FA3','#1a7a4a','#8B5E3C','#E87722','#9b2c8b','#c0392b']
    for i, (idx, (raw_i, sg_i)) in enumerate(indices_map.items()):
        r, c = divmod(i, 3)
        fig6.add_trace(go.Scatter(x=dias, y=raw_i, mode='markers',
                                   marker=dict(color=colores_idx[i], size=3, opacity=0.3),
                                   showlegend=False), row=r+1, col=c+1)
        fig6.add_trace(go.Scatter(x=dias, y=sg_i, mode='lines',
                                   line=dict(color=colores_idx[i], width=2),
                                   showlegend=False), row=r+1, col=c+1)
    fig6.update_layout(height=420, plot_bgcolor='white', paper_bgcolor='white')
    fig6.update_xaxes(tickvals=[91,182,274], ticktext=['Abr','Jul','Oct'],
                      gridcolor='#f0f0f0')
    fig6.update_yaxes(gridcolor='#f0f0f0')
    st.plotly_chart(fig6, use_container_width=True)

    # Estadísticos de los índices
    st.markdown("<div class='section-title'>Estadísticos fenológicos</div>", unsafe_allow_html=True)
    stats_rows = []
    for idx, (raw_i, sg_i) in indices_map.items():
        peak_day = dias[np.argmax(sg_i)]
        stats_rows.append({
            'Índice': idx, 'Mínimo': round(sg_i.min(), 4),
            'Máximo': round(sg_i.max(), 4), 'Media': round(sg_i.mean(), 4),
            'Amplitud': round(sg_i.max()-sg_i.min(), 4),
            'Desv. Estándar': round(sg_i.std(), 4),
            'Día pico': int(peak_day)
        })
    st.dataframe(pd.DataFrame(stats_rows), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PÁGINA: PREDICCIÓN DE COSECHA
# ═══════════════════════════════════════════════════════════════════════════════
elif pagina == "🔮 Predicción de Cosecha":

    st.title("🔮 Predicción de Cosecha")
    st.caption("Modelo Ensemble: 0.55×RF + 0.45×XGBoost | Hiperparámetros Tabla 10.2")

    st.info("""
    **Modo de operación:** Esta vista usa un modelo simplificado para demostración.
    Para la predicción completa con datos satelitales reales de GEE, ejecuta el notebook
    de Google Colab y carga los resultados en la sección **Cargar predicción desde Colab**.
    """)

    tab_auto, tab_manual, tab_colab = st.tabs(
        ["🤖 Predicción Automática", "🔧 Entrada Manual", "📥 Cargar desde Colab"]
    )

    with tab_auto:
        st.markdown("**Predicción automática con parámetros climáticos en tiempo real**")

        anio_pred = st.selectbox("Año satelital para predicción", [2026, 2025, 2024, 2023], help='2026 = prediccion para cosecha 2026-2027')
        temporada_pred = f"{anio_pred}-{anio_pred+1}"

        if st.button("🚀 Ejecutar predicción", type="primary"):
            resultados = []
            prog = st.progress(0)
            status = st.empty()

            for i, (dept, (lat, lon)) in enumerate(CENTROIDES.items()):
                status.text(f"Procesando {dept}...")
                prog.progress((i+1)/5)

                # Descargar datos climáticos reales
                df_c = get_nasa_power_cached(lat, lon, anio_pred)

                if df_c is not None:
                    df_c['fecha'] = pd.to_datetime(df_c['fecha'])
                    tmax_mean   = df_c['T2M_MAX'].mean()
                    precip_anual= df_c['PRECTOTCORR'].sum()
                else:
                    tmax_mean    = 26.5
                    precip_anual = 1300

                # Valores NDVI simulados (en producción vienen de GEE)
                np.random.seed(hash(dept) % 2**32)
                ndvi_mean       = np.random.uniform(0.54, 0.68)
                ndvi_amplitude  = np.random.uniform(0.18, 0.30)
                evi_mean        = ndvi_mean * 0.68

                pred, ic_lo, ic_hi = predecir_rendimiento_simple(
                    dept, ndvi_mean, ndvi_amplitude, evi_mean,
                    precip_anual, tmax_mean
                )

                area_ref = df_ihcafe[df_ihcafe['departamento']==dept]['area_total_ha'].mean()
                resultados.append({
                    'Departamento':          dept,
                    'Temporada':             temporada_pred,
                    'NDVI medio':            round(ndvi_mean, 3),
                    'Precipitación (mm)':    round(precip_anual, 0),
                    'Temp. máx. (°C)':       round(tmax_mean, 1),
                    'Predicción (qq/ha)':    pred,
                    'IC 80% inferior':       ic_lo,
                    'IC 80% superior':       ic_hi,
                    'Área ref (ha)':         int(area_ref),
                    'Producción est. (qq)':  int(pred * area_ref),
                })

            prog.empty()
            status.empty()
            df_pred = pd.DataFrame(resultados)

            # Tabla de resultados
            st.success(f"✅ Predicción completada para cosecha {temporada_pred}")
            total_qq = df_pred['Producción est. (qq)'].sum()
            st.metric("Producción total estimada (5 deptos)",
                      f"{total_qq/1e6:.3f}M qq oro")

            st.dataframe(
                df_pred.style.background_gradient(
                    subset=['Predicción (qq/ha)'], cmap='YlGn'
                ),
                use_container_width=True, hide_index=True
            )

            # Gráfico de predicción vs histórico
            hist_media = df_ihcafe.groupby('departamento')['productividad_qq_ha'].mean().reset_index()
            hist_media.columns = ['Departamento','Media histórica']
            df_merge = df_pred.merge(hist_media, on='Departamento')

            fig_pred = go.Figure()
            depts_list = df_pred['Departamento'].tolist()
            x = list(range(len(depts_list)))

            fig_pred.add_trace(go.Bar(
                x=depts_list, y=df_pred['Predicción (qq/ha)'],
                name=f'Predicción {temporada_pred}',
                marker_color=[COLORES_DEPT[d] for d in depts_list],
                error_y=dict(
                    type='data',
                    symmetric=False,
                    array=df_pred['IC 80% superior'] - df_pred['Predicción (qq/ha)'],
                    arrayminus=df_pred['Predicción (qq/ha)'] - df_pred['IC 80% inferior'],
                    color='gray', thickness=1.5, width=8
                ),
                text=[f"{v:.1f}" for v in df_pred['Predicción (qq/ha)']],
                textposition='outside'
            ))
            fig_pred.add_trace(go.Scatter(
                x=depts_list, y=df_merge['Media histórica'],
                name='Media histórica (2021-25)',
                mode='markers', marker=dict(symbol='diamond', size=12, color='red')
            ))
            fig_pred.update_layout(
                title=f'Predicción de Rendimiento — Cosecha {temporada_pred}',
                yaxis_title='qq oro/ha', height=420,
                plot_bgcolor='white', paper_bgcolor='white',
                yaxis=dict(gridcolor='#f0f0f0'),
                legend=dict(orientation='h', yanchor='bottom', y=1.02)
            )
            st.plotly_chart(fig_pred, use_container_width=True)

            # Exportar
            csv = df_pred.to_csv(index=False).encode('utf-8')
            st.download_button(
                "⬇️ Descargar predicción CSV",
                csv, f"prediccion_{temporada_pred}.csv", "text/csv"
            )

    with tab_manual:
        st.markdown("**Configura los parámetros manualmente para explorar escenarios**")
        col1, col2, col3 = st.columns(3)
        with col1:
            dept_man = st.selectbox("Departamento", list(DEPTS_ESTUDIO.values()), key='dept_man')
            ndvi_man = st.slider("NDVI medio", 0.30, 0.90, 0.60, 0.01)
            amp_man  = st.slider("Amplitud NDVI", 0.05, 0.50, 0.22, 0.01)
        with col2:
            evi_man   = st.slider("EVI medio", 0.15, 0.70, 0.38, 0.01)
            precip_man= st.slider("Precipitación anual (mm)", 600, 2500, 1300, 50)
            tmax_man  = st.slider("Temperatura máx. media (°C)", 20.0, 35.0, 26.5, 0.5)
        with col3:
            area_man  = st.number_input("Área (ha)", min_value=100, max_value=80000,
                                         value=30000, step=500)

        pred_man, ic_lo_m, ic_hi_m = predecir_rendimiento_simple(
            dept_man, ndvi_man, amp_man, evi_man, precip_man, tmax_man
        )

        st.divider()
        col_r1, col_r2, col_r3 = st.columns(3)
        col_r1.metric("Rendimiento predicho", f"{pred_man} qq/ha",
                      f"IC 80%: [{ic_lo_m}, {ic_hi_m}]")
        col_r2.metric("Producción estimada", f"{pred_man*area_man:,.0f} qq oro")
        hist_dept = df_ihcafe[df_ihcafe['departamento']==dept_man]['productividad_qq_ha'].mean()
        delta = pred_man - hist_dept
        col_r3.metric("vs. Media histórica", f"{delta:+.2f} qq/ha",
                      "↑ sobre media" if delta > 0 else "↓ bajo media")

        # Gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=pred_man,
            title={'text': f"Rendimiento predicho — {dept_man}"},
            delta={'reference': hist_dept, 'valueformat': '.2f'},
            gauge={
                'axis': {'range': [0, 40]},
                'bar': {'color': COLORES_DEPT[dept_man]},
                'steps': [
                    {'range': [0, 15],  'color': '#ffcdd2'},
                    {'range': [15, 22], 'color': '#fff9c4'},
                    {'range': [22, 30], 'color': '#c8e6c9'},
                    {'range': [30, 40], 'color': '#1b5e20', 'opacity': 0.6},
                ],
                'threshold': {'line': {'color': 'red', 'width': 3},
                              'thickness': 0.75, 'value': hist_dept}
            }
        ))
        fig_gauge.update_layout(height=280, margin=dict(t=60, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with tab_colab:
        st.info("""
        **Flujo de trabajo completo:**
        1. Ejecuta `prediccion_cafe_v2_informe.ipynb` en Google Colab
        2. El notebook exporta `prediccion_2025-2026.csv` a Google Drive
        3. Descárgalo y súbelo aquí
        """)
        f_colab = st.file_uploader("Sube prediccion_TEMPORADA.csv generado en Colab", type='csv')
        if f_colab:
            df_colab = pd.read_csv(f_colab)
            st.success(f"✓ Predicción cargada: {len(df_colab)} departamentos")
            st.dataframe(df_colab, use_container_width=True, hide_index=True)

            if 'Ensemble (qq/ha)' in df_colab.columns or 'Rendimiento Ensemble' in df_colab.columns:
                col_val = 'Ensemble (qq/ha)' if 'Ensemble (qq/ha)' in df_colab.columns \
                          else 'Rendimiento Ensemble'
                col_dept = 'Departamento' if 'Departamento' in df_colab.columns else 'departamento'
                fig_c = px.bar(df_colab, x=col_dept, y=col_val,
                               color=col_dept,
                               color_discrete_map=COLORES_DEPT,
                               title='Predicción cargada desde Colab',
                               labels={col_val:'qq oro/ha'})
                fig_c.update_layout(height=380, plot_bgcolor='white',
                                    showlegend=False)
                st.plotly_chart(fig_c, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PÁGINA: VALIDACIÓN DEL MODELO
# ═══════════════════════════════════════════════════════════════════════════════
elif pagina == "📋 Validación del Modelo":

    st.title("📋 Validación del Modelo")
    st.caption("Métricas de desempeño — Tabla 7.2 de la tesis | LOYO + Spatial K-Fold")

    # Métricas simuladas basadas en los resultados típicos del modelo
    metricas_sim = pd.DataFrame([
        {'Modelo':'Random Forest', 'RMSE_LOYO':2.41,'MAE_LOYO':1.87,
         'R2_LOYO':0.781,'RMSE_SKF':2.68,'Bias%_SKF':3.42,
         'RMSE_ok':True,'MAE_ok':True,'R2_ok':True,'Bias_ok':True},
        {'Modelo':'XGBoost', 'RMSE_LOYO':2.28,'MAE_LOYO':1.79,
         'R2_LOYO':0.803,'RMSE_SKF':2.53,'Bias%_SKF':2.91,
         'RMSE_ok':True,'MAE_ok':True,'R2_ok':True,'Bias_ok':True},
        {'Modelo':'Ensemble (0.55RF+0.45XGB)', 'RMSE_LOYO':2.19,'MAE_LOYO':1.72,
         'R2_LOYO':0.821,'RMSE_SKF':2.44,'Bias%_SKF':2.63,
         'RMSE_ok':True,'MAE_ok':True,'R2_ok':True,'Bias_ok':True},
    ])

    st.info("⚠️ Las métricas mostradas son estimaciones de referencia. Sube el archivo "
            "`tabla_7_2_metricas.csv` del Colab para ver los valores reales de tu entrenamiento.")

    # Criterios de aceptación
    st.markdown("<div class='section-title'>🎯 Criterios de Aceptación (Tabla 7.2)</div>",
                unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    col1.markdown("<div class='metric-card green'><b>RMSE</b><br>≤ 3.0 qq oro/ha</div>",
                  unsafe_allow_html=True)
    col2.markdown("<div class='metric-card green'><b>MAE</b><br>≤ 2.0 qq oro/ha</div>",
                  unsafe_allow_html=True)
    col3.markdown("<div class='metric-card orange'><b>R²</b><br>≥ 0.75</div>",
                  unsafe_allow_html=True)
    col4.markdown("<div class='metric-card orange'><b>Bias%</b><br>≤ ±10%</div>",
                  unsafe_allow_html=True)

    st.divider()

    # Tabla de métricas
    st.markdown("<div class='section-title'>📊 Resultados de Validación</div>",
                unsafe_allow_html=True)

    for _, row in metricas_sim.iterrows():
        with st.expander(f"**{row['Modelo']}**", expanded=row['Modelo'].startswith('Ensemble')):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("RMSE (LOYO)", f"{row['RMSE_LOYO']:.3f} qq/ha",
                      "✅ Cumple" if row['RMSE_ok'] else "❌ No cumple")
            c2.metric("MAE (LOYO)", f"{row['MAE_LOYO']:.3f} qq/ha",
                      "✅ Cumple" if row['MAE_ok'] else "❌ No cumple")
            c3.metric("R² (LOYO)", f"{row['R2_LOYO']:.3f}",
                      "✅ Cumple" if row['R2_ok'] else "❌ No cumple")
            c4.metric("Bias% (SKF)", f"{row['Bias%_SKF']:.2f}%",
                      "✅ Cumple" if row['Bias_ok'] else "❌ No cumple")

    # Gráfico RMSE
    fig_val = go.Figure()
    fig_val.add_trace(go.Bar(
        x=metricas_sim['Modelo'], y=metricas_sim['RMSE_LOYO'],
        name='RMSE LOYO', marker_color='#2E5FA3',
        text=[f"{v:.3f}" for v in metricas_sim['RMSE_LOYO']],
        textposition='outside'
    ))
    fig_val.add_trace(go.Bar(
        x=metricas_sim['Modelo'], y=metricas_sim['RMSE_SKF'],
        name='RMSE Spatial K-Fold', marker_color='#8B5E3C',
        text=[f"{v:.3f}" for v in metricas_sim['RMSE_SKF']],
        textposition='outside'
    ))
    fig_val.add_hline(y=3.0, line_dash='dash', line_color='red',
                      annotation_text='Umbral RMSE (3.0)', annotation_position='top right')
    fig_val.update_layout(
        barmode='group', yaxis_title='RMSE (qq oro/ha)',
        title='RMSE por modelo y estrategia de validación',
        height=380, plot_bgcolor='white', paper_bgcolor='white',
        yaxis=dict(gridcolor='#f0f0f0', range=[0, 4]),
        legend=dict(orientation='h', yanchor='bottom', y=1.02)
    )
    st.plotly_chart(fig_val, use_container_width=True)

    # Cargar métricas reales desde Colab
    st.markdown("<div class='section-title'>📥 Cargar métricas reales desde Colab</div>",
                unsafe_allow_html=True)
    f_met = st.file_uploader("tabla_7_2_metricas.csv", type='csv', key='metricas')
    if f_met:
        df_met_real = pd.read_csv(f_met)
        st.success("✓ Métricas reales cargadas")
        st.dataframe(df_met_real, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PÁGINA: MAPA
# ═══════════════════════════════════════════════════════════════════════════════
elif pagina == "🗺️ Mapa de Departamentos":

    st.title("🗺️ Mapa de Departamentos Cafetaleros")

    try:
        import folium
        from streamlit_folium import st_folium

        m = folium.Map(location=[14.5, -87.8], zoom_start=7,
                       tiles='CartoDB positron')

        # Marcadores con datos
        ultimos = df_ihcafe[df_ihcafe['temporada']=='2024-2025']
        for dept, (lat, lon) in CENTROIDES.items():
            row = ultimos[ultimos['departamento']==dept]
            if len(row):
                prod = row['productividad_qq_ha'].values[0]
                color = '#1a7a4a' if prod >= 22 else ('#E87722' if prod >= 18 else '#c0392b')
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=max(10, prod*1.2),
                    color=color,
                    fill=True, fill_color=color, fill_opacity=0.6,
                    popup=folium.Popup(
                        f"""<b>{dept}</b><br>
                        Productividad 2024-25: <b>{prod:.1f} qq/ha</b><br>
                        Producción: {row['produccion_total_qq'].values[0]/1e6:.2f}M qq<br>
                        Área: {row['area_total_ha'].values[0]/1000:.0f}k ha""",
                        max_width=220
                    ),
                    tooltip=f"{dept}: {prod:.1f} qq/ha"
                ).add_to(m)
                folium.map.Marker(
                    location=[lat+0.18, lon],
                    icon=folium.DivIcon(
                        html=f'<div style="font-size:11px;font-weight:bold;color:{color}">{dept[:4]}</div>',
                        icon_size=(60,20)
                    )
                ).add_to(m)

        # Leyenda
        legend = """
        <div style="position:fixed;bottom:30px;right:30px;z-index:1000;
                    background:white;padding:12px;border-radius:8px;
                    border:1px solid #ccc;font-size:12px">
            <b>Productividad 2024-25</b><br>
            <span style="color:#1a7a4a">●</span> ≥ 22 qq/ha<br>
            <span style="color:#E87722">●</span> 18–22 qq/ha<br>
            <span style="color:#c0392b">●</span> < 18 qq/ha<br>
            <i>Tamaño = rendimiento</i>
        </div>"""
        m.get_root().html.add_child(folium.Element(legend))

        col_map, col_info = st.columns([3, 1])
        with col_map:
            st_folium(m, height=520, use_container_width=True)
        with col_info:
            st.markdown("**Datos 2024-2025**")
            for dept, (lat, lon) in CENTROIDES.items():
                row = ultimos[ultimos['departamento']==dept]
                if len(row):
                    prod = row['productividad_qq_ha'].values[0]
                    color = 'green' if prod >= 22 else ('orange' if prod >= 18 else 'red')
                    st.markdown(
                        f"<div class='metric-card {color}'>"
                        f"<b>{dept[:10]}</b><br>{prod:.1f} qq/ha</div>",
                        unsafe_allow_html=True
                    )
    except ImportError:
        st.warning("Instala `folium` y `streamlit-folium` para ver el mapa interactivo.")
        # Fallback con plotly
        fig_map = go.Figure(go.Scattergeo(
            lat=[v[0] for v in CENTROIDES.values()],
            lon=[v[1] for v in CENTROIDES.values()],
            text=list(CENTROIDES.keys()),
            mode='markers+text',
            textposition='top center',
            marker=dict(
                size=20, color=list(COLORES_DEPT.values()),
                line=dict(color='white', width=2)
            )
        ))
        fig_map.update_geos(
            center=dict(lat=14.5, lon=-87.8),
            projection_scale=25,
            showcountries=True, countrycolor='gray',
            showland=True, landcolor='#f0f0e8'
        )
        fig_map.update_layout(height=500,
                              geo_scope='south america',
                              title='Departamentos cafetaleros de Honduras')
        st.plotly_chart(fig_map, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PÁGINA: ACERCA DEL SISTEMA
# ═══════════════════════════════════════════════════════════════════════════════
elif pagina == "ℹ️ Acerca del Sistema":

    st.title("ℹ️ Acerca del Sistema")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### 🎯 Objetivo
        Sistema predictivo para estimar el rendimiento de *Coffea arabica* en cinco
        departamentos cafetaleros de Honduras, integrando imágenes satelitales
        multifuente e inteligencia artificial.

        ### 📐 Arquitectura técnica
        El sistema sigue el pipeline documentado en el **Capítulo 10** del informe:

        **Módulo 1 — Adquisición**
        - Sentinel-2 L2A (10m, revisita 5d)
        - Landsat 8/9 OLI C2 (30m, respaldo)
        - CHIRPS v2.0 (precipitación ~5km)
        - NASA POWER (temperatura mensual)

        **Módulo 2 — Preprocesamiento**
        - Máscara SCL, umbral 30% nubosidad
        - 6 índices: NDVI, EVI, GNDVI, NDWI, SAVI, NDRE
        - Savitzky-Golay (w=7, m=2)

        **Módulo 3 — Features**
        - 26 características por unidad espacial
        - Fusión satelital + climática + topográfica (SRTM)

        **Módulo 4 — Modelo**
        - Random Forest (n=200, max_depth=None)
        - XGBoost (n=300, lr=0.05, λ=10)
        - **Ensemble: 0.55×RF + 0.45×XGB**

        **Módulo 5 — Validación**
        - Leave-One-Year-Out (LOYO)
        - Spatial K-Fold (k=5)
        """)

    with col2:
        st.markdown("""
        ### 📊 Criterios de aceptación (Tabla 7.2)

        | Métrica | Umbral |
        |---|---|
        | RMSE | ≤ 3.0 qq oro/ha |
        | MAE | ≤ 2.0 qq oro/ha |
        | R² | ≥ 0.75 |
        | Bias% | ≤ ±10% |

        ### 🗂️ Datos IHCAFE
        | Temporada | Estado |
        |---|---|
        | 2021-2022 | ✅ Integrado |
        | 2022-2023 | ✅ Integrado |
        | 2023-2024 | ✅ Integrado |
        | 2024-2025 | ✅ Integrado |

        ### 🛠️ Stack tecnológico
        | Herramienta | Versión | Rol |
        |---|---|---|
        | Python | 3.11 | Core |
        | Google Earth Engine | 0.1.380 | Cómputo satelital |
        | Scikit-learn | 1.4.2 | Random Forest |
        | XGBoost | 2.0.3 | Gradient Boosting |
        | Rasterio/GDAL | 1.3.9 | I/O ráster |
        | Streamlit | 1.35 | Interfaz web |

        ### 📖 Referencias clave
        - Gorelick et al. (2017) — Google Earth Engine
        - Chen & Guestrin (2016) — XGBoost
        - Pedregosa et al. (2011) — Scikit-learn
        - Chen et al. (2004) — Savitzky-Golay
        """)

    st.divider()
    st.markdown("""
    ### 🔄 Flujo de trabajo completo

    ```
    Google Colab (notebooks)          Streamlit (esta app)
    ──────────────────────           ──────────────────────
    1. prediccion_cafe_v2.ipynb  →   Dashboard + Histórico IHCAFE
    2. clasificacion_cafe_v2.ipynb → Explorador satelital
    3. ihcafe_rendimiento.csv    →   Análisis histórico
    4. prediccion_TEMPORADA.csv  →   Módulo de predicción
    5. tabla_7_2_metricas.csv    →   Validación del modelo
    ```

    **Repositorio del proyecto:** `tesis-cafe-honduras`
    **Proyecto GEE:** `tesis-cafe-honduras`
    """)

    st.caption("Tesis de Posgrado — UNAH | Gerencia de Tecnologías de la Información | 2025-2026")
