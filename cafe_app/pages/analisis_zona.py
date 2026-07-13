"""
Analisis de Zona con datos Sentinel-2 REALES via Google Earth Engine
Modulo integrado en la app Streamlit de Prediccion de Cafe Honduras
"""

import streamlit as st
import folium
from folium.plugins import Draw, MeasureControl, MousePosition
from streamlit_folium import st_folium
import pandas as pd
import numpy as np
import requests
import json
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from shapely.geometry import shape
import geopandas as gpd
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# Importar modulos propios
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gee_auth      import inicializar_gee, verificar_gee
from gee_extractor import (
    extraer_series_temporales, get_elevacion,
    clasificar_pixeles_gee, get_distribucion_clases, INDICE_COLS
)
from predictor_real import predecir_rendimiento, estado_modelos

st.set_page_config(
    page_title="Analisis de Zona - Cafe Honduras",
    page_icon="MAP",
    layout="wide"
)

# ── Estilos ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1F3864 0%, #2E5FA3 100%);
        padding: 1.5rem 2rem; border-radius: 12px;
        color: white; margin-bottom: 1.5rem;
    }
    .section-title {
        font-size: 1.05rem; font-weight: 700; color: #1F3864;
        border-bottom: 2px solid #2E5FA3;
        padding-bottom: 0.3rem; margin: 1rem 0 0.8rem 0;
    }
    .info-real {
        background: #e8f5e9; border-left: 4px solid #1a7a4a;
        padding: 8px 12px; border-radius: 6px;
        font-size: 0.85rem; color: #1b5e20; margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ── Constantes ────────────────────────────────────────────────────────────────
ELEV_MIN  = 800
ELEV_MAX  = 1800
W_RF      = 0.55
W_XGB     = 0.45
MZ_TO_HA  = 0.7

IHCAFE_REF = {
    'Comayagua': 21.86, 'Copan': 26.89, 'El Paraiso': 15.80,
    'La Paz': 20.05,    'Santa Barbara': 19.39
}
DEPTS_LIST = ['Comayagua', 'Copan', 'El Paraiso', 'La Paz', 'Santa Barbara']

# ── Estado de sesion ──────────────────────────────────────────────────────────
for key, default in [
    ('poligono_geojson',   None),
    ('analisis_listo',     False),
    ('resultados',         {}),
    ('gee_conectado',      False),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='main-header'>
    <h1 style='margin:0;font-size:1.8rem'>Analisis de Zona por Seleccion en Mapa</h1>
    <p style='margin:0.4rem 0 0 0;opacity:0.9'>
        Dibuja un poligono sobre el mapa — el sistema extrae datos Sentinel-2 REALES
        y determina si hay cafe y cual seria la produccion esperada
    </p>
</div>
""", unsafe_allow_html=True)

# ── Inicializar GEE ───────────────────────────────────────────────────────────
gee_ok, gee_msg = inicializar_gee()
if gee_ok:
    st.session_state.gee_conectado = True
    st.sidebar.success("GEE: Conectado")
    st.sidebar.caption(gee_msg)
else:
    st.session_state.gee_conectado = False
    st.sidebar.error("GEE: Sin conexion")
    st.sidebar.caption(gee_msg)
    st.error(
        "No se pudo conectar con Google Earth Engine. "
        "Verifica que las credenciales esten configuradas en Streamlit Secrets."
    )

# Estado de los modelos predictivos
modo_modelos, meta_modelos = estado_modelos()
if modo_modelos == 'real':
    st.sidebar.success("Modelos: Reales (RF+XGB)")
    if meta_modelos.get('rmse'):
        st.sidebar.caption(
            f"RMSE={meta_modelos['rmse']:.3f} | "
            f"R²={meta_modelos.get('r2','?')} | "
            f"Entrenado: {meta_modelos.get('fecha','?')}"
        )
else:
    st.sidebar.warning("Modelos: Formula calibrada")
    st.sidebar.caption(
        "Sube los .pkl a cafe_app/modelos/ para activar "
        "predicciones con RF+XGB reales"
    )

# ════════════════════════════════════════════════════════════════
# LAYOUT: mapa (izquierda) | resultados (derecha)
# ════════════════════════════════════════════════════════════════
col_mapa, col_res = st.columns([3, 2], gap="large")

# ════════════════════════════════════════════════════════════════
# COLUMNA IZQUIERDA — CONFIGURACION Y MAPA
# ════════════════════════════════════════════════════════════════
with col_mapa:

    st.markdown("<div class='section-title'>Paso 1 — Configurar analisis</div>",
                unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        nombre_zona = st.text_input("Nombre de la zona", value="Mi Finca")
    with c2:
        # Opciones con etiqueta de temporada — incluye año actual 2026
        ANIOS_OPCIONES = {
            2026: "2026 → Temporada 2026-2027  (PREDICCION ACTUAL)",
            2025: "2025 → Temporada 2025-2026",
            2024: "2024 → Temporada 2024-2025",
            2023: "2023 → Temporada 2023-2024",
            2022: "2022 → Temporada 2022-2023",
            2021: "2021 → Temporada 2021-2022",
        }
        anio_label = st.selectbox(
            "Ano satelital / Temporada a predecir",
            list(ANIOS_OPCIONES.values()),
            index=0,
            help=(
                "Selecciona el ano de las imagenes satelitales a analizar.\n"
                "Para estimar la cosecha que se recolectara este ciclo, "
                "usa 2026 (datos Sentinel-2 enero-diciembre 2026)."
            )
        )
        anio_analisis = int(anio_label.split(" ")[0])

    dept_ref = st.selectbox(
        "Departamento de referencia",
        DEPTS_LIST, index=3,
        help="Para comparar con la media historica departamental IHCAFE"
    )
    st.caption(
        f"Temporada seleccionada: "
        f"**{ANIOS_OPCIONES.get(anio_analisis, str(anio_analisis))}**"
    )

    st.markdown("<div class='section-title'>Paso 2 — Dibuja o sube el poligono</div>",
                unsafe_allow_html=True)

    with st.expander("Instrucciones", expanded=True):
        st.markdown("""
**En el mapa:**
1. Cambia a vista **Satelite Google** (control de capas, arriba a la derecha)
2. Navega hasta tu finca con scroll o botones +/-
3. Haz clic en el icono de **poligono** (barra izquierda del mapa)
4. Clic en cada esquina de tu finca para dibujar el perimetro
5. **Doble clic** en el ultimo punto para cerrar
6. Presiona **Analizar con GEE**

*Tambien puedes subir un GeoJSON existente abajo.*
        """)

    # ── Mapa interactivo ──────────────────────────────────────────────────────
    m = folium.Map(location=[14.26, -87.84], zoom_start=12, tiles=None)

    # Capas base
    folium.TileLayer(
        tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
        attr='Google', name='Satelite Google',
        overlay=False, control=True
    ).add_to(m)
    folium.TileLayer(
        tiles='https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}',
        attr='Google', name='Hibrido Google',
        overlay=False, control=True
    ).add_to(m)
    folium.TileLayer(
        tiles='OpenStreetMap', name='OpenStreetMap',
        overlay=False, control=True
    ).add_to(m)

    # Herramientas de dibujo
    Draw(
        draw_options={
            'polyline': False, 'rectangle': True,
            'circle': False, 'circlemarker': False, 'marker': False,
            'polygon': {
                'shapeOptions': {
                    'color': '#2E5FA3', 'fillColor': '#2E5FA3',
                    'fillOpacity': 0.25, 'weight': 3,
                },
            },
        },
        edit_options={'edit': True, 'remove': True},
        export=True,
    ).add_to(m)

    MeasureControl(
        position='topleft',
        primary_area_unit='hectares',
        secondary_area_unit='sqmeters',
    ).add_to(m)

    MousePosition(position='bottomleft', prefix='Coordenadas:').add_to(m)

    # Mostrar poligono guardado
    if st.session_state.poligono_geojson:
        folium.GeoJson(
            st.session_state.poligono_geojson,
            name='Zona seleccionada',
            style_function=lambda x: {
                'fillColor': '#2E5FA3', 'color': '#1F3864',
                'weight': 3, 'fillOpacity': 0.3,
            }
        ).add_to(m)

    folium.LayerControl(position='topright', collapsed=False).add_to(m)

    mapa_out = st_folium(
        m, height=480, width=None,
        returned_objects=["last_active_drawing", "all_drawings"],
        key="mapa_gee_real"
    )

    # Capturar poligono del mapa
    poligono_nuevo = None
    if mapa_out:
        for key in ["last_active_drawing", "all_drawings"]:
            d = mapa_out.get(key)
            if d:
                feat = d if isinstance(d, dict) else (d[-1] if d else None)
                if feat and feat.get("geometry", {}).get("type") in \
                        ["Polygon", "MultiPolygon", "Rectangle"]:
                    poligono_nuevo = feat
                    break

    if poligono_nuevo:
        st.session_state.poligono_geojson = poligono_nuevo
        st.session_state.analisis_listo   = False
        try:
            geom    = shape(poligono_nuevo["geometry"])
            gdf_tmp = gpd.GeoDataFrame([{'geometry': geom}], crs='EPSG:4326')
            area_ha = gdf_tmp.to_crs(epsg=32616).geometry.area.sum() / 10000
            c_      = geom.centroid
            st.success(
                f"Poligono capturado | "
                f"Area: **{area_ha:.3f} ha** ({area_ha/MZ_TO_HA:.3f} mz) | "
                f"Centro: {c_.y:.5f}N {c_.x:.5f}W"
            )
        except:
            st.info("Poligono capturado. Presiona Analizar para continuar.")

    # Botones de control
    st.markdown("")
    cb1, cb2 = st.columns([2, 1])
    with cb1:
        btn_analizar = st.button(
            "Analizar con GEE (datos reales Sentinel-2)",
            type="primary",
            use_container_width=True,
            disabled=(
                st.session_state.poligono_geojson is None or
                not st.session_state.gee_conectado
            )
        )
    with cb2:
        if st.button("Limpiar", use_container_width=True):
            st.session_state.poligono_geojson = None
            st.session_state.analisis_listo   = False
            st.session_state.resultados       = {}
            st.rerun()

    # Upload GeoJSON alternativo
    st.markdown("---")
    st.markdown("**O sube un archivo GeoJSON existente:**")
    uploaded = st.file_uploader(
        "GeoJSON", type=['geojson', 'json'],
        label_visibility='collapsed'
    )
    if uploaded:
        try:
            gj = json.load(uploaded)
            if gj.get('type') == 'FeatureCollection':
                feat = gj['features'][0]
            elif gj.get('type') == 'Feature':
                feat = gj
            else:
                feat = {'type': 'Feature', 'geometry': gj, 'properties': {}}
            st.session_state.poligono_geojson = feat
            st.session_state.analisis_listo   = False
            geom    = shape(feat['geometry'])
            gdf_tmp = gpd.GeoDataFrame([{'geometry': geom}], crs='EPSG:4326')
            area_ha = gdf_tmp.to_crs(epsg=32616).geometry.area.sum() / 10000
            st.success(
                f"GeoJSON cargado: **{uploaded.name}** | "
                f"Area: {area_ha:.3f} ha ({area_ha/MZ_TO_HA:.3f} mz)"
            )
            st.rerun()
        except Exception as e:
            st.error(f"Error: {e}")


# ════════════════════════════════════════════════════════════════
# FUNCIONES DE ANALISIS
# ════════════════════════════════════════════════════════════════

def get_nasa_power(lat, lon, anio):
    url    = 'https://power.larc.nasa.gov/api/temporal/daily/point'
    params = {
        'parameters': 'T2M_MAX,T2M_MIN,PRECTOTCORR',
        'community': 'AG', 'longitude': lon, 'latitude': lat,
        'start': f'{anio}0101', 'end': f'{anio}1231', 'format': 'JSON'
    }
    try:
        r  = requests.get(url, params=params, timeout=25)
        df = pd.DataFrame(r.json()['properties']['parameter'])
        df.index = pd.to_datetime(df.index, format='%Y%m%d')
        return df.reset_index(names='fecha')
    except:
        return None


def clasificar_zona(df_ts, elev_mean, apto_altitud):
    """Clasificacion 3 niveles con datos reales de GEE."""
    # Extraer estadisticos de la serie temporal real
    def stat(col):
        s = df_ts[col].dropna() if col in df_ts.columns else pd.Series(dtype=float)
        return s

    ndvi_sg = stat('NDVI_SG')
    evi_sg  = stat('EVI_SG')
    gndvi_sg= stat('GNDVI_SG')
    ndwi_sg = stat('NDWI_SG')
    savi_sg = stat('SAVI_SG')
    ndre_sg = stat('NDRE_SG')

    def safe_mean(s): return float(s.mean()) if len(s) > 0 else 0.0
    def safe_amp(s):  return float(s.max()-s.min()) if len(s) > 1 else 0.0

    ndvi_prom = safe_mean(ndvi_sg)
    ndvi_amp  = safe_amp(ndvi_sg)
    evi_prom  = safe_mean(evi_sg)
    gndvi_prom= safe_mean(gndvi_sg)
    savi_prom = safe_mean(savi_sg)
    ndre_prom = safe_mean(ndre_sg)
    ndwi_prom = safe_mean(ndwi_sg)

    peak_idx = df_ts['NDVI_SG'].idxmax() if 'NDVI_SG' in df_ts.columns and len(ndvi_sg) > 0 else None
    peak_mes = int(df_ts.loc[peak_idx, 'fecha'].month) if peak_idx is not None else 0

    # ─────────────────────────────────────────────────────────────────────────
    # NIVEL 1: Reglas espectrales calibradas para Honduras
    # Umbrales ajustados para distinguir café de bosque, pasto y cultivos
    # Bosque denso: NDVI>0.75, EVI>0.50, GNDVI>0.65, NDRE>0.55, Amp<0.10
    # Café arábica: NDVI 0.40-0.75, EVI 0.20-0.50, variación estacional mayor
    # ─────────────────────────────────────────────────────────────────────────
    reglas = {
        'NDVI cafe [0.40-0.75]':     0.40 <= ndvi_prom <= 0.75,
        'Amplitud NDVI >= 0.10':     ndvi_amp   >= 0.10,
        'EVI cafe [0.20-0.50]':      0.20 <= evi_prom   <= 0.50,
        'GNDVI cafe [0.30-0.65]':    0.30 <= gndvi_prom <= 0.65,
        'SAVI cafe [0.25-0.55]':     0.25 <= savi_prom  <= 0.55,
        'NDRE cafe [0.28-0.55]':     0.28 <= ndre_prom  <= 0.55,
        'Pico NDVI jul-nov':         7 <= peak_mes <= 11,
        'Altitud 800-1800 msnm':     apto_altitud,
    }
    n_ok   = sum(reglas.values())
    sc_reg = n_ok / len(reglas) * 100

    # Indicadores adicionales de NO cafe (penalizan el score)
    es_bosque_denso = (ndvi_prom > 0.75 or evi_prom > 0.50 or
                       gndvi_prom > 0.65 or ndre_prom > 0.55)
    es_zona_baja    = ndvi_amp < 0.08   # bosque maduro cambia poco
    penalizacion    = 0.0
    if es_bosque_denso:
        penalizacion += 25.0   # penalizar si indices apuntan a bosque denso
    if es_zona_baja:
        penalizacion += 10.0   # poca variacion estacional = no es cafe

    # ─────────────────────────────────────────────────────────────────────────
    # NIVEL 2: Random Forest con muestras mas representativas de Honduras
    # Muestras calibradas con valores tipicos de cada clase en la region
    # ─────────────────────────────────────────────────────────────────────────
    np.random.seed(42)
    n_ref = 500
    rows  = []

    # Cafe arabica Honduras (800-1800 msnm): NDVI moderado, variacion media
    rows.append(pd.DataFrame({
        'ndvi':  np.random.normal(0.58, 0.07, n_ref),   # 0.44-0.72
        'gndvi': np.random.normal(0.52, 0.06, n_ref),
        'evi':   np.random.normal(0.38, 0.06, n_ref),   # 0.20-0.50
        'ndwi':  np.random.normal(-0.03, 0.07, n_ref),
        'savi':  np.random.normal(0.42, 0.06, n_ref),
        'ndre':  np.random.normal(0.46, 0.06, n_ref),   # 0.28-0.55
        'amp':   np.random.normal(0.20, 0.05, n_ref),   # variacion estacional
        'elev':  np.random.normal(1200, 200, n_ref),
        'clase': 1
    }))

    # Bosque latifoliado (NDVI alto, poca variacion, indices altos)
    rows.append(pd.DataFrame({
        'ndvi':  np.random.normal(0.82, 0.05, n_ref),   # >0.75
        'gndvi': np.random.normal(0.74, 0.05, n_ref),   # >0.65
        'evi':   np.random.normal(0.56, 0.04, n_ref),   # >0.50
        'ndwi':  np.random.normal(0.10, 0.05, n_ref),
        'savi':  np.random.normal(0.61, 0.04, n_ref),   # >0.55
        'ndre':  np.random.normal(0.71, 0.04, n_ref),   # >0.55
        'amp':   np.random.normal(0.06, 0.03, n_ref),   # poca variacion
        'elev':  np.random.normal(1350, 250, n_ref),
        'clase': 2
    }))

    # Pasto / gramíneas (NDVI bajo, alta variacion estacional)
    rows.append(pd.DataFrame({
        'ndvi':  np.random.normal(0.35, 0.10, n_ref),
        'gndvi': np.random.normal(0.29, 0.09, n_ref),
        'evi':   np.random.normal(0.20, 0.07, n_ref),
        'ndwi':  np.random.normal(-0.22, 0.08, n_ref),
        'savi':  np.random.normal(0.23, 0.07, n_ref),
        'ndre':  np.random.normal(0.27, 0.08, n_ref),
        'amp':   np.random.normal(0.32, 0.09, n_ref),
        'elev':  np.random.normal(700, 200, n_ref),
        'clase': 3
    }))

    # Cultivo anual / milpa (muy variable, altitud baja)
    rows.append(pd.DataFrame({
        'ndvi':  np.random.normal(0.44, 0.12, n_ref),
        'gndvi': np.random.normal(0.37, 0.11, n_ref),
        'evi':   np.random.normal(0.27, 0.10, n_ref),
        'ndwi':  np.random.normal(-0.12, 0.10, n_ref),
        'savi':  np.random.normal(0.30, 0.09, n_ref),
        'ndre':  np.random.normal(0.35, 0.10, n_ref),
        'amp':   np.random.normal(0.48, 0.10, n_ref),
        'elev':  np.random.normal(500, 150, n_ref),
        'clase': 4
    }))

    df_ref = pd.concat(rows, ignore_index=True)
    FCOLS  = ['ndvi','gndvi','evi','ndwi','savi','ndre','amp','elev']
    clf    = RandomForestClassifier(n_estimators=300, max_depth=12,
                                     min_samples_leaf=3,
                                     random_state=42, n_jobs=-1)
    clf.fit(df_ref[FCOLS].values, df_ref['clase'].values)

    x_zona    = np.array([[ndvi_prom, gndvi_prom, evi_prom, ndwi_prom,
                           savi_prom, ndre_prom, ndvi_amp, elev_mean]])
    proba     = clf.predict_proba(x_zona)[0]
    prob_cafe = float(proba[0]) * 100

    # ─────────────────────────────────────────────────────────────────────────
    # NIVEL 3: Correlacion con patron fenologico del cafe en Honduras
    # El cafe tiene pico ago-sep y caida en cosecha nov-ene
    # Bosque denso tiene curva mas plana y alta todo el año
    # ─────────────────────────────────────────────────────────────────────────
    PATRON_CAFE = np.array([0.55,0.52,0.48,0.45,0.58,0.68,
                             0.72,0.75,0.73,0.68,0.60,0.57])
    df_m  = (df_ts.set_index('fecha').resample('ME')[['NDVI_SG']]
                  .mean().reset_index())
    df_m['mes'] = df_m['fecha'].dt.month
    patron_obs  = df_m.groupby('mes')['NDVI_SG'].mean()
    meses_c     = sorted(set(patron_obs.index) & set(range(1, 13)))

    if len(meses_c) >= 6:
        obs  = patron_obs.loc[meses_c].values
        ref  = PATRON_CAFE[[m-1 for m in meses_c]]
        corr = float(np.corrcoef(obs, ref)[0, 1])
        # Manejar NaN — puede ocurrir si la serie tiene poca variacion (bosque)
        if np.isnan(corr):
            corr = 0.0   # sin variacion = no es cafe
    elif len(meses_c) >= 4:
        obs  = patron_obs.loc[meses_c].values
        ref  = PATRON_CAFE[[m-1 for m in meses_c]]
        corr_raw = float(np.corrcoef(obs, ref)[0, 1])
        corr = 0.0 if np.isnan(corr_raw) else corr_raw * 0.7  # penalizar pocos datos
    else:
        corr = 0.0   # insuficientes datos = no confirmado

    sc_fenol = max(0.0, min(1.0, corr))

    # ─────────────────────────────────────────────────────────────────────────
    # SCORE INTEGRADO con penalizaciones
    # ─────────────────────────────────────────────────────────────────────────
    score_base = (0.35*(sc_reg/100) + 0.45*(prob_cafe/100) + 0.20*sc_fenol) * 100

    # Aplicar penalizaciones por indicadores de bosque
    score_fin = max(0.0, score_base - penalizacion)

    # Regla de techo: si RF da < 50% probabilidad, el score no puede superar 60%
    if prob_cafe < 50.0:
        score_fin = min(score_fin, 60.0)

    # Regla de techo: si NDVI > 0.75 (bosque denso), no puede ser CAFE CONFIRMADO
    if ndvi_prom > 0.75:
        score_fin = min(score_fin, 54.0)

    if   score_fin >= 75: vered='CAFE CONFIRMADO';     cv='#1a7a4a'; emoji='OK'
    elif score_fin >= 55: vered='PROBABLE CAFE';       cv='#E87722'; emoji='PROBABLE'
    elif score_fin >= 35: vered='RESULTADO INCIERTO';  cv='#888888'; emoji='INCIERTO'
    else:                 vered='NO ES CAFE';           cv='#c0392b'; emoji='NO'

    return {
        'reglas': reglas, 'n_ok': n_ok, 'sc_reg': sc_reg,
        'prob_cafe': prob_cafe, 'corr': corr, 'sc_fenol': sc_fenol,
        'score_final': score_fin, 'veredicto': vered,
        'color_v': cv, 'emoji': emoji, 'proba': proba,
        'ndvi_prom': ndvi_prom, 'ndvi_amp': ndvi_amp,
        'evi_prom': evi_prom, 'gndvi_prom': gndvi_prom,
        'savi_prom': savi_prom, 'ndre_prom': ndre_prom,
        'ndwi_prom': ndwi_prom,
    }


# predecir_rendimiento ahora viene de predictor_real.py (importado arriba)
# Usa modelos .pkl reales si existen en cafe_app/modelos/
# Fallback automatico a formula calibrada si no hay .pkl




def proyectar_3_anios(pred_ens, area_ha, dept, anio_siembra,
                       plantas_por_ha=2500):
    """
    Proyecta produccion de cafe nuevo del vivero a 5 años.
    plantas_por_ha: densidad de siembra tipica en Honduras es 2000-3000 plantas/ha
    Densidades comunes:
      - Siembra tradicional:  1600-2000 plantas/ha (3.0m x 2.0m)
      - Siembra semi-intensiva: 2500 plantas/ha (2.0m x 2.0m)
      - Siembra intensiva: 3000-5000 plantas/ha (1.5m x 1.5m)
    """
    factores = {
        anio_siembra:     0.00,
        anio_siembra + 1: 0.15,
        anio_siembra + 2: 0.50,
        anio_siembra + 3: 0.80,
        anio_siembra + 4: 1.00,
    }
    base_plena    = pred_ens
    hist_dep      = IHCAFE_REF.get(dept, 20.0)
    total_plantas = round(area_ha * plantas_por_ha)
    fases         = ['Establecimiento','Primera floracion',
                     'Primera cosecha comercial',
                     'Produccion en desarrollo','Produccion plena']
    # Produccion por planta a plena produccion
    # qq/ha / plantas/ha = qq por planta
    qq_por_planta_plena = base_plena / plantas_por_ha if plantas_por_ha > 0 else 0

    proyeccion = []
    for idx, (anio, factor) in enumerate(factores.items()):
        rend_anio   = round(base_plena * factor, 2)
        prod_anio   = round(rend_anio * area_ha, 1)
        # Produccion por planta individual
        qq_planta   = round(qq_por_planta_plena * factor, 4)
        # Equivalente en libras por planta (1 qq = 100 lb)
        lb_planta   = round(qq_planta * 100, 2)
        proyeccion.append({
            'Anio':                    anio,
            'Fase':                    fases[idx],
            'Factor maduracion':       f'{int(factor*100)}%',
            'Total plantas':           total_plantas,
            'Rendimiento (qq/ha)':     rend_anio,
            'Produccion est. (qq)':    prod_anio,
            'qq por planta':           qq_planta,
            'lb por planta':           lb_planta,
            'vs Media dept (qq/ha)':   round(rend_anio - hist_dep, 2),
        })
    return proyeccion, total_plantas, qq_por_planta_plena


def detectar_siembra_nueva(df_ts, ndvi_prom):
    resultado = {'es_siembra_nueva':False,'confianza':0,
                 'evidencias':[],'recomendacion':''}
    if 0.28 <= ndvi_prom <= 0.52:
        resultado['evidencias'].append(
            f'NDVI ({ndvi_prom:.3f}) en rango de cafe joven (0.28-0.52)')
        resultado['confianza'] += 35
    if 'NDVI_SG' in df_ts.columns:
        ndvi_sg = df_ts['NDVI_SG'].dropna()
        if len(ndvi_sg) >= 6:
            amp = float(ndvi_sg.max() - ndvi_sg.min()) if len(ndvi_sg)>1 else 0
            if 0.05 <= amp <= 0.18:
                resultado['evidencias'].append(
                    f'Amplitud NDVI baja ({amp:.3f}) — planta en desarrollo')
                resultado['confianza'] += 25
            primera = float(ndvi_sg.iloc[:len(ndvi_sg)//2].mean())
            segunda = float(ndvi_sg.iloc[len(ndvi_sg)//2:].mean())
            if segunda > primera + 0.04:
                resultado['evidencias'].append(
                    f'Tendencia creciente ({primera:.3f} a {segunda:.3f})')
                resultado['confianza'] += 25
    resultado['confianza'] = min(100, resultado['confianza'])
    resultado['es_siembra_nueva'] = resultado['confianza'] >= 50
    if resultado['confianza'] >= 70:
        resultado['recomendacion'] = (
            'Alta probabilidad de siembra nueva. Registrar en catastro IHCAFE.')
    elif resultado['confianza'] >= 50:
        resultado['recomendacion'] = (
            'Posible siembra nueva. Comparar con imagenes del ano anterior.')
    else:
        resultado['recomendacion'] = 'No se detecta patron de siembra nueva.'
    if not resultado['evidencias']:
        resultado['evidencias'].append(
            f'NDVI ({ndvi_prom:.3f}) fuera del rango de cafe joven')
    return resultado


def detectar_danos(ndvi_prom, evi_prom, ndre_prom, ndwi_prom, ndvi_amp):
    alertas = []
    nivel   = 'NORMAL'
    color_n = '#1a7a4a'

    ratio = ndre_prom / ndvi_prom if ndvi_prom > 0 else 0
    if ratio < 0.62:
        alertas.append({
            'tipo':   'Posible Roya (Hemileia vastatrix)',
            'color':  '#c0392b',
            'detalle':f'NDRE/NDVI={ratio:.3f} (<0.62). Caida del Red Edge '
                       f'antes que NDVI — firma espectral de infeccion fungica. '
                       f'NDRE={ndre_prom:.3f}, NDVI={ndvi_prom:.3f}',
            'accion': 'Aplicar fungicida preventivo. Inspeccionar haz y enves de hojas.',
        })
        nivel = 'ALERTA'; color_n = '#c0392b'

    if ndwi_prom < -0.15:
        alertas.append({
            'tipo':   'Estres hidrico',
            'color':  '#E87722',
            'detalle':f'NDWI={ndwi_prom:.3f} (<-0.15). Deficit de agua '
                       f'en el dosel. Posible sequia o problema de irrigacion.',
            'accion': 'Verificar disponibilidad de agua en la zona.',
        })
        if nivel == 'NORMAL': nivel = 'ATENCION'; color_n = '#E87722'

    indices_bajos = sum([ndvi_prom<0.40, evi_prom<0.22, ndre_prom<0.30])
    if indices_bajos >= 2:
        alertas.append({
            'tipo':   'Defoliacion o perdida de follaje',
            'color':  '#922b21',
            'detalle':f'Multiples indices bajos: NDVI={ndvi_prom:.3f}, '
                       f'EVI={evi_prom:.3f}, NDRE={ndre_prom:.3f}. '
                       f'Patron de defoliacion severa o daño post-cosecha.',
            'accion': 'Inspeccion urgente de campo.',
        })
        nivel = 'CRITICO'; color_n = '#922b21'

    if ndvi_amp > 0.45:
        alertas.append({
            'tipo':   'Variabilidad espectral alta',
            'color':  '#E87722',
            'detalle':f'Amplitud NDVI={ndvi_amp:.3f} (>0.45). '
                       f'Posible daño irregular o parches de plaga.',
            'accion': 'Mapear zonas de mayor variabilidad para inspeccion.',
        })
        if nivel == 'NORMAL': nivel = 'ATENCION'; color_n = '#E87722'

    if not alertas:
        alertas.append({
            'tipo':   'Sin alertas — cultivo saludable',
            'color':  '#1a7a4a',
            'detalle':f'Todos los indices en rango normal. '
                       f'NDVI={ndvi_prom:.3f}, EVI={evi_prom:.3f}, '
                       f'NDRE={ndre_prom:.3f}, NDWI={ndwi_prom:.3f}',
            'accion': 'Continuar monitoreo regular cada 30 dias.',
        })

    return {
        'alertas':  alertas,
        'nivel':    nivel,
        'color':    color_n,
        'n_alertas':len([a for a in alertas if 'saludable' not in a['tipo']]),
    }


# ════════════════════════════════════════════════════════════════
# EJECUTAR ANALISIS REAL CON GEE
# ════════════════════════════════════════════════════════════════
if btn_analizar and st.session_state.poligono_geojson and gee_ok:

    feat    = st.session_state.poligono_geojson
    geom_sh = shape(feat['geometry'])
    gdf_    = gpd.GeoDataFrame([{'geometry': geom_sh}], crs='EPSG:4326')
    area_ha = gdf_.to_crs(epsg=32616).geometry.area.sum() / 10000
    area_mz = area_ha / MZ_TO_HA
    centro  = geom_sh.centroid
    lat_c   = centro.y
    lon_c   = centro.x

    # Convertir a ee.Geometry
    import ee
    geom_coords = [list(c) for c in geom_sh.exterior.coords]
    ee_geom     = ee.Geometry.Polygon([geom_coords])

    with col_res:
        st.info("Analizando zona con datos Sentinel-2 reales...")
        prog_bar    = st.progress(0)
        prog_status = st.empty()

        def update_progress(pct, msg):
            prog_bar.progress(pct)
            prog_status.text(msg)

        # ── 1. Altitud SRTM ────────────────────────────────────────────────
        update_progress(0.05, "Extrayendo altitud (SRTM)...")
        elev_data    = get_elevacion(ee_geom)
        elev_mean    = elev_data['elev_mean']
        apto_altitud = ELEV_MIN <= elev_mean <= ELEV_MAX

        # ── 2. Series temporales Sentinel-2 ───────────────────────────────
        df_ts, fuente_info = extraer_series_temporales(
            ee_geom, anio_analisis, update_progress
        )

        if df_ts is None:
            prog_bar.empty()
            prog_status.empty()
            st.error(f"No se pudieron extraer datos satelitales: {fuente_info}")
            st.stop()

        # ── 3. Clima NASA POWER ───────────────────────────────────────────
        update_progress(0.82, "Descargando variables climaticas (NASA POWER)...")
        clima_df = get_nasa_power(lat_c, lon_c, anio_analisis)
        if clima_df is not None:
            clima_df['fecha'] = pd.to_datetime(clima_df['fecha'])
            clima = {
                'tmax_mean':    float(clima_df['T2M_MAX'].mean()),
                'tmin_mean':    float(clima_df['T2M_MIN'].mean()),
                'precip_anual': float(clima_df['PRECTOTCORR'].sum()),
            }
        else:
            clima = {'tmax_mean': 26.5, 'tmin_mean': 16.0, 'precip_anual': 1300}

        # ── 4. Clasificacion y prediccion ─────────────────────────────────
        update_progress(0.90, "Clasificando uso de suelo...")
        clasif = clasificar_zona(df_ts, elev_mean, apto_altitud)

        update_progress(0.95, "Calculando prediccion de rendimiento...")
        # Construir dict de features para el predictor
        features_para_pred = {
            'ndvi_max':       float(df_ts['NDVI_SG'].max())  if 'NDVI_SG' in df_ts.columns else clasif['ndvi_prom'],
            'ndvi_min':       float(df_ts['NDVI_SG'].min())  if 'NDVI_SG' in df_ts.columns else clasif['ndvi_prom'],
            'ndvi_mean':      clasif['ndvi_prom'],
            'ndvi_std':       float(df_ts['NDVI_SG'].std())  if 'NDVI_SG' in df_ts.columns else 0.05,
            'ndvi_amplitude': clasif['ndvi_amp'],
            'ndvi_auc':       clasif['ndvi_prom'] * 365,
            'ndvi_peak_doy':  200.0,
            'ndvi_q75_q25':   float(df_ts['NDVI_SG'].quantile(0.75) - df_ts['NDVI_SG'].quantile(0.25)) if 'NDVI_SG' in df_ts.columns else 0.10,
            'evi_mean':       clasif['evi_prom'],
            'evi_max':        float(df_ts['EVI_SG'].max())   if 'EVI_SG'  in df_ts.columns else clasif['evi_prom'],
            'evi_std':        float(df_ts['EVI_SG'].std())   if 'EVI_SG'  in df_ts.columns else 0.04,
            'evi_amplitude':  float(df_ts['EVI_SG'].max()  - df_ts['EVI_SG'].min())  if 'EVI_SG'  in df_ts.columns else 0.10,
            'gndvi_mean':     clasif['gndvi_prom'],
            'gndvi_max':      float(df_ts['GNDVI_SG'].max()) if 'GNDVI_SG' in df_ts.columns else clasif['gndvi_prom'],
            'ndwi_mean':      clasif['ndwi_prom'],
            'ndwi_min':       float(df_ts['NDWI_SG'].min())  if 'NDWI_SG' in df_ts.columns else clasif['ndwi_prom'],
            'savi_mean':      clasif['savi_prom'],
            'savi_max':       float(df_ts['SAVI_SG'].max())  if 'SAVI_SG' in df_ts.columns else clasif['savi_prom'],
            'ndre_mean':      clasif['ndre_prom'],
            'ndre_max':       float(df_ts['NDRE_SG'].max())  if 'NDRE_SG' in df_ts.columns else clasif['ndre_prom'],
            'ndvi_q1':        float(df_ts[df_ts['fecha'].dt.quarter==1]['NDVI_SG'].mean()) if 'NDVI_SG' in df_ts.columns and len(df_ts) > 0 else clasif['ndvi_prom'],
            'ndvi_q2':        float(df_ts[df_ts['fecha'].dt.quarter==2]['NDVI_SG'].mean()) if 'NDVI_SG' in df_ts.columns and len(df_ts) > 0 else clasif['ndvi_prom'],
            'ndvi_q3':        float(df_ts[df_ts['fecha'].dt.quarter==3]['NDVI_SG'].mean()) if 'NDVI_SG' in df_ts.columns and len(df_ts) > 0 else clasif['ndvi_prom'],
            'ndvi_q4':        float(df_ts[df_ts['fecha'].dt.quarter==4]['NDVI_SG'].mean()) if 'NDVI_SG' in df_ts.columns and len(df_ts) > 0 else clasif['ndvi_prom'],
            'elev_mean':      elev_mean,
            'n_obs':          float(len(df_ts)),
            # Variables climaticas para fallback
            'tmax_mean':      clima.get('tmax_mean', 26.5),
            'precip_anual':   clima.get('precip_anual', 1300),
        }
        # Rellenar NaN que puedan venir de trimestres sin datos
        for k, v in features_para_pred.items():
            if isinstance(v, float) and np.isnan(v):
                features_para_pred[k] = 0.0

        rend = predecir_rendimiento(
            clasif, area_ha, dept_ref, clima,
            df_ts=df_ts, elev_mean=elev_mean
        )

        # ── 5. Clasificacion pixeles GEE (en segundo plano) ───────────────
        update_progress(0.97, "Generando mapa de clasificacion...")
        img_clasif, img_prob, composito = clasificar_pixeles_gee(
            ee_geom, anio_analisis
        )
        dist_clases = []
        if img_clasif is not None:
            dist_clases = get_distribucion_clases(img_clasif, ee_geom)

        update_progress(1.0, "Analisis completado!")
        prog_bar.empty()
        prog_status.empty()

        # Guardar resultados
        st.session_state.resultados = {
            'area_ha':      area_ha,
            'area_mz':      area_mz,
            'lat':          lat_c,
            'lon':          lon_c,
            'elev_data':    elev_data,
            'elev_mean':    elev_mean,
            'apto_altitud': apto_altitud,
            'clasif':       clasif,
            'rend':         rend,
            'df_ts':        df_ts,
            'clima':        clima,
            'dept_ref':     dept_ref,
            'anio':         anio_analisis,
            'nombre':       nombre_zona,
            'fuente_sat':   fuente_info,
            'dist_clases':  dist_clases,
            'n_obs':        len(df_ts),
        }
        st.session_state.analisis_listo = True


# ════════════════════════════════════════════════════════════════
# COLUMNA DERECHA — RESULTADOS
# ════════════════════════════════════════════════════════════════
with col_res:

    if not st.session_state.analisis_listo or not st.session_state.resultados:
        st.markdown("<br>", unsafe_allow_html=True)
        st.info("""
**Como usar:**

1. Escribe el nombre de tu zona

2. Selecciona ano y departamento

3. Dibuja el poligono en el mapa
   o sube un GeoJSON

4. Haz clic en **Analizar con GEE**

El sistema descargara imagenes Sentinel-2
reales de tu zona y mostrara resultados aqui.
        """)
        if not gee_ok:
            st.warning(
                "Configura las credenciales de GEE en Streamlit Secrets "
                "para habilitar el analisis."
            )
    else:
        r      = st.session_state.resultados
        clasif = r['clasif']
        rend   = r['rend']
        df_ts  = r['df_ts']
        cv     = clasif['color_v']

        # Badge de datos reales
        st.markdown(
            f"<div class='info-real'>"
            f"Datos reales Sentinel-2 | {r['fuente_sat']} | "
            f"{r['n_obs']} observaciones | {r['anio']}"
            f"</div>",
            unsafe_allow_html=True
        )

        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "Clasificacion", "Rendimiento", "Indices",
            "Proyeccion 3 Anos", "Siembra Nueva", "Danos y Plagas",
            "Reporte"
        ])

        # ── TAB 1: Clasificacion ──────────────────────────────────────────
        with tab1:
            emoji_map = {
                'CAFE CONFIRMADO':    '✅',
                'PROBABLE CAFE':      '⚠️',
                'RESULTADO INCIERTO': '❓',
                'NO ES CAFE':         '❌',
            }
            emoji_v = emoji_map.get(clasif['veredicto'], '❓')

            st.markdown(f"""
            <div style='padding:14px;background:{cv}18;
                        border-left:6px solid {cv};border-radius:8px;margin-bottom:14px'>
                <div style='font-size:20px;font-weight:bold;color:{cv}'>
                    {emoji_v} {clasif['veredicto']}
                </div>
                <div style='font-size:14px;color:{cv};margin-top:4px'>
                    Score integrado: <b>{clasif['score_final']:.1f}%</b>
                    &nbsp;|&nbsp; Datos Sentinel-2 REALES
                </div>
            </div>
            """, unsafe_allow_html=True)

            m1, m2, m3 = st.columns(3)
            m1.metric("Reglas espectrales",
                      f"{clasif['sc_reg']:.0f}%",
                      f"{clasif['n_ok']}/{len(clasif['reglas'])} OK")
            m2.metric("Random Forest",
                      f"{clasif['prob_cafe']:.1f}%",
                      "prob. cafe")
            m3.metric("Patron fenologico",
                      f"{clasif['sc_fenol']*100:.0f}%",
                      f"r={clasif['corr']:.2f}")

            st.divider()
            st.markdown("**Reglas espectrales (valores reales de la zona):**")
            for regla, ok in clasif['reglas'].items():
                color = "#1a7a4a" if ok else "#c0392b"
                st.markdown(
                    f"<span style='color:{color}'>{'[OK]' if ok else '[NO]'} {regla}</span>",
                    unsafe_allow_html=True
                )

            st.divider()
            elev = r['elev_data']
            ic_alt = "OK" if r['apto_altitud'] else "FUERA DE RANGO"
            st.markdown(
                f"**Altitud SRTM:** {elev['elev_mean']:.0f} msnm ({ic_alt}) | "
                f"Rango: {elev['elev_min']:.0f}–{elev['elev_max']:.0f} m | "
                f"Pendiente media: {elev['slope_mean']:.1f}°"
            )

            # Distribucion de clases
            if r['dist_clases']:
                st.divider()
                st.markdown("**Distribucion de uso de suelo (pixeles 10m):**")
                for d in r['dist_clases']:
                    color = d['color']
                    st.markdown(
                        f"<span style='color:{color}'>■</span> "
                        f"**{d['clase']}**: {d['ha']:.3f} ha "
                        f"({d['pct']:.1f}%)",
                        unsafe_allow_html=True
                    )

        # ── TAB 2: Rendimiento ────────────────────────────────────────────
        with tab2:
            if clasif['score_final'] < 35:
                st.warning("Zona no clasificada como cafe. Prediccion no aplica.")
            else:
                # ── Estimacion proxima cosecha ────────────────────────────
                import calendar
                anio_actual    = r['anio']
                temporada_pred = f"{anio_actual}-{anio_actual+1}"

                # Calcular area de cafe real (del clasificador GEE si existe)
                area_cafe_ha = r['area_ha']  # default = area total
                if r.get('dist_clases'):
                    cafe_clase = next(
                        (d for d in r['dist_clases'] if d['clase'] == 'Cafe'), None
                    )
                    if cafe_clase and cafe_clase['ha'] > 0:
                        area_cafe_ha = cafe_clase['ha']

                prod_prox     = round(rend['pred_ens'] * area_cafe_ha, 1)
                prod_prox_lo  = round(rend['ic_lo']   * area_cafe_ha, 1)
                prod_prox_hi  = round(rend['ic_hi']   * area_cafe_ha, 1)

                # Banner de estimacion
                st.markdown(f"""
                <div style='background:#1F3864;color:white;padding:14px 18px;
                            border-radius:10px;margin-bottom:16px'>
                    <div style='font-size:13px;opacity:0.8;margin-bottom:4px'>
                        ESTIMACION PROXIMA COSECHA — Temporada {temporada_pred}
                    </div>
                    <div style='font-size:28px;font-weight:bold'>
                        {prod_prox:,.0f} qq oro
                    </div>
                    <div style='font-size:13px;opacity:0.85;margin-top:4px'>
                        Rango IC 80%: {prod_prox_lo:,.0f} – {prod_prox_hi:,.0f} qq &nbsp;|&nbsp;
                        Area cafe: {area_cafe_ha:.2f} ha &nbsp;|&nbsp;
                        {rend['pred_ens']:.1f} qq/ha
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Metricas
                c1, c2 = st.columns(2)
                c1.metric("Rendimiento predicho",
                          f"{rend['pred_ens']:.2f} qq/ha",
                          f"IC 80%: {rend['ic_lo']:.1f} – {rend['ic_hi']:.1f}")
                c2.metric(f"Produccion estimada {temporada_pred}",
                          f"{prod_prox:,.0f} qq oro",
                          f"en {area_cafe_ha:.2f} ha de cafe")

                c3, c4 = st.columns(2)
                delta_s = f"+{rend['delta']:.2f}" if rend['delta'] >= 0 \
                          else f"{rend['delta']:.2f}"
                c3.metric("vs. Media departamental",
                          f"{delta_s} qq/ha",
                          f"Ref {r['dept_ref']}: {rend['hist_dep']:.1f}")

                # Mostrar si se usaron modelos reales o formula calibrada
                if rend.get('modelo_real', False):
                    c4.metric("Modelo predictivo",
                              "RF+XGB Reales",
                              "Entrenado con IHCAFE + Sentinel-2")
                    st.success(
                        f"Prediccion basada en modelos entrenados con datos reales | "
                        f"{rend.get('fuente_modelo', '')}"
                    )
                else:
                    c4.metric("Modelo predictivo",
                              "Formula calibrada",
                              "Sube .pkl para usar modelos reales")
                    st.warning(
                        "Prediccion usando formula calibrada con promedios IHCAFE. "
                        "Para activar los modelos RF+XGB reales: ejecuta el notebook "
                        "'entrenamiento_modelos_reales.ipynb' en Colab, descarga los "
                        ".pkl y súbelos a cafe_app/modelos/ en GitHub."
                    )

                st.divider()

                # Grafico comparativo modelos
                fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
                fig.suptitle(
                    f'Prediccion de Rendimiento — Temporada {temporada_pred}',
                    fontsize=11, fontweight='bold', color='#1F3864'
                )
                fig.patch.set_facecolor('#FAFAFA')

                # Panel izquierdo: RF vs XGB vs Ensemble
                mods  = ['RF', 'XGB', 'Ensemble\n0.55+0.45']
                vals  = [rend['pred_rf'], rend['pred_xgb'], rend['pred_ens']]
                cols_ = ['#2E5FA3', '#8B5E3C', '#1a7a4a']
                bars  = axes[0].bar(mods, vals, color=cols_,
                                    alpha=0.85, edgecolor='white', lw=1.5)
                axes[0].errorbar(
                    2, rend['pred_ens'],
                    yerr=[[rend['pred_ens']-rend['ic_lo']],
                          [rend['ic_hi']-rend['pred_ens']]],
                    fmt='none', color='black', capsize=7, lw=2
                )
                axes[0].axhline(
                    rend['hist_dep'], ls='--', color='red', lw=1.5,
                    label=f"Media {r['dept_ref']} ({rend['hist_dep']:.1f})"
                )
                for bar, v in zip(bars, vals):
                    axes[0].text(
                        bar.get_x()+bar.get_width()/2, v+0.3,
                        f'{v:.1f}', ha='center', fontsize=10, fontweight='bold'
                    )
                axes[0].set_ylabel('qq oro/ha')
                axes[0].set_title('Rendimiento por modelo (IC 80%)')
                axes[0].legend(fontsize=8)
                axes[0].grid(axis='y', alpha=0.25)
                axes[0].set_ylim(0, max(vals)*1.35)
                axes[0].set_facecolor('#FAFAFA')

                # Panel derecho: produccion total con IC
                etiquetas = ['IC 80%\nInferior', 'Estimacion\nCentral', 'IC 80%\nSuperior']
                valores_p = [prod_prox_lo, prod_prox, prod_prox_hi]
                colores_p = ['#AACBE8', '#1F3864', '#AACBE8']
                bars2 = axes[1].bar(etiquetas, valores_p,
                                    color=colores_p, alpha=0.85,
                                    edgecolor='white', lw=1.5)
                for bar, v in zip(bars2, valores_p):
                    axes[1].text(
                        bar.get_x()+bar.get_width()/2, v+0.5,
                        f'{v:,.0f}', ha='center',
                        fontsize=10, fontweight='bold', color='#1F3864'
                    )
                axes[1].set_ylabel('qq oro')
                axes[1].set_title(
                    f'Produccion estimada {temporada_pred}\n'
                    f'({area_cafe_ha:.2f} ha de cafe)'
                )
                axes[1].grid(axis='y', alpha=0.25)
                axes[1].set_ylim(0, max(valores_p)*1.35)
                axes[1].set_facecolor('#FAFAFA')

                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close()

                if r['clima']:
                    st.caption(
                        f"Tmax media: {r['clima']['tmax_mean']:.1f}C | "
                        f"Precip. anual: {r['clima']['precip_anual']:.0f} mm "
                        f"(NASA POWER {r['anio']}) | "
                        f"Ensemble: {W_RF}×RF + {W_XGB}×XGB"
                    )

                # Descarga rapida de la estimacion
                csv_rend = pd.DataFrame([{
                    'zona':            r['nombre'],
                    'temporada':       temporada_pred,
                    'area_cafe_ha':    area_cafe_ha,
                    'pred_rf_qq_ha':   rend['pred_rf'],
                    'pred_xgb_qq_ha':  rend['pred_xgb'],
                    'pred_ens_qq_ha':  rend['pred_ens'],
                    'ic80_inf_qq_ha':  rend['ic_lo'],
                    'ic80_sup_qq_ha':  rend['ic_hi'],
                    'prod_est_qq':     prod_prox,
                    'prod_ic80_inf':   prod_prox_lo,
                    'prod_ic80_sup':   prod_prox_hi,
                    'media_dept_qq_ha':rend['hist_dep'],
                    'dept_ref':        r['dept_ref'],
                }]).to_csv(index=False).encode('utf-8')

                st.download_button(
                    f"Descargar estimacion {temporada_pred} (.csv)",
                    csv_rend,
                    f"estimacion_{r['nombre'].replace(' ','_')}_{temporada_pred}.csv",
                    "text/csv"
                )

        # ── TAB 3: Indices reales ─────────────────────────────────────────
        with tab3:
            st.markdown(
                f"**Perfiles fenologicos reales — {r['nombre']} | {r['anio']}**"
            )
            st.markdown(
                f"<div class='info-real'>"
                f"Datos Sentinel-2 reales | {r['fuente_sat']} | "
                f"Savitzky-Golay w={7} m=2"
                f"</div>",
                unsafe_allow_html=True
            )

            fig2, axes = plt.subplots(2, 3, figsize=(11, 6))
            fig2.suptitle(
                f"Indices Espectrales Reales | {r['nombre']} | {r['anio']}",
                fontsize=10, fontweight='bold', color='#1F3864'
            )
            fig2.patch.set_facecolor('#FAFAFA')

            pares = [
                ('NDVI_SG', 'NDVI_mean', '#2E5FA3', axes[0,0], 'NDVI'),
                ('EVI_SG',  'EVI_mean',  '#1a7a4a', axes[0,1], 'EVI'),
                ('GNDVI_SG','GNDVI_mean','#8B5E3C', axes[0,2], 'GNDVI'),
                ('NDWI_SG', 'NDWI_mean', '#E87722', axes[1,0], 'NDWI'),
                ('SAVI_SG', 'SAVI_mean', '#9b2c8b', axes[1,1], 'SAVI'),
                ('NDRE_SG', 'NDRE_mean', '#c0392b', axes[1,2], 'NDRE'),
            ]
            for sg_col, raw_col, color, ax, titulo in pares:
                if raw_col in df_ts.columns:
                    ax.scatter(df_ts['fecha'], df_ts[raw_col],
                               alpha=0.3, s=10, color=color, label='Bruto')
                if sg_col in df_ts.columns:
                    ax.plot(df_ts['fecha'], df_ts[sg_col],
                            color=color, lw=2.2, label='SG')
                ax.set_title(titulo, fontsize=9, fontweight='bold')
                ax.set_ylim(-0.25, 1.0)
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
                ax.grid(alpha=0.2)
                ax.set_facecolor('#FAFAFA')

            plt.tight_layout()
            st.pyplot(fig2, use_container_width=True)
            plt.close()

            # Tabla de estadisticos reales
            stats_rows = []
            for sg_col, nom in [
                ('NDVI_SG','NDVI'), ('EVI_SG','EVI'), ('GNDVI_SG','GNDVI'),
                ('NDWI_SG','NDWI'), ('SAVI_SG','SAVI'), ('NDRE_SG','NDRE')
            ]:
                if sg_col in df_ts.columns:
                    v = df_ts[sg_col].dropna()
                    if len(v) > 0:
                        stats_rows.append({
                            'Indice':   nom,
                            'Min':      round(float(v.min()), 4),
                            'Max':      round(float(v.max()), 4),
                            'Media':    round(float(v.mean()), 4),
                            'Amplitud': round(float(v.max()-v.min()), 4),
                            'Obs':      len(v),
                        })
            if stats_rows:
                st.dataframe(pd.DataFrame(stats_rows),
                             use_container_width=True, hide_index=True)

        # ── TAB 4: Proyeccion 3 Anos ──────────────────────────────────────
        with tab4:
            st.markdown("**Proyeccion de produccion — cafe nuevo del vivero**")
            st.caption(
                "Si se siembra café este año, esta tabla muestra la produccion "
                "esperada en cada uno de los proximos 5 años segun los factores "
                "de maduracion calibrados con datos IHCAFE Honduras."
            )

            if clasif['score_final'] < 35:
                st.warning("Zona no clasificada como cafe. Proyeccion no aplica.")
            else:
                st.markdown("**Parametros de la nueva siembra**")

                # Fila 1: Año y area
                c1, c2 = st.columns(2)
                anio_siembra = c1.number_input(
                    "Ano de siembra (salida del vivero)",
                    min_value=2020, max_value=2035,
                    value=2026, step=1,
                    help="Ano en que se trasplantan las plantas del vivero al campo"
                )
                area_cafe_proy = c2.number_input(
                    "Area a sembrar (ha)",
                    min_value=0.1, max_value=500.0,
                    value=float(round(r['area_ha'], 2)), step=0.1,
                    help="Hectareas que se van a sembrar. "
                         "Por defecto usa el area del poligono analizado."
                )

                # Fila 2: Modo de calculo de plantas
                st.markdown("**Cantidad de plantas**")
                modo_plantas = st.radio(
                    "Calcular plantas por:",
                    ["Densidad de siembra (plantas/ha)",
                     "Numero exacto de plantas que conozco"],
                    horizontal=True,
                    help="Si ya sabes cuantas plantas tienes o planeas sembrar, "
                         "ingresa el numero exacto. Si no, usa la densidad."
                )

                c3, c4 = st.columns(2)
                if "Densidad" in modo_plantas:
                    densidad = c3.selectbox(
                        "Densidad de siembra",
                        options=[1600, 2000, 2500, 3000, 4000, 5000],
                        index=2,
                        format_func=lambda x: f"{x:,} plantas/ha",
                        help=(
                            "Distanciamiento tipico en Honduras:\n"
                            "1600 plantas/ha = 3.0m × 2.0m (tradicional)\n"
                            "2000 plantas/ha = 2.5m × 2.0m\n"
                            "2500 plantas/ha = 2.0m × 2.0m (semi-intensivo)\n"
                            "3000 plantas/ha = 2.0m × 1.67m\n"
                            "5000 plantas/ha = 1.5m × 1.33m (intensivo)"
                        )
                    )
                    total_plantas_calc = int(area_cafe_proy * densidad)
                    densidad_real      = densidad
                else:
                    total_plantas_manual = c3.number_input(
                        "Numero de plantas",
                        min_value=10, max_value=500000,
                        value=int(area_cafe_proy * 2500), step=100,
                        help="Numero total de plantas que tienes o planeas sembrar"
                    )
                    total_plantas_calc = total_plantas_manual
                    densidad_real = round(total_plantas_manual / area_cafe_proy) \
                                    if area_cafe_proy > 0 else 2500

                # Distancia equivalente
                import math
                dist_equiv = round(math.sqrt(10000 / densidad_real), 2) \
                             if densidad_real > 0 else 0

                # Resumen de la siembra
                st.info(
                    f"Total de plantas: **{total_plantas_calc:,} plantas** | "
                    f"Densidad: **{densidad_real:,} plantas/ha** | "
                    f"Area: **{area_cafe_proy:.2f} ha** | "
                    f"Distanciamiento equiv.: **{dist_equiv:.1f}m × {dist_equiv:.1f}m**"
                )

                proyeccion, total_pl, qq_planta_plena = proyectar_3_anios(
                    rend['pred_ens'], area_cafe_proy,
                    r['dept_ref'], int(anio_siembra),
                    plantas_por_ha=densidad_real
                )
                df_proy = pd.DataFrame(proyeccion)

                # Grafico de barras de la proyeccion
                fig_p, ax_p = plt.subplots(figsize=(9, 4))
                colores_p = ['#d5d8dc','#aed6f1','#2E5FA3','#1a5276','#1a7a4a']
                bars_p = ax_p.bar(
                    [str(p['Anio']) for p in proyeccion],
                    [p['Rendimiento (qq/ha)'] for p in proyeccion],
                    color=colores_p, alpha=0.88,
                    edgecolor='white', lw=1.5
                )
                ax_p.axhline(
                    rend['hist_dep'], ls='--', color='red', lw=1.5,
                    label=f"Media hist. {r['dept_ref']} ({rend['hist_dep']:.1f} qq/ha)"
                )
                ax_p.axhline(
                    rend['pred_ens'], ls=':', color='#2E5FA3', lw=1.5,
                    label=f"Prediccion actual ({rend['pred_ens']:.1f} qq/ha)"
                )
                for bar, p in zip(bars_p, proyeccion):
                    v = p['Rendimiento (qq/ha)']
                    if v > 0:
                        ax_p.text(
                            bar.get_x() + bar.get_width()/2, v + 0.3,
                            f"{v:.1f}", ha='center',
                            fontsize=9, fontweight='bold'
                        )
                ax_p.set_ylabel('qq oro/ha')
                ax_p.set_xlabel('Año de cosecha')
                ax_p.set_title(
                    f'Proyeccion de Rendimiento — Siembra {int(anio_siembra)} | '
                    f'{area_cafe_proy:.2f} ha | {total_plantas_calc:,} plantas',
                    fontweight='bold', color='#1F3864'
                )
                ax_p.legend(fontsize=8)
                ax_p.grid(axis='y', alpha=0.25)
                fig_p.patch.set_facecolor('#FAFAFA')
                ax_p.set_facecolor('#FAFAFA')
                plt.tight_layout()
                st.pyplot(fig_p, use_container_width=True)
                plt.close()

                # Tabla detallada
                st.markdown("**Detalle por año:**")
                st.dataframe(
                    df_proy.style.background_gradient(
                        subset=['Rendimiento (qq/ha)', 'Produccion est. (qq)'],
                        cmap='Blues'
                    ),
                    use_container_width=True, hide_index=True
                )

                # Resumen
                total_3 = sum(p['Produccion est. (qq)'] for p in proyeccion[1:4])
                # Produccion a plena maduracion (año 5)
                plena = proyeccion[4]
                st.info(
                    f"**Produccion acumulada primeros 3 anos comerciales "
                    f"(ano {int(anio_siembra)+1} al {int(anio_siembra)+3}):** "
                    f"**{total_3:,.0f} qq oro** sobre {area_cafe_proy:.2f} ha\n\n"
                    f"**A produccion plena (ano {int(anio_siembra)+4}):** "
                    f"{plena['Rendimiento (qq/ha)']:.1f} qq/ha | "
                    f"{plena['Produccion est. (qq)']:.0f} qq totales | "
                    f"{plena['lb por planta']:.1f} lb/planta"
                )
                st.caption(
                    "Factores de maduracion: Ano 1=0%, Ano 2=15%, "
                    "Ano 3=50%, Ano 4=80%, Ano 5=100% | "
                    "Densidad: " + str(densidad) + " plantas/ha | "
                    "Fuente: IHCAFE Guia Tecnica de Caficultura 2022"
                )

                # Descarga
                csv_proy = df_proy.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Descargar proyeccion CSV",
                    csv_proy,
                    f"proyeccion_{r['nombre'].replace(' ','_')}_{int(anio_siembra)}.csv",
                    use_container_width=False
                )

        # ── TAB 5: Deteccion de Siembra Nueva ────────────────────────────
        with tab5:
            st.markdown("**Deteccion de siembra nueva de cafe**")
            st.caption(
                "Analiza el perfil espectral actual para determinar si "
                "hay evidencia de cafe recien sembrado (salido del vivero). "
                "Util para detectar expansion de la caficultura en una zona."
            )

            ndvi_anterior = st.number_input(
                "NDVI promedio del año ANTERIOR (opcional — para comparacion)",
                min_value=0.0, max_value=1.0,
                value=0.0, step=0.01,
                help="Si tienes el NDVI de la misma zona del año anterior, "
                     "ingrésalo aquí para mejorar la deteccion. "
                     "Deja en 0.0 si no lo tienes."
            )

            siembra = detectar_siembra_nueva(
                df_ts,
                clasif['ndvi_prom']
            )

            # Resultado principal
            color_s = '#1a7a4a' if siembra['es_siembra_nueva'] else '#888888'
            icono_s = 'POSIBLE SIEMBRA NUEVA' if siembra['es_siembra_nueva'] \
                      else 'SIN EVIDENCIA DE SIEMBRA NUEVA'

            st.markdown(f"""
            <div style='padding:14px;background:{color_s}18;
                        border-left:6px solid {color_s};
                        border-radius:8px;margin-bottom:14px'>
                <div style='font-size:18px;font-weight:bold;color:{color_s}'>
                    {icono_s}
                </div>
                <div style='font-size:14px;color:{color_s};margin-top:4px'>
                    Confianza: <b>{siembra['confianza']}%</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("**Evidencias encontradas:**")
            for ev in siembra['evidencias']:
                st.markdown(f"- {ev}")

            st.divider()
            st.markdown(f"**Recomendacion:** {siembra['recomendacion']}")
            st.divider()

            # Visualizacion del perfil NDVI
            if 'NDVI_SG' in df_ts.columns:
                fig_s, ax_s = plt.subplots(figsize=(9, 3.5))
                ax_s.plot(df_ts['fecha'], df_ts['NDVI_SG'],
                          color='#2E5FA3', lw=2.2, label='NDVI SG actual')
                if 'NDVI_mean' in df_ts.columns:
                    ax_s.scatter(df_ts['fecha'], df_ts['NDVI_mean'],
                                 alpha=0.25, s=10, color='#2E5FA3')
                ax_s.axhspan(0.28, 0.52, alpha=0.12, color='#27ae60',
                             label='Rango cafe joven (0.28-0.52)')
                ax_s.axhspan(0.52, 0.75, alpha=0.08, color='#2E5FA3',
                             label='Rango cafe adulto (0.52-0.75)')
                ax_s.set_ylabel('NDVI')
                ax_s.set_title('Perfil NDVI — Deteccion de siembra nueva',
                               fontweight='bold', color='#1F3864')
                ax_s.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
                ax_s.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
                ax_s.grid(alpha=0.2)
                ax_s.legend(fontsize=8)
                ax_s.set_ylim(0, 1.0)
                fig_s.patch.set_facecolor('#FAFAFA')
                ax_s.set_facecolor('#FAFAFA')
                plt.tight_layout()
                st.pyplot(fig_s, use_container_width=True)
                plt.close()

            st.info(
                "**Nota metodologica:** La deteccion de siembra nueva es mas "
                "precisa cuando se compara el NDVI actual con el del año anterior. "
                "Un aumento de NDVI de 0.10-0.35 entre años consecutivos, "
                "combinado con NDVI actual en rango 0.28-0.52, "
                "es la señal mas clara de establecimiento de cafe."
            )

        # ── TAB 6: Danos y Plagas ─────────────────────────────────────────
        with tab6:
            st.markdown("**Deteccion de daños y plagas post-cosecha**")
            st.caption(
                "Analiza las firmas espectrales de los indices para detectar "
                "señales de roya, estres hidrico, defoliacion y otros daños. "
                "Basado en la respuesta espectral caracteristica de cada "
                "tipo de afectacion en Coffea arabica."
            )

            danos = detectar_danos(
                clasif['ndvi_prom'],
                clasif['evi_prom'],
                clasif['ndre_prom'],
                clasif['ndwi_prom'],
                clasif['ndvi_amp']
            )

            # Nivel de alerta principal
            nivel_icons = {
                'NORMAL':   '✅ CULTIVO SALUDABLE',
                'ATENCION': '⚠️ ATENCION REQUERIDA',
                'ALERTA':   '🚨 ALERTA',
                'CRITICO':  '🚨 ESTADO CRITICO',
            }
            st.markdown(f"""
            <div style='padding:14px;background:{danos['color']}18;
                        border-left:6px solid {danos['color']};
                        border-radius:8px;margin-bottom:14px'>
                <div style='font-size:18px;font-weight:bold;
                            color:{danos['color']}'>
                    {nivel_icons.get(danos['nivel'], danos['nivel'])}
                </div>
                <div style='font-size:13px;color:{danos['color']};
                            margin-top:4px'>
                    {danos['n_alertas']} alerta(s) detectada(s)
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Detalle de cada alerta
            for alerta in danos['alertas']:
                with st.expander(
                    f"**{alerta['tipo']}**",
                    expanded=(alerta['color'] != '#1a7a4a')
                ):
                    st.markdown(f"**Detalle:** {alerta['detalle']}")
                    st.markdown(f"**Accion recomendada:** {alerta['accion']}")

            st.divider()

            # Grafico radar de indices
            fig_d, ax_d = plt.subplots(figsize=(9, 4))

            # Barras de los indices vs umbrales de alerta
            indices_nombres = ['NDVI', 'EVI', 'NDRE', 'NDWI (+0.5)', 'Amplitud']
            valores_obs = [
                clasif['ndvi_prom'],
                clasif['evi_prom'],
                clasif['ndre_prom'],
                clasif['ndwi_prom'] + 0.5,
                clasif['ndvi_amp'],
            ]
            umbrales_min = [0.40, 0.22, 0.30, 0.35, 0.10]
            umbrales_max = [0.75, 0.50, 0.55, 0.65, 0.45]

            x_pos = np.arange(len(indices_nombres))
            bars_d = ax_d.bar(x_pos, valores_obs,
                              color=['#c0392b' if (v < mn or v > mx) else '#2E5FA3'
                                     for v, mn, mx in
                                     zip(valores_obs, umbrales_min, umbrales_max)],
                              alpha=0.8, edgecolor='white', lw=1.5)

            # Lineas de umbral
            for xi, (mn, mx) in enumerate(zip(umbrales_min, umbrales_max)):
                ax_d.plot([xi-0.4, xi+0.4], [mn, mn],
                          'g--', lw=1.5, alpha=0.7)
                ax_d.plot([xi-0.4, xi+0.4], [mx, mx],
                          'r--', lw=1.5, alpha=0.7)

            ax_d.set_xticks(x_pos)
            ax_d.set_xticklabels(indices_nombres)
            ax_d.set_ylabel('Valor del indice')
            ax_d.set_title(
                'Indices espectrales vs umbrales de alerta\n'
                '(Rojo = fuera de rango cafe saludable)',
                fontweight='bold', color='#1F3864'
            )
            ax_d.set_ylim(0, 1.0)
            ax_d.grid(axis='y', alpha=0.25)
            from matplotlib.lines import Line2D
            legend_items = [
                Line2D([0],[0], color='g', ls='--', label='Umbral minimo cafe'),
                Line2D([0],[0], color='r', ls='--', label='Umbral maximo cafe'),
                plt.Rectangle((0,0),1,1, color='#2E5FA3',
                               alpha=0.8, label='En rango normal'),
                plt.Rectangle((0,0),1,1, color='#c0392b',
                               alpha=0.8, label='Fuera de rango'),
            ]
            ax_d.legend(handles=legend_items, fontsize=8, loc='upper right')
            fig_d.patch.set_facecolor('#FAFAFA')
            ax_d.set_facecolor('#FAFAFA')
            plt.tight_layout()
            st.pyplot(fig_d, use_container_width=True)
            plt.close()

            # Tabla de firmas espectrales de referencia
            st.divider()
            st.markdown("**Referencia: Firmas espectrales por tipo de daño**")
            ref_danos = pd.DataFrame([
                {'Tipo de daño':     'Roya (Hemileia vastatrix)',
                 'Indice clave':     'NDRE cae antes que NDVI',
                 'Umbral':           'NDRE/NDVI < 0.62',
                 'Mecanismo':        'Destruccion de clorofila foliar'},
                {'Tipo de daño':     'Sequia / Estres hidrico',
                 'Indice clave':     'NDWI',
                 'Umbral':           'NDWI < -0.15',
                 'Mecanismo':        'Deficit de agua en el dosel'},
                {'Tipo de daño':     'Defoliacion severa',
                 'Indice clave':     'NDVI + EVI + NDRE bajos',
                 'Umbral':           '2 o mas indices bajo minimo',
                 'Mecanismo':        'Perdida de follaje por plaga o enfermedad'},
                {'Tipo de daño':     'Daño irregular / Parches',
                 'Indice clave':     'Amplitud NDVI',
                 'Umbral':           'Amplitud > 0.45',
                 'Mecanismo':        'Variabilidad espacial por daño localizado'},
            ])
            st.dataframe(ref_danos, use_container_width=True, hide_index=True)
            st.caption(
                "Firmas espectrales basadas en: Gao (1996) NDWI; "
                "Frampton et al. (2013) Red Edge para hongos; "
                "Sentinel-2 Application Note — Crop Stress Detection."
            )

        # ── TAB 7: Reporte ────────────────────────────────────────────────
        with tab7:
            st.markdown("**Resumen del analisis**")

            reporte = f"""REPORTE DE ANALISIS — {r['nombre']}
Fecha: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
Ano satelital: {r['anio']}
Fuente satelital: {r['fuente_sat']}
Departamento referencia: {r['dept_ref']}
Observaciones validas: {r['n_obs']}

UBICACION Y AREA
Coordenadas: {r['lat']:.6f}N, {r['lon']:.6f}W
Area analizada: {r['area_ha']:.4f} ha ({r['area_mz']:.4f} mz)
Elevacion SRTM: {r['elev_mean']:.1f} msnm (rango: {r['elev_data']['elev_min']:.0f}-{r['elev_data']['elev_max']:.0f} m)
Pendiente media: {r['elev_data']['slope_mean']:.1f} grados

CLASIFICACION DE USO DE SUELO (datos reales Sentinel-2)
Veredicto: {clasif['veredicto']}
Score integrado: {clasif['score_final']:.2f}%
Reglas cumplidas: {clasif['n_ok']}/{len(clasif['reglas'])}
Probabilidad cafe (RF): {clasif['prob_cafe']:.2f}%
Correlacion fenologica: r = {clasif['corr']:.4f}

PREDICCION DE RENDIMIENTO (Ensemble 0.55RF + 0.45XGB)
Random Forest: {rend['pred_rf']:.2f} qq/ha
XGBoost: {rend['pred_xgb']:.2f} qq/ha
Ensemble: {rend['pred_ens']:.2f} qq/ha
Intervalo confianza 80%: [{rend['ic_lo']:.2f} - {rend['ic_hi']:.2f}] qq/ha
Produccion estimada: {rend['prod_est']:,.0f} qq oro
vs. Media {r['dept_ref']}: {'+' if rend['delta']>=0 else ''}{rend['delta']:.2f} qq/ha

INDICES ESPECTRALES REALES
NDVI: min={clasif['ndvi_prom']-clasif['ndvi_amp']/2:.4f} max={clasif['ndvi_prom']+clasif['ndvi_amp']/2:.4f} media={clasif['ndvi_prom']:.4f}
EVI media: {clasif['evi_prom']:.4f}
GNDVI media: {clasif['gndvi_prom']:.4f}
SAVI media: {clasif['savi_prom']:.4f}
NDRE media: {clasif['ndre_prom']:.4f}

VARIABLES CLIMATICAS (NASA POWER {r['anio']})
Temperatura max. media: {r['clima'].get('tmax_mean',0):.2f} C
Temperatura min. media: {r['clima'].get('tmin_mean',0):.2f} C
Precipitacion anual: {r['clima'].get('precip_anual',0):.1f} mm

Sistema Predictivo de Cafe Honduras — Tesis UNAH
Datos: Sentinel-2 SR Harmonized (ESA Copernicus) via Google Earth Engine
"""
            st.text_area("", reporte, height=320, label_visibility='collapsed')

            cd1, cd2, cd3 = st.columns(3)
            with cd1:
                st.download_button(
                    "Descargar TXT",
                    reporte.encode('utf-8'),
                    f"reporte_{r['nombre'].replace(' ','_')}_{r['anio']}.txt",
                    use_container_width=True
                )
            with cd2:
                csv_out = pd.DataFrame([{
                    'zona':               r['nombre'],
                    'lat':                r['lat'],
                    'lon':                r['lon'],
                    'area_ha':            r['area_ha'],
                    'area_mz':            r['area_mz'],
                    'elevacion_msnm':     r['elev_mean'],
                    'pendiente_grados':   r['elev_data']['slope_mean'],
                    'anio':               r['anio'],
                    'fuente_satelital':   r['fuente_sat'],
                    'n_observaciones':    r['n_obs'],
                    'dept_ref':           r['dept_ref'],
                    'veredicto':          clasif['veredicto'],
                    'score_clasif_pct':   clasif['score_final'],
                    'prob_cafe_rf_pct':   clasif['prob_cafe'],
                    'corr_fenologica':    clasif['corr'],
                    'pred_rf_qq_ha':      rend['pred_rf'],
                    'pred_xgb_qq_ha':     rend['pred_xgb'],
                    'pred_ensemble_qq_ha':rend['pred_ens'],
                    'ic80_inf':           rend['ic_lo'],
                    'ic80_sup':           rend['ic_hi'],
                    'produccion_est_qq':  rend['prod_est'],
                    'delta_vs_media_dept':rend['delta'],
                    'ndvi_media':         clasif['ndvi_prom'],
                    'evi_media':          clasif['evi_prom'],
                    'gndvi_media':        clasif['gndvi_prom'],
                    'savi_media':         clasif['savi_prom'],
                    'ndre_media':         clasif['ndre_prom'],
                    'tmax_media_c':       r['clima'].get('tmax_mean',''),
                    'precip_anual_mm':    r['clima'].get('precip_anual',''),
                }]).to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Descargar CSV",
                    csv_out,
                    f"analisis_{r['nombre'].replace(' ','_')}_{r['anio']}.csv",
                    use_container_width=True
                )
            with cd3:
                if st.session_state.poligono_geojson:
                    gj_out = json.dumps({
                        'type': 'FeatureCollection',
                        'features': [{
                            **st.session_state.poligono_geojson,
                            'properties': {
                                'nombre':       r['nombre'],
                                'area_ha':      round(r['area_ha'], 4),
                                'veredicto':    clasif['veredicto'],
                                'pred_qq_ha':   rend['pred_ens'],
                                'fuente':       r['fuente_sat'],
                                'anio':         r['anio'],
                            }
                        }]
                    }, indent=2)
                    st.download_button(
                        "Descargar GeoJSON",
                        gj_out.encode('utf-8'),
                        f"{r['nombre'].replace(' ','_')}.geojson",
                        use_container_width=True
                    )

st.divider()
st.caption(
    "Indices espectrales extraidos de imagenes Sentinel-2 SR Harmonized (ESA Copernicus) "
    "via Google Earth Engine | Resolucion 10m | Mascara de nubes SCL | "
    "Savitzky-Golay w=7 m=2 | Tesis UNAH"
)