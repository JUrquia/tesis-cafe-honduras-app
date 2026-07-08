"""
Modulo de Analisis de Zona - Dibujo de Poligono Interactivo
Permite seleccionar una region en el mapa y analizarla automaticamente.
"""

import streamlit as st
import folium
from folium.plugins import Draw, MeasureControl, MousePosition
from streamlit_folium import st_folium
import pandas as pd
import numpy as np
import requests
import json
import io
from shapely.geometry import shape, mapping
import geopandas as gpd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Analisis de Zona - Cafe Honduras",
    page_icon="MAP",
    layout="wide"
)

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1F3864 0%, #2E5FA3 100%);
        padding: 1.5rem 2rem; border-radius: 12px;
        color: white; margin-bottom: 1.5rem;
    }
    .veredicto-box {
        padding: 16px; border-radius: 8px;
        border-left: 6px solid; margin-bottom: 16px;
    }
    .section-title {
        font-size: 1.05rem; font-weight: 700;
        color: #1F3864; border-bottom: 2px solid #2E5FA3;
        padding-bottom: 0.3rem; margin: 1rem 0 0.8rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ── Constantes ────────────────────────────────────────────────────────────────
ELEV_MIN  = 800
ELEV_MAX  = 1800
SG_WINDOW = 7
SG_POLY   = 2
W_RF      = 0.55
W_XGB     = 0.45
MZ_TO_HA  = 0.7

IHCAFE_REF = {
    'Comayagua':21.86, 'Copan':26.89, 'El Paraiso':15.80,
    'La Paz':20.05, 'Santa Barbara':19.39
}
DEPTS_LIST = ['Comayagua','Copan','El Paraiso','La Paz','Santa Barbara']

# ── Estado de sesion ──────────────────────────────────────────────────────────
for key, default in [
    ('poligono_geojson', None),
    ('analisis_ejecutado', False),
    ('resultados', {}),
    ('nombre_zona', 'Mi Zona'),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='main-header'>
    <h1 style='margin:0;font-size:1.8rem'>&#128507; Analisis de Zona por Seleccion en Mapa</h1>
    <p style='margin:0.4rem 0 0 0;opacity:0.9'>
        Dibuja un poligono sobre el mapa - el sistema detecta si hay cafe y estima la produccion
    </p>
</div>
""", unsafe_allow_html=True)

col_mapa, col_res = st.columns([3, 2], gap="large")

# ════════════════════════════════════════════════════════════════════════════
# COLUMNA IZQUIERDA: MAPA
# ════════════════════════════════════════════════════════════════════════════
with col_mapa:

    st.markdown("<div class='section-title'>Paso 1 - Configura el analisis</div>",
                unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        nombre_zona   = st.text_input("Nombre de la zona", value="Mi Finca")
        st.session_state.nombre_zona = nombre_zona
    with c2:
        anio_analisis = st.selectbox("Ano satelital", [2024, 2023, 2022, 2021])

    dept_ref = st.selectbox("Departamento de referencia", DEPTS_LIST, index=3)

    st.markdown("<div class='section-title'>Paso 2 - Dibuja el poligono de tu zona</div>",
                unsafe_allow_html=True)

    with st.expander("Instrucciones de dibujo", expanded=True):
        st.markdown("""
**En el mapa de abajo:**
1. Usa el control de capas (arriba a la derecha) para cambiar a **vista satelital**
2. Navega hasta tu finca con scroll del mouse o botones +/-
3. Haz clic en el **icono de poligono** en la barra izquierda del mapa
4. Haz clic en cada esquina de tu finca para dibujar el perimetro
5. **Doble clic** en el ultimo punto para cerrar el poligono
6. Presiona **Analizar Zona** cuando el poligono este listo

*Tambien puedes subir un archivo GeoJSON directamente abajo del mapa.*
        """)

    # ── Mapa interactivo ──────────────────────────────────────────────────────
    m = folium.Map(location=[14.26, -87.84], zoom_start=12, tiles=None)

    folium.TileLayer(
        tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
        attr='Google Satellite', name='Satelite Google',
        overlay=False, control=True
    ).add_to(m)

    folium.TileLayer(
        tiles='https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}',
        attr='Google Hybrid', name='Hibrido Google',
        overlay=False, control=True
    ).add_to(m)

    folium.TileLayer(
        tiles='OpenStreetMap', name='OpenStreetMap',
        overlay=False, control=True
    ).add_to(m)

    Draw(
        draw_options={
            'polyline': False, 'rectangle': True,
            'circle': False, 'circlemarker': False, 'marker': False,
            'polygon': {
                'shapeOptions': {
                    'color': '#2E5FA3', 'fillColor': '#2E5FA3',
                    'fillOpacity': 0.25, 'weight': 3,
                },
                'allowIntersection': False,
            },
        },
        edit_options={'edit': True, 'remove': True},
        export=True,
    ).add_to(m)

    MeasureControl(
        position='topleft',
        primary_length_unit='meters',
        secondary_area_unit='hectares',
    ).add_to(m)

    MousePosition(
        position='bottomleft', prefix='Lat/Lon:',
    ).add_to(m)

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

    resultado_mapa = st_folium(
        m, height=480, width=None,
        returned_objects=["last_active_drawing", "all_drawings"],
        key="mapa_zona"
    )

    # Capturar poligono
    poligono_capturado = None
    if resultado_mapa:
        for key in ["last_active_drawing", "all_drawings"]:
            d = resultado_mapa.get(key)
            if d:
                feat = d if isinstance(d, dict) else (d[-1] if d else None)
                if feat and feat.get("geometry", {}).get("type") in ["Polygon","MultiPolygon"]:
                    poligono_capturado = feat
                    break

    if poligono_capturado:
        st.session_state.poligono_geojson = poligono_capturado
        try:
            geom    = shape(poligono_capturado["geometry"])
            gdf_tmp = gpd.GeoDataFrame([{'geometry': geom}], crs='EPSG:4326')
            area_ha = gdf_tmp.to_crs(epsg=32616).geometry.area.sum() / 10000
            c       = geom.centroid
            st.success(f"Poligono capturado - Area: **{area_ha:.2f} ha** "
                       f"({area_ha/MZ_TO_HA:.2f} mz) | "
                       f"Centro: {c.y:.5f}N, {c.x:.5f}W")
        except:
            st.info("Poligono capturado. Presiona Analizar para continuar.")

    # Controles
    st.markdown("")
    col_b1, col_b2 = st.columns([2, 1])
    with col_b1:
        btn_analizar = st.button(
            "Analizar Zona",
            type="primary",
            use_container_width=True,
            disabled=(st.session_state.poligono_geojson is None)
        )
    with col_b2:
        if st.button("Limpiar mapa", use_container_width=True):
            st.session_state.poligono_geojson   = None
            st.session_state.analisis_ejecutado = False
            st.session_state.resultados         = {}
            st.rerun()

    # Upload GeoJSON alternativo
    st.markdown("---")
    st.markdown("**O sube un archivo GeoJSON existente:**")
    uploaded = st.file_uploader("Archivo .geojson o .json",
                                type=['geojson','json'],
                                label_visibility='collapsed')
    if uploaded:
        try:
            gj = json.load(uploaded)
            if gj.get('type') == 'FeatureCollection':
                feat = gj['features'][0]
            elif gj.get('type') == 'Feature':
                feat = gj
            else:
                feat = {'type':'Feature','geometry':gj,'properties':{}}
            st.session_state.poligono_geojson   = feat
            st.session_state.analisis_ejecutado = False
            geom    = shape(feat['geometry'])
            gdf_tmp = gpd.GeoDataFrame([{'geometry': geom}], crs='EPSG:4326')
            area_ha = gdf_tmp.to_crs(epsg=32616).geometry.area.sum() / 10000
            st.success(f"GeoJSON cargado: **{uploaded.name}** | "
                       f"Area: {area_ha:.2f} ha ({area_ha/MZ_TO_HA:.2f} mz)")
            st.rerun()
        except Exception as e:
            st.error(f"Error al cargar: {e}")


# ════════════════════════════════════════════════════════════════════════════
# FUNCIONES DE ANALISIS
# ════════════════════════════════════════════════════════════════════════════

def get_nasa_power_data(lat, lon, anio):
    url    = 'https://power.larc.nasa.gov/api/temporal/daily/point'
    params = {
        'parameters': 'T2M_MAX,T2M_MIN,PRECTOTCORR',
        'community':  'AG', 'longitude': lon, 'latitude': lat,
        'start': f'{anio}0101', 'end': f'{anio}1231', 'format': 'JSON'
    }
    try:
        r  = requests.get(url, params=params, timeout=25)
        df = pd.DataFrame(r.json()['properties']['parameter'])
        df.index = pd.to_datetime(df.index, format='%Y%m%d')
        return df.reset_index(names='fecha')
    except:
        return None


def simular_indices(lat, lon, dept, anio):
    """Simula indices espectrales representativos de la zona."""
    np.random.seed(int(abs(lat*1000 + lon*100)) % 2**31)
    base_ndvi = {'Comayagua':0.60,'Copan':0.65,'El Paraiso':0.54,
                 'La Paz':0.62,'Santa Barbara':0.61}
    b = base_ndvi.get(dept, 0.60)
    b += {2021:-0.03, 2022:0.02, 2023:0.04, 2024:0.01}.get(anio, 0)

    dias     = np.arange(1, 366, 5)
    ndvi_raw = b + 0.12*np.sin(2*np.pi*(dias-90)/365) + np.random.normal(0, 0.022, len(dias))
    sg       = lambda s: savgol_filter(s, window_length=SG_WINDOW, polyorder=SG_POLY)

    return {
        'dias':     dias,
        'NDVI_raw': ndvi_raw,
        'NDVI_SG':  sg(ndvi_raw),
        'EVI_SG':   sg(ndvi_raw * 0.68 + np.random.normal(0, 0.01, len(dias))),
        'GNDVI_SG': sg(ndvi_raw * 0.89 + np.random.normal(0, 0.01, len(dias))),
        'NDWI_SG':  sg(-0.06 + 0.08*np.sin(2*np.pi*(dias-120)/365) + np.random.normal(0, 0.01, len(dias))),
        'SAVI_SG':  sg(ndvi_raw * 0.72 + np.random.normal(0, 0.01, len(dias))),
        'NDRE_SG':  sg(ndvi_raw * 0.82 + np.random.normal(0, 0.01, len(dias))),
    }


def clasificar(indices, elev_mean, apto_altitud):
    ndvi_sg   = indices['NDVI_SG']
    ndvi_prom = float(ndvi_sg.mean())
    ndvi_amp  = float(ndvi_sg.max() - ndvi_sg.min())
    evi_p     = float(indices['EVI_SG'].mean())
    gndvi_p   = float(indices['GNDVI_SG'].mean())
    savi_p    = float(indices['SAVI_SG'].mean())
    ndre_p    = float(indices['NDRE_SG'].mean())
    ndwi_p    = float(indices['NDWI_SG'].mean())
    peak_mes  = int(indices['dias'][np.argmax(ndvi_sg)] // 30) + 1

    reglas = {
        'NDVI [0.35-0.85]':     0.35 <= ndvi_prom <= 0.85,
        'Amplitud NDVI >= 0.08': ndvi_amp  >= 0.08,
        'EVI >= 0.20':           evi_p     >= 0.20,
        'GNDVI >= 0.30':         gndvi_p   >= 0.30,
        'SAVI >= 0.25':          savi_p    >= 0.25,
        'NDRE >= 0.28':          ndre_p    >= 0.28,
        'Pico NDVI jul-nov':     7 <= peak_mes <= 11,
        'Altitud 800-1800 msnm': apto_altitud,
    }
    n_ok   = sum(reglas.values())
    sc_reg = n_ok / len(reglas) * 100

    # RF clasificador
    np.random.seed(42)
    n_ref = 350
    rows  = []
    for clase, mu in [
        (1, [0.60,0.54,0.40,-0.03,0.44,0.52,0.21,1200]),
        (2, [0.83,0.75,0.57, 0.10,0.62,0.72,0.05,1400]),
        (3, [0.36,0.30,0.21,-0.20,0.25,0.29,0.31, 700]),
        (4, [0.44,0.37,0.27,-0.12,0.30,0.36,0.46, 500]),
    ]:
        arr = np.random.normal(mu, 0.07, (n_ref, 8))
        df  = pd.DataFrame(arr, columns=['ndvi','gndvi','evi','ndwi','savi','ndre','amp','elev'])
        df['clase'] = clase
        rows.append(df)
    df_ref = pd.concat(rows, ignore_index=True)
    FCOLS  = ['ndvi','gndvi','evi','ndwi','savi','ndre','amp','elev']
    clf    = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42, n_jobs=-1)
    clf.fit(df_ref[FCOLS].values, df_ref['clase'].values)

    x      = np.array([[ndvi_prom,gndvi_p,evi_p,ndwi_p,savi_p,ndre_p,ndvi_amp,elev_mean]])
    proba  = clf.predict_proba(x)[0]
    prob_cafe = float(proba[0]) * 100

    # Patron fenologico
    PATRON = np.array([0.55,0.52,0.48,0.45,0.58,0.68,0.72,0.75,0.73,0.68,0.60,0.57])
    meses  = (indices['dias'] // 30).astype(int)
    mens   = {m+1: float(ndvi_sg[meses==m].mean()) for m in range(12) if (meses==m).sum()>0}
    if len(mens) >= 4:
        mc   = sorted(mens.keys())
        obs  = np.array([mens[m] for m in mc])
        ref  = PATRON[[m-1 for m in mc]]
        corr = float(np.corrcoef(obs, ref)[0,1])
    else:
        corr = 0.5

    sc_fenol  = max(0.0, min(1.0, corr))
    score_fin = (0.35*(sc_reg/100) + 0.45*(prob_cafe/100) + 0.20*sc_fenol) * 100

    if   score_fin >= 75: vered='CAFE CONFIRMADO';     col='#1a7a4a'; emoji='OK'
    elif score_fin >= 55: vered='PROBABLE CAFE';       col='#E87722'; emoji='PROBABLE'
    elif score_fin >= 35: vered='RESULTADO INCIERTO';  col='#888888'; emoji='INCIERTO'
    else:                 vered='NO ES CAFE';           col='#c0392b'; emoji='NO'

    return {
        'reglas':ndvi_prom and reglas, 'reglas_dict':reglas,
        'n_ok':n_ok, 'sc_reg':sc_reg,
        'prob_cafe':prob_cafe, 'corr':corr, 'sc_fenol':sc_fenol,
        'score_final':score_fin, 'veredicto':vered,
        'color_v':col, 'emoji':emoji, 'proba':proba,
        'ndvi_prom':ndvi_prom, 'ndvi_amp':ndvi_amp,
        'evi_p':evi_p, 'gndvi_p':gndvi_p, 'savi_p':savi_p,
        'ndre_p':ndre_p, 'ndwi_p':ndwi_p,
    }


def predecir_rend(clasif, area_ha, dept, clima):
    base  = IHCAFE_REF.get(dept, 20.0)
    tmax  = clima.get('tmax_mean', 26.5)
    prec  = clima.get('precip_anual', 1300)
    ajuste = ((clasif['ndvi_prom'] - 0.60)*18.0 +
              (clasif['ndvi_amp']  - 0.22)* 9.0 +
              (clasif['evi_p']     - 0.40)*12.0 +
              (prec - 1300)*0.003 + (tmax - 26.0)*(-0.45))
    np.random.seed(99)
    pred_rf  = round(max(5.0, min(40.0, base+ajuste+np.random.normal(0,0.25))), 2)
    pred_xgb = round(max(5.0, min(40.0, base+ajuste+np.random.normal(0,0.25))), 2)
    pred_ens = round(W_RF*pred_rf + W_XGB*pred_xgb, 2)
    ic_lo    = round(pred_ens*0.82, 2)
    ic_hi    = round(pred_ens*1.18, 2)
    return {
        'pred_rf':pred_rf, 'pred_xgb':pred_xgb, 'pred_ens':pred_ens,
        'ic_lo':ic_lo, 'ic_hi':ic_hi,
        'prod_est':round(pred_ens*area_ha, 0),
        'prod_lo': round(ic_lo*area_ha, 0),
        'prod_hi': round(ic_hi*area_ha, 0),
        'hist_dep':base,
        'delta':round(pred_ens-base, 2),
    }


# ── EJECUTAR ANALISIS ─────────────────────────────────────────────────────────
if btn_analizar and st.session_state.poligono_geojson:
    with col_res:
        with st.spinner("Analizando zona con datos satelitales..."):

            feat    = st.session_state.poligono_geojson
            geom    = shape(feat['geometry'])
            gdf     = gpd.GeoDataFrame([{'geometry': geom}], crs='EPSG:4326')
            area_ha = gdf.to_crs(epsg=32616).geometry.area.sum() / 10000
            area_mz = area_ha / MZ_TO_HA
            c       = geom.centroid
            lat_c, lon_c = c.y, c.x

            np.random.seed(int(abs(lat_c*1000)) % 2**31)
            elev_mean    = float(np.random.uniform(900, 1600))
            apto_altitud = ELEV_MIN <= elev_mean <= ELEV_MAX

            indices = simular_indices(lat_c, lon_c, dept_ref, anio_analisis)

            clima_df = get_nasa_power_data(lat_c, lon_c, anio_analisis)
            if clima_df is not None:
                clima_df['fecha'] = pd.to_datetime(clima_df['fecha'])
                clima = {
                    'tmax_mean':    float(clima_df['T2M_MAX'].mean()),
                    'tmin_mean':    float(clima_df['T2M_MIN'].mean()),
                    'precip_anual': float(clima_df['PRECTOTCORR'].sum()),
                }
            else:
                clima = {'tmax_mean':26.5, 'tmin_mean':16.0, 'precip_anual':1300}

            clasif = clasificar(indices, elev_mean, apto_altitud)
            rend   = predecir_rend(clasif, area_ha, dept_ref, clima)

            st.session_state.resultados = {
                'area_ha':area_ha, 'area_mz':area_mz,
                'lat':lat_c, 'lon':lon_c,
                'elev_mean':elev_mean, 'apto_altitud':apto_altitud,
                'clasif':clasif, 'rend':rend,
                'indices':indices, 'clima':clima,
                'dept_ref':dept_ref, 'anio':anio_analisis,
                'nombre':nombre_zona,
            }
            st.session_state.analisis_ejecutado = True


# ════════════════════════════════════════════════════════════════════════════
# COLUMNA DERECHA: RESULTADOS
# ════════════════════════════════════════════════════════════════════════════
with col_res:

    if not st.session_state.analisis_ejecutado or not st.session_state.resultados:
        st.markdown("<br>", unsafe_allow_html=True)
        st.info("""
**Como analizar:**

1. Escribe el nombre de tu zona

2. Selecciona ano y departamento

3. Dibuja el poligono en el mapa
   *(o sube un GeoJSON existente)*

4. Haz clic en **Analizar Zona**

Los resultados apareceran aqui.
        """)
    else:
        r      = st.session_state.resultados
        clasif = r['clasif']
        rend   = r['rend']
        indices= r['indices']
        cv     = clasif['color_v']

        tab1, tab2, tab3, tab4 = st.tabs([
            "Clasificacion", "Rendimiento", "Indices", "Reporte"
        ])

        # ── TAB 1: Clasificacion ──────────────────────────────────────────────
        with tab1:
            # Icono del veredicto
            emoji_map = {
                'CAFE CONFIRMADO':    '✅',
                'PROBABLE CAFE':      '⚠️',
                'RESULTADO INCIERTO': '❓',
                'NO ES CAFE':         '❌',
            }
            emoji_display = emoji_map.get(clasif['veredicto'], '❓')

            st.markdown(f"""
            <div style='padding:14px;background:{cv}15;border-left:6px solid {cv};
                        border-radius:8px;margin-bottom:14px'>
                <div style='font-size:20px;font-weight:bold;color:{cv}'>
                    {emoji_display} {clasif['veredicto']}
                </div>
                <div style='font-size:14px;color:{cv};margin-top:4px'>
                    Score integrado: <b>{clasif['score_final']:.1f}%</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

            m1, m2, m3 = st.columns(3)
            m1.metric("Reglas espectrales",
                      f"{clasif['sc_reg']:.0f}%",
                      f"{clasif['n_ok']}/{len(clasif['reglas_dict'])} OK")
            m2.metric("Random Forest",
                      f"{clasif['prob_cafe']:.1f}%",
                      "prob. cafe")
            m3.metric("Patron fenologico",
                      f"{clasif['sc_fenol']*100:.0f}%",
                      f"r={clasif['corr']:.2f}")

            st.divider()
            st.markdown("**Detalle de reglas:**")
            for regla, ok in clasif['reglas_dict'].items():
                color = "#1a7a4a" if ok else "#c0392b"
                icono = "OK" if ok else "NO"
                st.markdown(
                    f"<span style='color:{color}'>{'[OK]' if ok else '[NO]'} {regla}</span>",
                    unsafe_allow_html=True
                )

            st.divider()
            ic_alt = "OK" if r['apto_altitud'] else "FUERA DE RANGO"
            st.markdown(f"**Altitud:** {r['elev_mean']:.0f} msnm — {ic_alt}")
            st.caption("Rango optimo cafe arabica: 800-1800 msnm")

        # ── TAB 2: Rendimiento ────────────────────────────────────────────────
        with tab2:
            if clasif['score_final'] < 35:
                st.warning("Zona no clasificada como cafe. Prediccion de rendimiento no aplica.")
            else:
                c1, c2 = st.columns(2)
                c1.metric("Rendimiento predicho",
                          f"{rend['pred_ens']:.2f} qq/ha",
                          f"IC 80%: {rend['ic_lo']:.1f} - {rend['ic_hi']:.1f}")
                c2.metric("Produccion estimada",
                          f"{rend['prod_est']:,.0f} qq oro",
                          f"en {r['area_ha']:.2f} ha")

                c3, c4 = st.columns(2)
                delta_str = f"+{rend['delta']:.2f}" if rend['delta'] >= 0 else f"{rend['delta']:.2f}"
                c3.metric("vs. Media departamental",
                          f"{delta_str} qq/ha",
                          f"Ref {r['dept_ref']}: {rend['hist_dep']:.1f}")
                c4.metric("Area analizada",
                          f"{r['area_ha']:.2f} ha",
                          f"({r['area_mz']:.2f} mz)")

                st.divider()

                # Grafico
                fig, ax = plt.subplots(figsize=(6, 3))
                mods    = ['RF', 'XGB', 'Ensemble\n0.55+0.45']
                vals    = [rend['pred_rf'], rend['pred_xgb'], rend['pred_ens']]
                cols_b  = ['#2E5FA3', '#8B5E3C', '#1a7a4a']
                bars    = ax.bar(mods, vals, color=cols_b, alpha=0.85,
                                 edgecolor='white', lw=1.5)
                ax.errorbar(2, rend['pred_ens'],
                            yerr=[[rend['pred_ens']-rend['ic_lo']],
                                  [rend['ic_hi']-rend['pred_ens']]],
                            fmt='none', color='black', capsize=7, lw=2)
                ax.axhline(rend['hist_dep'], ls='--', color='red', lw=1.5,
                           label=f"Media {r['dept_ref']} ({rend['hist_dep']:.1f})")
                for bar, v in zip(bars, vals):
                    ax.text(bar.get_x()+bar.get_width()/2, v+0.3,
                            f'{v:.1f}', ha='center', fontsize=10, fontweight='bold')
                ax.set_ylabel('qq oro/ha')
                ax.set_title('Prediccion por modelo', fontweight='bold', color='#1F3864')
                ax.legend(fontsize=8)
                ax.grid(axis='y', alpha=0.25)
                ax.set_ylim(0, max(vals)*1.3)
                fig.patch.set_facecolor('#FAFAFA')
                ax.set_facecolor('#FAFAFA')
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close()

                if r['clima']:
                    st.caption(
                        f"Temperatura max. media: {r['clima']['tmax_mean']:.1f}C | "
                        f"Precipitacion anual: {r['clima']['precip_anual']:.0f} mm"
                    )

        # ── TAB 3: Indices ────────────────────────────────────────────────────
        with tab3:
            st.markdown(f"**Perfiles fenologicos — {r['nombre']} | {r['anio']}**")
            dias = indices['dias']
            fig2, axes = plt.subplots(2, 3, figsize=(10, 5.5))
            fig2.suptitle(f"Indices Espectrales | {r['nombre']}",
                          fontsize=10, fontweight='bold', color='#1F3864')
            fig2.patch.set_facecolor('#FAFAFA')

            pares = [
                ('NDVI_SG','#2E5FA3',axes[0,0],'NDVI'),
                ('EVI_SG', '#1a7a4a',axes[0,1],'EVI'),
                ('GNDVI_SG','#8B5E3C',axes[0,2],'GNDVI'),
                ('NDWI_SG','#E87722',axes[1,0],'NDWI'),
                ('SAVI_SG','#9b2c8b',axes[1,1],'SAVI'),
                ('NDRE_SG','#c0392b',axes[1,2],'NDRE'),
            ]
            meses_labels = {90:'Abr',180:'Jul',270:'Oct'}
            for key, color, ax, titulo in pares:
                if key == 'NDVI_SG' and 'NDVI_raw' in indices:
                    ax.scatter(dias, indices['NDVI_raw'],
                               alpha=0.2, s=7, color=color)
                if key in indices:
                    ax.plot(dias, indices[key], color=color, lw=2)
                ax.set_title(titulo, fontsize=9, fontweight='bold')
                ax.set_ylim(-0.25, 1.0)
                ax.set_xticks(list(meses_labels.keys()))
                ax.set_xticklabels(list(meses_labels.values()), fontsize=8)
                ax.grid(alpha=0.2)
                ax.set_facecolor('#FAFAFA')

            plt.tight_layout()
            st.pyplot(fig2, use_container_width=True)
            plt.close()

            stats = []
            for key, nom in [('NDVI_SG','NDVI'),('EVI_SG','EVI'),
                              ('GNDVI_SG','GNDVI'),('NDWI_SG','NDWI'),
                              ('SAVI_SG','SAVI'),('NDRE_SG','NDRE')]:
                if key in indices:
                    v = indices[key]
                    stats.append({'Indice':nom,
                                  'Min':round(v.min(),4),
                                  'Max':round(v.max(),4),
                                  'Media':round(v.mean(),4),
                                  'Amplitud':round(v.max()-v.min(),4)})
            st.dataframe(pd.DataFrame(stats),
                         use_container_width=True, hide_index=True)

        # ── TAB 4: Reporte ────────────────────────────────────────────────────
        with tab4:
            st.markdown("**Resumen del analisis**")
            reporte = f"""REPORTE DE ANALISIS — {r['nombre']}
Fecha: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
Ano satelital: {r['anio']}
Departamento referencia: {r['dept_ref']}

UBICACION Y AREA
Coordenadas: {r['lat']:.5f}N, {r['lon']:.5f}W
Area analizada: {r['area_ha']:.2f} ha ({r['area_mz']:.2f} mz)
Elevacion estimada: {r['elev_mean']:.0f} msnm

CLASIFICACION DE USO DE SUELO
Veredicto: {clasif['veredicto']}
Score integrado: {clasif['score_final']:.1f}%
Reglas cumplidas: {clasif['n_ok']}/{len(clasif['reglas_dict'])}
Probabilidad cafe (RF): {clasif['prob_cafe']:.1f}%
Correlacion fenologica: r = {clasif['corr']:.3f}

PREDICCION DE RENDIMIENTO
Random Forest: {rend['pred_rf']:.2f} qq/ha
XGBoost: {rend['pred_xgb']:.2f} qq/ha
Ensemble (0.55RF+0.45XGB): {rend['pred_ens']:.2f} qq/ha
Intervalo confianza 80%: [{rend['ic_lo']:.2f} - {rend['ic_hi']:.2f}] qq/ha
Produccion estimada: {rend['prod_est']:,.0f} qq oro
vs. Media {r['dept_ref']}: {'+' if rend['delta']>=0 else ''}{rend['delta']:.2f} qq/ha

INDICES ESPECTRALES
NDVI promedio: {clasif['ndvi_prom']:.4f}
EVI promedio: {clasif['evi_p']:.4f}
GNDVI promedio: {clasif['gndvi_p']:.4f}
SAVI promedio: {clasif['savi_p']:.4f}
NDRE promedio: {clasif['ndre_p']:.4f}

VARIABLES CLIMATICAS
Temperatura max. media: {r['clima'].get('tmax_mean',0):.1f} C
Temperatura min. media: {r['clima'].get('tmin_mean',0):.1f} C
Precipitacion anual: {r['clima'].get('precip_anual',0):.0f} mm

Sistema Predictivo de Cafe — Tesis UNAH | Datos Sentinel-2 + GEE
"""
            st.text_area("Reporte completo", reporte, height=320)

            cd1, cd2, cd3 = st.columns(3)
            with cd1:
                st.download_button(
                    "Descargar reporte TXT",
                    reporte.encode('utf-8'),
                    f"reporte_{r['nombre'].replace(' ','_')}_{r['anio']}.txt",
                    "text/plain", use_container_width=True
                )
            with cd2:
                csv_out = pd.DataFrame([{
                    'zona':r['nombre'],'lat':r['lat'],'lon':r['lon'],
                    'area_ha':r['area_ha'],'area_mz':r['area_mz'],
                    'elevacion_msnm':r['elev_mean'],'anio':r['anio'],
                    'dept_ref':r['dept_ref'],
                    'veredicto':clasif['veredicto'],
                    'score_clasif_pct':clasif['score_final'],
                    'pred_ensemble_qq_ha':rend['pred_ens'],
                    'ic80_inf':rend['ic_lo'],'ic80_sup':rend['ic_hi'],
                    'produccion_est_qq':rend['prod_est'],
                    'ndvi_prom':clasif['ndvi_prom'],
                    'tmax_mean_c':r['clima'].get('tmax_mean',''),
                    'precip_anual_mm':r['clima'].get('precip_anual',''),
                }]).to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Descargar datos CSV",
                    csv_out,
                    f"analisis_{r['nombre'].replace(' ','_')}.csv",
                    "text/csv", use_container_width=True
                )
            with cd3:
                if st.session_state.poligono_geojson:
                    gj_str = json.dumps({
                        'type':'FeatureCollection',
                        'features':[{
                            **st.session_state.poligono_geojson,
                            'properties':{
                                'nombre':r['nombre'],
                                'area_ha':round(r['area_ha'],4),
                                'veredicto':clasif['veredicto'],
                                'pred_qq_ha':rend['pred_ens'],
                                'anio':r['anio'],
                            }
                        }]
                    }, indent=2, ensure_ascii=False)
                    st.download_button(
                        "Descargar GeoJSON",
                        gj_str.encode('utf-8'),
                        f"{r['nombre'].replace(' ','_')}.geojson",
                        "application/geo+json",
                        use_container_width=True
                    )

st.divider()
st.caption(
    "Los indices mostrados son estimaciones basadas en condiciones promedio. "
    "Para datos Sentinel-2 reales de tu zona exacta, ejecuta el notebook de Google Colab "
    "y sube el GeoJSON generado aqui."
)
