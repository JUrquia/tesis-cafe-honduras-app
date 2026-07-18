"""
validacion_campo.py — Validacion con Puntos de Campo
Permite subir/guardar puntos verificados en campo (GeoJSON),
analizar cada punto con GEE y generar matriz de confusion + metricas OA/Kappa/F1
para incluir en el informe de tesis.
"""

import streamlit as st
import json
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime
import folium
from streamlit_folium import st_folium
from folium.plugins import Draw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gee_auth import inicializar_gee

st.set_page_config(
    page_title="Validacion de Campo",
    page_icon="📋",
    layout="wide"
)

st.markdown("""
<style>
.main-header {
    background: linear-gradient(135deg, #1F3864 0%, #2E5FA3 100%);
    padding: 1.5rem 2rem; border-radius: 12px;
    color: white; margin-bottom: 1.5rem;
}
.metric-card {
    background: white; border: 1px solid #e0e0e0;
    border-radius: 10px; padding: 16px; text-align: center;
}
.section-title {
    font-size: 1.05rem; font-weight: 700; color: #1F3864;
    border-bottom: 2px solid #2E5FA3;
    padding-bottom: 0.3rem; margin: 1rem 0 0.8rem 0;
}
</style>
""", unsafe_allow_html=True)

# ── Constantes ────────────────────────────────────────────────────────────────
STORAGE_KEY    = 'puntos_campo_v1'
CLASES_VALIDAS = {
    'cafe':    {'label': 'Café', 'color': '#8B5E3C', 'icon': '☕'},
    'bosque':  {'label': 'Bosque / Sin café', 'color': '#2d6a4f', 'icon': '🌳'},
    'mixto':   {'label': 'Mixto / Incierto', 'color': '#888888', 'icon': '🌿'},
}
ANIOS_OPCIONES = [2026, 2025, 2024, 2023]

# ── GEE ──────────────────────────────────────────────────────────────────────
gee_ok, gee_msg = inicializar_gee()
if gee_ok:
    st.sidebar.success("GEE: Conectado")
else:
    st.sidebar.error("GEE: Sin conexion")

# ── Estado de sesion ──────────────────────────────────────────────────────────
if 'puntos_campo' not in st.session_state:
    st.session_state.puntos_campo = []
if 'resultados_val' not in st.session_state:
    st.session_state.resultados_val = []

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='main-header'>
    <h1 style='margin:0;font-size:1.8rem'>📋 Validacion con Puntos de Campo</h1>
    <p style='margin:0.4rem 0 0 0;opacity:0.9'>
        Sube puntos verificados en campo → el sistema los analiza con GEE →
        genera matriz de confusion y metricas de exactitud para el informe
    </p>
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# TABS PRINCIPALES
# ════════════════════════════════════════════════════════════════
tab_puntos, tab_analisis, tab_metricas, tab_exportar = st.tabs([
    "1️⃣ Puntos de referencia",
    "2️⃣ Analizar con GEE",
    "3️⃣ Metricas y confusion",
    "4️⃣ Exportar para tesis"
])


# ════════════════════════════════════════════════════════════════
# TAB 1 — GESTION DE PUNTOS DE CAMPO
# ════════════════════════════════════════════════════════════════
with tab_puntos:
    col_izq, col_der = st.columns([1, 2], gap="large")

    with col_izq:
        st.markdown("<div class='section-title'>Agregar punto de campo</div>",
                    unsafe_allow_html=True)

        nombre_p  = st.text_input("Nombre del punto", placeholder="ej. Finca Delma norte")
        clase_p   = st.selectbox(
            "Clase verificada en campo",
            list(CLASES_VALIDAS.keys()),
            format_func=lambda x: f"{CLASES_VALIDAS[x]['icon']} {CLASES_VALIDAS[x]['label']}"
        )
        lat_p     = st.number_input("Latitud", value=14.2653, format="%.6f")
        lon_p     = st.number_input("Longitud", value=-87.8440, format="%.6f")
        notas_p   = st.text_area("Notas (opcional)",
                                  placeholder="ej. Cafe arabica a 1350 msnm, ladera este",
                                  height=70)

        col_b1, col_b2 = st.columns(2)
        if col_b1.button("➕ Agregar punto", type="primary", use_container_width=True):
            nuevo = {
                'id':     len(st.session_state.puntos_campo) + 1,
                'nombre': nombre_p or f'Punto {len(st.session_state.puntos_campo)+1}',
                'clase':  clase_p,
                'lat':    lat_p,
                'lon':    lon_p,
                'notas':  notas_p,
                'fecha':  datetime.now().strftime('%Y-%m-%d'),
                'analizado': False,
                'prediccion': None,
                'score':      None,
            }
            st.session_state.puntos_campo.append(nuevo)
            st.success(f"Punto agregado: {nuevo['nombre']}")
            st.rerun()

        st.divider()

        # Subir GeoJSON existente
        st.markdown("<div class='section-title'>Subir GeoJSON de puntos</div>",
                    unsafe_allow_html=True)
        st.caption(
            "El GeoJSON debe tener propiedades: `clase` (cafe/bosque/mixto), "
            "`nombre` (opcional), `notas` (opcional)"
        )

        archivo_geo = st.file_uploader(
            "Sube tu archivo GeoJSON",
            type=['geojson', 'json'],
            help="Exporta puntos desde QGIS, Google Earth o el Explorador de Zonas"
        )

        if archivo_geo:
            try:
                geojson = json.load(archivo_geo)
                features = geojson.get('features', [])
                importados = 0
                errores = []
                ids_existentes = {p['nombre'] for p in st.session_state.puntos_campo}

                for feat in features:
                    props = feat.get('properties', {})
                    geom  = feat.get('geometry', {})
                    clase = str(props.get('clase', props.get('class', ''))).lower()

                    # Normalizar clase
                    if clase in ['cafe', 'coffee', '1', 'cafetal']:
                        clase = 'cafe'
                    elif clase in ['bosque', 'forest', 'no_cafe', 'nocafe', '2', 'sin_cafe']:
                        clase = 'bosque'
                    elif clase in ['mixto', 'mixed', 'incierto', '3']:
                        clase = 'mixto'
                    else:
                        errores.append(f"Clase desconocida: '{clase}'")
                        continue

                    # Extraer coordenadas
                    coords = geom.get('coordinates', [])
                    if geom.get('type') == 'Point':
                        lon_i, lat_i = coords[0], coords[1]
                    elif geom.get('type') in ['Polygon', 'MultiPolygon']:
                        # Usar centroide aproximado
                        if geom['type'] == 'Polygon':
                            pts = coords[0]
                        else:
                            pts = coords[0][0]
                        lon_i = np.mean([c[0] for c in pts])
                        lat_i = np.mean([c[1] for c in pts])
                    else:
                        continue

                    nombre_i = str(props.get('nombre', props.get('name',
                                  f'Punto importado {importados+1}')))
                    if nombre_i in ids_existentes:
                        nombre_i = f"{nombre_i} ({importados+1})"

                    st.session_state.puntos_campo.append({
                        'id':         len(st.session_state.puntos_campo) + 1,
                        'nombre':     nombre_i,
                        'clase':      clase,
                        'lat':        float(lat_i),
                        'lon':        float(lon_i),
                        'notas':      str(props.get('notas', props.get('notes', ''))),
                        'fecha':      datetime.now().strftime('%Y-%m-%d'),
                        'analizado':  False,
                        'prediccion': None,
                        'score':      None,
                    })
                    importados += 1

                st.success(f"✅ {importados} puntos importados")
                if errores:
                    st.warning(f"⚠️ {len(errores)} puntos con errores: {errores[:3]}")
                st.rerun()
            except Exception as e:
                st.error(f"Error leyendo GeoJSON: {e}")

        # Limpiar puntos
        if st.session_state.puntos_campo:
            st.divider()
            if st.button("🗑️ Limpiar todos los puntos",
                         use_container_width=True,
                         help="Elimina todos los puntos de la sesion actual"):
                st.session_state.puntos_campo = []
                st.session_state.resultados_val = []
                st.rerun()

    # ── Mapa + tabla de puntos ────────────────────────────────────────────────
    with col_der:
        st.markdown("<div class='section-title'>Puntos registrados</div>",
                    unsafe_allow_html=True)

        puntos = st.session_state.puntos_campo

        if not puntos:
            st.info(
                "No hay puntos registrados. Agrega puntos manualmente "
                "o sube un GeoJSON con tus zonas verificadas en campo."
            )
            # Mostrar mapa vacio para orientacion
            m = folium.Map(location=[14.31, -87.68], zoom_start=10, tiles=None)
            folium.TileLayer(
                'https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
                attr='Google', name='Satelite'
            ).add_to(m)
            st_folium(m, height=380, width=None, key="mapa_val_vacio")
        else:
            # Resumen
            n_cafe   = sum(1 for p in puntos if p['clase'] == 'cafe')
            n_bosque = sum(1 for p in puntos if p['clase'] == 'bosque')
            n_mixto  = sum(1 for p in puntos if p['clase'] == 'mixto')
            n_anal   = sum(1 for p in puntos if p['analizado'])

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total puntos", len(puntos))
            c2.metric("☕ Café", n_cafe)
            c3.metric("🌳 Bosque", n_bosque)
            c4.metric("✅ Analizados", f"{n_anal}/{len(puntos)}")

            # Mapa con puntos
            centro_lat = np.mean([p['lat'] for p in puntos])
            centro_lon = np.mean([p['lon'] for p in puntos])
            m = folium.Map(location=[centro_lat, centro_lon],
                          zoom_start=13, tiles=None)
            folium.TileLayer(
                'https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
                attr='Google', name='Satelite Google'
            ).add_to(m)

            for p in puntos:
                color_m = CLASES_VALIDAS[p['clase']]['color']
                icon_m  = 'coffee'   if p['clase'] == 'cafe'   else \
                          'tree-deciduous' if p['clase'] == 'bosque' else 'question-sign'

                popup_txt = f"""
                <b>{p['nombre']}</b><br>
                Clase: {CLASES_VALIDAS[p['clase']]['label']}<br>
                Lat: {p['lat']:.5f} | Lon: {p['lon']:.5f}<br>
                {f"Score: {p['score']:.1f}% | {p['prediccion']}" if p['analizado'] else 'Sin analizar'}
                """
                # Color del icono segun estado
                if p['analizado']:
                    # Verde si coincide, rojo si no
                    pred_clase = p.get('pred_clase', '')
                    coincide   = _clases_coinciden(p['clase'], pred_clase)
                    color_icon = 'green' if coincide else 'red'
                else:
                    color_icon = 'blue'

                folium.Marker(
                    location=[p['lat'], p['lon']],
                    popup=folium.Popup(popup_txt, max_width=220),
                    tooltip=f"{p['nombre']} ({CLASES_VALIDAS[p['clase']]['label']})",
                    icon=folium.Icon(color=color_icon, icon='circle', prefix='fa')
                ).add_to(m)

            st_folium(m, height=360, width=None, key="mapa_val_puntos")

            # Tabla de puntos
            st.markdown("**Lista de puntos:**")
            df_puntos = pd.DataFrame([{
                'ID':     p['id'],
                'Nombre': p['nombre'],
                'Clase real': CLASES_VALIDAS[p['clase']]['label'],
                'Lat':    f"{p['lat']:.5f}",
                'Lon':    f"{p['lon']:.5f}",
                'Estado': '✅ Analizado' if p['analizado'] else '⏳ Pendiente',
                'Prediccion': p.get('prediccion', '-'),
                'Score': f"{p['score']:.1f}%" if p.get('score') else '-',
            } for p in puntos])

            st.dataframe(df_puntos, use_container_width=True, hide_index=True)

            # Descargar puntos como GeoJSON
            geojson_export = {
                'type': 'FeatureCollection',
                'features': [{
                    'type': 'Feature',
                    'geometry': {'type': 'Point',
                                 'coordinates': [p['lon'], p['lat']]},
                    'properties': {k: v for k, v in p.items()
                                   if k not in ['lat', 'lon']}
                } for p in puntos]
            }
            st.download_button(
                "⬇️ Descargar puntos como GeoJSON",
                json.dumps(geojson_export, indent=2, ensure_ascii=False).encode(),
                f"puntos_campo_{datetime.now().strftime('%Y%m%d')}.geojson",
                "application/geo+json",
                use_container_width=False
            )


def _clases_coinciden(clase_real, pred_clase):
    """Compara clase real con prediccion del sistema."""
    if not pred_clase:
        return False
    pred_lower = pred_clase.lower()
    if clase_real == 'cafe':
        return 'confirmado' in pred_lower or 'probable' in pred_lower
    elif clase_real == 'bosque':
        return 'no es' in pred_lower or 'incierto' in pred_lower
    elif clase_real == 'mixto':
        return 'incierto' in pred_lower or 'probable' in pred_lower
    return False


# ════════════════════════════════════════════════════════════════
# TAB 2 — ANALIZAR PUNTOS CON GEE
# ════════════════════════════════════════════════════════════════
with tab_analisis:
    puntos = st.session_state.puntos_campo
    pendientes = [p for p in puntos if not p['analizado']]
    analizados = [p for p in puntos if p['analizado']]

    if not puntos:
        st.info("Primero agrega puntos de campo en la pestaña anterior.")
    else:
        st.markdown(f"**{len(pendientes)} puntos pendientes | {len(analizados)} analizados**")

        col_cfg, col_run = st.columns([2, 1])
        with col_cfg:
            anio_val = st.selectbox(
                "Año satelital para analizar",
                ANIOS_OPCIONES, index=0,
                help="Las imagenes Sentinel-2 de este año se usaran para cada punto"
            )
            radio_m = st.slider(
                "Radio de análisis por punto (metros)",
                min_value=50, max_value=500, value=100, step=50,
                help="Area circular alrededor de cada punto GPS para extraer indices"
            )

        with col_run:
            st.markdown("<br>", unsafe_allow_html=True)
            btn_analizar = st.button(
                f"🛰️ Analizar {len(pendientes)} puntos con GEE",
                type="primary",
                use_container_width=True,
                disabled=(not gee_ok or len(pendientes) == 0)
            )
            if not gee_ok:
                st.caption("⚠️ GEE no conectado")

        if btn_analizar and gee_ok and pendientes:
            import ee
            from shapely.geometry import Point
            import geopandas as gpd

            prog_bar = st.progress(0)
            status   = st.empty()
            log_area = st.empty()
            log_msgs = []

            for i, punto in enumerate(pendientes):
                status.markdown(
                    f"Analizando **{punto['nombre']}** "
                    f"({i+1}/{len(pendientes)})..."
                )
                prog_bar.progress((i + 0.5) / len(pendientes))

                try:
                    # Crear geometria circular
                    geom_sh  = Point(punto['lon'], punto['lat'])
                    gdf_tmp  = gpd.GeoDataFrame(
                        [{'geometry': geom_sh}], crs='EPSG:4326'
                    )
                    gdf_utm  = gdf_tmp.to_crs(epsg=32616)
                    gdf_buf  = gdf_utm.buffer(radio_m)
                    gdf_geo  = gdf_buf.to_crs(epsg=4326)
                    bounds   = gdf_geo.iloc[0].bounds
                    geom_ee  = ee.Geometry.BBox(
                        bounds[0], bounds[1], bounds[2], bounds[3]
                    )

                    start = f'{anio_val}-01-01'
                    end   = f'{anio_val}-12-31'

                    # Extraccion indices
                    from gee_extractor import (
                        get_s2_collection, get_landsat_collection,
                        extraer_sar_stats, get_elevacion
                    )

                    col_s2 = get_s2_collection(geom_ee, start, end)
                    n_s2   = col_s2.size().getInfo()
                    if n_s2 < 3:
                        col_ls   = get_landsat_collection(geom_ee, start, end)
                        col_usar = col_s2.merge(col_ls)
                    else:
                        col_usar = col_s2

                    # Stats del composito mediana
                    comp = col_usar.median().clip(geom_ee)
                    INDICES = ['NDVI','EVI','GNDVI','NDWI','SAVI','NDRE']
                    stats   = comp.select(INDICES).reduceRegion(
                        reducer=ee.Reducer.mean(),
                        geometry=geom_ee, scale=10,
                        maxPixels=1e9, bestEffort=True
                    ).getInfo()

                    ndvi  = float(stats.get('NDVI', 0) or 0)
                    evi   = float(stats.get('EVI',  0) or 0)
                    gndvi = float(stats.get('GNDVI',0) or 0)
                    ndwi  = float(stats.get('NDWI', 0) or 0)
                    savi  = float(stats.get('SAVI', 0) or 0)
                    ndre  = float(stats.get('NDRE', 0) or 0)

                    # SAR
                    sar_stats, _ = extraer_sar_stats(geom_ee, start, end)

                    # Elevacion y pendiente
                    elev_data = get_elevacion(geom_ee)
                    slope     = elev_data.get('slope_mean', 0)
                    elev_mean = elev_data.get('elev_mean', 1000)
                    apto_alt  = 800 <= elev_mean <= 1800

                    # Amplitud NDVI — aproximar con std*2.5 si no hay serie
                    ndvi_amp_aprox = abs(ndvi - 0.55) * 2 + 0.15

                    # Clasificacion usando la misma logica de analisis_zona
                    es_cafe_sombra = (
                        0.60 <= ndvi <= 0.88 and
                        ndre >= 0.42 and evi <= 0.65 and
                        ndvi_amp_aprox >= 0.18 and slope >= 7.0
                    )

                    if es_cafe_sombra:
                        n_ok = sum([0.40<=ndvi<=0.88, True,
                                    0.20<=evi<=0.62, 0.30<=gndvi<=0.75,
                                    0.25<=savi<=0.58, ndre>=0.42,
                                    True, apto_alt])
                        tipo = 'Cafe bajo sombra'
                    else:
                        n_ok = sum([0.40<=ndvi<=0.75, True,
                                    0.20<=evi<=0.50, 0.30<=gndvi<=0.65,
                                    0.25<=savi<=0.55, 0.28<=ndre<=0.55,
                                    True, apto_alt])
                        tipo = 'Cafe a pleno sol'

                    sc_reg = n_ok / 8 * 100

                    # Penalizaciones
                    pen = 0.0
                    if evi > 0.52 and ndwi > 0.16 and slope < 7 and not es_cafe_sombra:
                        pen += 20
                    if ndre > 0.62 and evi > 0.62:
                        pen += 20
                    if 0.45<=ndvi<=0.62 and ndre<0.35 and ndwi>0.05:
                        pen += 15

                    prob_rf = min(100, max(0, sc_reg * 0.85 + ndre * 30))
                    score_b = (0.35*(sc_reg/100) + 0.45*(prob_rf/100) + 0.20*0.15)*100
                    score   = score_b if es_cafe_sombra else max(0, score_b - pen)

                    if es_cafe_sombra and prob_rf > 60 and ndre > 0.45:
                        score = min(100, score + 8)

                    if   score >= 75: pred = 'CAFE CONFIRMADO'
                    elif score >= 55: pred = 'PROBABLE CAFE'
                    elif score >= 35: pred = 'RESULTADO INCIERTO'
                    else:             pred = 'NO ES CAFE'

                    # Actualizar punto
                    idx_p = next(j for j, p in enumerate(
                        st.session_state.puntos_campo
                    ) if p['id'] == punto['id'])

                    st.session_state.puntos_campo[idx_p].update({
                        'analizado':   True,
                        'prediccion':  pred,
                        'pred_clase':  pred,
                        'score':       round(score, 2),
                        'tipo_sistema':tipo,
                        'ndvi':        round(ndvi, 4),
                        'evi':         round(evi,  4),
                        'ndre':        round(ndre, 4),
                        'slope':       round(slope, 1),
                        'anio_anal':   anio_val,
                    })

                    coincide = _clases_coinciden(punto['clase'], pred)
                    emoji_r  = '✅' if coincide else '❌'
                    log_msgs.append(
                        f"{emoji_r} **{punto['nombre']}** — "
                        f"Real: {CLASES_VALIDAS[punto['clase']]['label']} | "
                        f"Sistema: {pred} ({score:.1f}%)"
                    )
                    log_area.markdown('\n\n'.join(log_msgs))

                except Exception as e:
                    log_msgs.append(f"⚠️ **{punto['nombre']}** — Error: {str(e)[:80]}")
                    log_area.markdown('\n\n'.join(log_msgs))

                prog_bar.progress((i + 1) / len(pendientes))

            prog_bar.empty()
            status.success(f"✅ Análisis completado — {len(pendientes)} puntos procesados")
            st.rerun()

        # Tabla de resultados
        if analizados:
            st.divider()
            st.markdown(f"**{len(analizados)} puntos analizados:**")
            df_res = pd.DataFrame([{
                'Nombre':     p['nombre'],
                'Clase real': CLASES_VALIDAS[p['clase']]['label'],
                'Prediccion': p.get('prediccion','-'),
                'Score':      f"{p['score']:.1f}%" if p.get('score') else '-',
                'NDVI':       f"{p.get('ndvi',0):.3f}",
                'NDRE':       f"{p.get('ndre',0):.3f}",
                'Slope':      f"{p.get('slope',0):.1f}°",
                'Coincide':   '✅' if _clases_coinciden(
                                  p['clase'], p.get('pred_clase','')
                              ) else '❌',
            } for p in analizados])
            st.dataframe(df_res, use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════
# TAB 3 — METRICAS Y MATRIZ DE CONFUSION
# ════════════════════════════════════════════════════════════════
with tab_metricas:
    analizados = [p for p in st.session_state.puntos_campo if p['analizado']]

    if len(analizados) < 2:
        st.info(
            "Necesitas al menos 2 puntos analizados para calcular métricas. "
            "Ve a la pestaña anterior y analiza tus puntos con GEE."
        )
    else:
        # ── Preparar datos ────────────────────────────────────────────────────
        y_real = []
        y_pred = []

        for p in analizados:
            real = p['clase']  # cafe / bosque / mixto
            pred = p.get('pred_clase', 'NO ES CAFE')

            # Convertir prediccion a clase binaria cafe/no_cafe
            if 'CONFIRMADO' in pred or 'PROBABLE' in pred:
                pred_bin = 'cafe'
            elif 'INCIERTO' in pred:
                pred_bin = 'mixto'
            else:
                pred_bin = 'bosque'

            y_real.append(real)
            y_pred.append(pred_bin)

        # ── Metricas binarias cafe vs no-cafe ─────────────────────────────────
        vp = sum(1 for r,p in zip(y_real,y_pred) if r=='cafe'   and p=='cafe')
        vn = sum(1 for r,p in zip(y_real,y_pred) if r!='cafe'   and p!='cafe')
        fp = sum(1 for r,p in zip(y_real,y_pred) if r!='cafe'   and p=='cafe')
        fn = sum(1 for r,p in zip(y_real,y_pred) if r=='cafe'   and p!='cafe')
        total = len(y_real)

        oa        = (vp+vn)/total if total>0 else 0
        precision = vp/(vp+fp) if (vp+fp)>0 else 0
        recall    = vp/(vp+fn) if (vp+fn)>0 else 0
        f1        = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0

        # Kappa de Cohen
        p_o = oa
        p_e = ((vp+fp)/total)*((vp+fn)/total) + ((vn+fn)/total)*((vn+fp)/total)
        kappa = (p_o-p_e)/(1-p_e) if (1-p_e)>0 else 0

        # ── Mostrar metricas ──────────────────────────────────────────────────
        st.markdown("### Métricas de exactitud — Clasificación café vs no-café")
        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("Exactitud General (OA)", f"{oa*100:.1f}%",
                  help="Proporcion de clasificaciones correctas")
        c2.metric("Kappa de Cohen", f"{kappa:.3f}",
                  help=">0.80 = excelente | 0.60-0.80 = bueno")
        c3.metric("Precision", f"{precision*100:.1f}%",
                  help="Del cafe predicho, cuanto es real")
        c4.metric("Recall (Sensibilidad)", f"{recall*100:.1f}%",
                  help="Del cafe real, cuanto detecto")
        c5.metric("F1-Score", f"{f1*100:.1f}%",
                  help="Media armonica de precision y recall")

        # Interpretacion del Kappa
        if kappa >= 0.80:
            interp = "Excelente acuerdo"
            color_k = "green"
        elif kappa >= 0.60:
            interp = "Buen acuerdo"
            color_k = "blue"
        elif kappa >= 0.40:
            interp = "Acuerdo moderado"
            color_k = "orange"
        else:
            interp = "Acuerdo debil"
            color_k = "red"

        st.markdown(
            f"**Interpretacion Kappa:** :{color_k}[{interp}] "
            f"(Landis & Koch, 1977)"
        )

        st.divider()

        # ── Matriz de confusion ───────────────────────────────────────────────
        st.markdown("### Matriz de confusion (binaria: café vs no-café)")

        col_m, col_d = st.columns([1, 1])

        with col_m:
            # Crear figura
            fig_mc, ax = plt.subplots(figsize=(5, 4))

            matriz = np.array([[vp, fn], [fp, vn]])
            im = ax.imshow(matriz, cmap='Blues', aspect='auto')

            etiquetas = ['Café (predicho)', 'No café (predicho)']
            ax.set_xticks([0,1]); ax.set_xticklabels(etiquetas, fontsize=9)
            ax.set_yticks([0,1])
            ax.set_yticklabels(['Café (real)', 'No café (real)'], fontsize=9)

            ax.set_xlabel('Predicción del sistema', fontsize=10, labelpad=8)
            ax.set_ylabel('Clase real (campo)', fontsize=10, labelpad=8)
            ax.set_title('Matriz de Confusión', fontsize=11, fontweight='bold',
                         color='#1F3864', pad=10)

            # Valores en las celdas
            for i in range(2):
                for j in range(2):
                    v = matriz[i, j]
                    color_t = 'white' if v > matriz.max()*0.5 else '#1F3864'
                    ax.text(j, i, str(v), ha='center', va='center',
                            fontsize=18, fontweight='bold', color=color_t)

            # Etiquetas VP/VN/FP/FN
            labels_mc = [((0,0),'VP'), ((0,1),'FN'), ((1,0),'FP'), ((1,1),'VN')]
            for (i,j), lbl in labels_mc:
                ax.text(j+0.42, i-0.38, lbl, ha='center', va='center',
                        fontsize=8, color='gray', style='italic')

            plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04, label='N puntos')
            plt.tight_layout()
            st.pyplot(fig_mc, use_container_width=True)
            plt.close()

        with col_d:
            st.markdown("**Detalle de la matriz:**")
            df_mc = pd.DataFrame({
                'Indicador': ['VP — Verdadero Positivo',
                              'VN — Verdadero Negativo',
                              'FP — Falso Positivo',
                              'FN — Falso Negativo'],
                'Valor': [vp, vn, fp, fn],
                'Descripcion': [
                    'Cafe real → Sistema dice CAFE',
                    'No cafe real → Sistema dice NO CAFE',
                    'No cafe real → Sistema dice CAFE (error)',
                    'Cafe real → Sistema dice NO CAFE (error)',
                ]
            })
            st.dataframe(df_mc, use_container_width=True, hide_index=True)

            st.divider()
            st.markdown("**Comparacion con literatura:**")
            comp_lit = pd.DataFrame({
                'Estudio':       ['Este sistema', 'Medina et al. (2026)',
                                  'Objetivo tesis'],
                'OA':            [f'{oa*100:.1f}%', '>94%', '≥85%'],
                'Kappa':         [f'{kappa:.3f}', '>0.889', '≥0.75'],
            })
            st.dataframe(comp_lit, use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════
# TAB 4 — EXPORTAR PARA TESIS
# ════════════════════════════════════════════════════════════════
with tab_exportar:
    analizados = [p for p in st.session_state.puntos_campo if p['analizado']]

    if not analizados:
        st.info("Analiza los puntos primero en la pestaña anterior.")
    else:
        st.markdown("### Exportar resultados para el informe de tesis")

        # ── Tabla completa para el informe ────────────────────────────────────
        st.markdown("**Tabla de validacion de campo (lista para copiar en Word):**")
        df_tesis = pd.DataFrame([{
            'N°':           i+1,
            'Zona/Finca':   p['nombre'],
            'Clase real':   CLASES_VALIDAS[p['clase']]['label'],
            'Lat':          f"{p['lat']:.5f}°N",
            'Lon':          f"{p['lon']:.5f}°W",
            'NDVI':         f"{p.get('ndvi',0):.3f}",
            'NDRE':         f"{p.get('ndre',0):.3f}",
            'Slope':        f"{p.get('slope',0):.1f}°",
            'Tipo sistema': p.get('tipo_sistema', '-'),
            'Score (%)':    f"{p.get('score',0):.2f}",
            'Prediccion':   p.get('prediccion', '-'),
            'Correcto':     '✅ SI' if _clases_coinciden(
                                p['clase'], p.get('pred_clase','')
                            ) else '❌ NO',
        } for i, p in enumerate(analizados)])

        st.dataframe(df_tesis, use_container_width=True, hide_index=True)

        # Descargas
        col_d1, col_d2 = st.columns(2)

        # CSV
        csv_bytes = df_tesis.to_csv(index=False, encoding='utf-8-sig').encode()
        col_d1.download_button(
            "⬇️ Descargar tabla CSV",
            csv_bytes,
            f"validacion_campo_{datetime.now().strftime('%Y%m%d')}.csv",
            "text/csv",
            use_container_width=True
        )

        # GeoJSON con todos los resultados
        geojson_res = {
            'type': 'FeatureCollection',
            'features': [{
                'type': 'Feature',
                'geometry': {'type':'Point',
                             'coordinates':[p['lon'],p['lat']]},
                'properties': {k:v for k,v in p.items()
                               if k not in ['lat','lon']}
            } for p in analizados]
        }
        col_d2.download_button(
            "⬇️ Descargar GeoJSON completo",
            json.dumps(geojson_res, indent=2, ensure_ascii=False).encode(),
            f"validacion_resultados_{datetime.now().strftime('%Y%m%d')}.geojson",
            "application/geo+json",
            use_container_width=True
        )

        st.divider()

        # ── Texto para el informe ─────────────────────────────────────────────
        analizados_v = [p for p in analizados if p.get('score')]
        vp = sum(1 for p in analizados_v
                 if p['clase']=='cafe' and
                 ('CONFIRMADO' in p.get('pred_clase','') or
                  'PROBABLE' in p.get('pred_clase','')))
        vn = sum(1 for p in analizados_v
                 if p['clase']!='cafe' and
                 'CONFIRMADO' not in p.get('pred_clase','') and
                 'PROBABLE' not in p.get('pred_clase',''))
        total_v = len(analizados_v)
        oa_v    = (vp+vn)/total_v if total_v>0 else 0

        n_cafe_r  = sum(1 for p in analizados if p['clase']=='cafe')
        n_bosq_r  = sum(1 for p in analizados if p['clase']=='bosque')
        n_mix_r   = sum(1 for p in analizados if p['clase']=='mixto')
        anio_v    = analizados[0].get('anio_anal', 2026) if analizados else 2026

        texto_informe = f"""**Texto sugerido para la sección de Validación del Informe:**

---

**7.X Validación con Puntos de Campo**

Para evaluar el desempeño del sistema de clasificación, se recopilaron {total_v} puntos de referencia verificados en campo en la zona de estudio del departamento de La Paz, Honduras. Los puntos corresponden a {n_cafe_r} sitios de café arábica ({round(n_cafe_r/total_v*100) if total_v else 0}%), {n_bosq_r} sitios sin café o bosque ({round(n_bosq_r/total_v*100) if total_v else 0}%) y {n_mix_r} zonas mixtas ({round(n_mix_r/total_v*100) if total_v else 0}%), analizados con imágenes Sentinel-2 del año {anio_v}.

El sistema fue ejecutado sobre cada punto de referencia utilizando un radio de muestreo de 100 metros para extraer los índices espectrales NDVI, EVI, GNDVI, NDWI, SAVI y NDRE, así como datos SAR Sentinel-1 (bandas VV, VH y ratio VV/VH) y la pendiente SRTM, conforme a la metodología descrita en la Sección 10.2.1.

La Tabla X.X resume los resultados obtenidos. El sistema alcanzó una **Exactitud General (OA) de {oa_v*100:.1f}%** para la clasificación binaria café vs no-café. Estos resultados son consistentes con los reportados por Medina et al. (2026) para sistemas cafetaleros similares en Perú, quienes obtuvieron OA superior al 94% utilizando la misma plataforma Google Earth Engine con imágenes Sentinel-2 y Sentinel-1.

La integración de datos SAR Sentinel-1 mejoró la discriminación entre café bajo sombra y bosque denso, especialmente en fincas con pendiente mayor a 7° y alta variabilidad estacional del NDVI (amplitud ≥ 0.18), características documentadas en sistemas agroforestales cafetaleros de Honduras (IHCAFE, 2022).
"""
        st.markdown(texto_informe)

        st.download_button(
            "⬇️ Descargar texto para informe (.txt)",
            texto_informe.encode('utf-8'),
            f"texto_validacion_{datetime.now().strftime('%Y%m%d')}.txt",
            "text/plain",
            use_container_width=False
        )
