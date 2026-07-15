"""
Explorador de Zonas Cafetaleras
Genera una capa de probabilidad de cafe sobre el mapa
para que el usuario vea donde hay cafe antes de dibujar su poligono.
"""

import streamlit as st
import folium
from folium.plugins import Draw, MeasureControl, MousePosition
from streamlit_folium import st_folium
import json
import os
import sys
import numpy as np
import pandas as pd
import requests
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gee_auth import inicializar_gee

st.set_page_config(
    page_title="Explorador de Zonas Cafetaleras",
    page_icon="COFFEE",
    layout="wide"
)

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #3d1c02 0%, #8B5E3C 100%);
        padding: 1.5rem 2rem; border-radius: 12px;
        color: white; margin-bottom: 1.5rem;
    }
    .legend-box {
        background: white; border: 1px solid #ccc;
        border-radius: 8px; padding: 12px;
        font-size: 13px;
    }
    .section-title {
        font-size: 1.05rem; font-weight: 700; color: #1F3864;
        border-bottom: 2px solid #8B5E3C;
        padding-bottom: 0.3rem; margin: 1rem 0 0.8rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ── Constantes ────────────────────────────────────────────────────────────────
DEPTS_LIST  = ['Comayagua', 'Copan', 'El Paraiso', 'La Paz', 'Santa Barbara']
MZ_TO_HA    = 0.7

CENTROIDES_DEPT = {
    'Comayagua':    [14.44, -87.64],
    'Copan':        [14.84, -88.87],
    'El Paraiso':   [13.87, -86.78],
    'La Paz':       [14.31, -87.68],
    'Santa Barbara':[14.92, -88.23],
}

ALTITUDES_REF = {
    'Comayagua': 933, 'Copan': 924, 'El Paraiso': 850,
    'La Paz': 1150, 'Santa Barbara': 910,
}

# Rangos NDVI promedio por departamento (datos reales de la extracción satelital)
NDVI_REF = {
    'Comayagua':    {'cafe': 0.60, 'bosque': 0.82, 'pasto': 0.35},
    'Copan':        {'cafe': 0.65, 'bosque': 0.84, 'pasto': 0.33},
    'El Paraiso':   {'cafe': 0.54, 'bosque': 0.80, 'pasto': 0.36},
    'La Paz':       {'cafe': 0.62, 'bosque': 0.83, 'pasto': 0.34},
    'Santa Barbara':{'cafe': 0.61, 'bosque': 0.82, 'pasto': 0.35},
}

# Estado de sesion
for key, default in [
    ('dept_generado',      None),
    ('anio_generado',      None),
    ('capa_generada',      False),
    ('tile_url_cafe',      None),
    ('poligono_explorador', None),
    ('geojson_listo',       None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='main-header'>
    <h1 style='margin:0;font-size:1.8rem'>Explorador de Zonas Cafetaleras</h1>
    <p style='margin:0.4rem 0 0 0;opacity:0.9'>
        El mapa muestra automaticamente donde hay cafe —
        dibuja tu poligono sobre una zona coloreada para analizarla
    </p>
</div>
""", unsafe_allow_html=True)

# ── GEE ──────────────────────────────────────────────────────────────────────
gee_ok, gee_msg = inicializar_gee()
if gee_ok:
    st.sidebar.success("GEE: Conectado")
else:
    st.sidebar.error("GEE: Sin conexion")
    st.sidebar.caption(gee_msg)

# ════════════════════════════════════════════════════════════════
# PANEL DE CONFIGURACION
# ════════════════════════════════════════════════════════════════
col_cfg, col_mapa = st.columns([1, 3], gap="large")

with col_cfg:
    st.markdown("<div class='section-title'>Configuracion</div>",
                unsafe_allow_html=True)

    dept_exp = st.selectbox(
        "Departamento a explorar",
        DEPTS_LIST, index=3,
        help="El mapa cargara la capa de probabilidad de cafe para ese departamento"
    )

    anio_exp = st.selectbox(
        "Ano satelital",
        [2026, 2025, 2024, 2023, 2022],
        index=0,
        help="Imagenes Sentinel-2 del ano seleccionado"
    )

    st.divider()
    st.markdown("**Leyenda de colores:**")
    st.markdown("""
    <div class='legend-box'>
        <div style='margin-bottom:6px'>
            <span style='background:#1F3864;color:white;padding:2px 8px;
                         border-radius:4px;font-size:12px'>■ Azul oscuro</span>
            Alta probabilidad de cafe
        </div>
        <div style='margin-bottom:6px'>
            <span style='background:#2E5FA3;color:white;padding:2px 8px;
                         border-radius:4px;font-size:12px'>■ Azul medio</span>
            Probable cafe
        </div>
        <div style='margin-bottom:6px'>
            <span style='background:#AACBE8;color:#333;padding:2px 8px;
                         border-radius:4px;font-size:12px'>■ Azul claro</span>
            Posible cafe / mixto
        </div>
        <div>
            <span style='background:#f0f0f0;color:#333;padding:2px 8px;
                         border-radius:4px;font-size:12px'>■ Transparente</span>
            No es cafe
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    btn_generar = st.button(
        "Generar capa de cafe",
        type="primary",
        use_container_width=True,
        disabled=not gee_ok,
        help="Descarga la capa de probabilidad de cafe desde GEE (~30-60 seg)"
    )

    st.divider()

    # Info del departamento
    st.markdown(f"**{dept_exp}**")
    alt = ALTITUDES_REF.get(dept_exp, 1000)
    ndvi_cafe = NDVI_REF.get(dept_exp, {}).get('cafe', 0.60)
    st.caption(
        f"Elevacion media: {alt} msnm\n"
        f"NDVI cafe tipico: {ndvi_cafe:.2f}"
    )

    # Si hay poligono dibujado, mostrar boton para ir a Analisis de Zona
    if st.session_state.geojson_listo:
        st.success("Poligono listo para analizar")
        geojson_str = json.dumps(st.session_state.geojson_listo)
        st.download_button(
            "Descargar GeoJSON",
            geojson_str.encode(),
            f"zona_{dept_exp}.geojson",
            "application/geo+json",
            use_container_width=True,
            help="Descarga el poligono y subelo en Analisis de Zona para ver resultados completos"
        )
        st.info(
            "Para ver los resultados completos:\n"
            "1. Descarga el GeoJSON\n"
            "2. Ve a **Analisis de Zona**\n"
            "3. Sube el GeoJSON en la seccion de carga"
        )


# ════════════════════════════════════════════════════════════════
# GENERAR CAPA DE PROBABILIDAD CON GEE
# ════════════════════════════════════════════════════════════════

tile_url_cafe  = None
tile_url_ndvi  = None
tile_url_rgb   = None
capa_disponible = False

if btn_generar and gee_ok:
    with col_mapa:
        with st.spinner(f"Generando capa de cafe para {dept_exp} ({anio_exp})..."):
            try:
                import ee
                from gee_extractor import (
                    mask_s2_scl, add_6_indices, get_s2_collection,
                    get_landsat_collection
                )

                # Cargar geometria del departamento
                NOMBRES_GEE = {
                    'Comayagua': 'Comayagua', 'Copan': 'Copan',
                    'El Paraiso': 'El Paraiso', 'La Paz': 'La Paz',
                    'Santa Barbara': 'Santa Barbara',
                }
                honduras = ee.FeatureCollection('FAO/GAUL/2015/level1') \
                    .filter(ee.Filter.eq('ADM0_NAME', 'Honduras'))
                geom_dept = honduras.filter(
                    ee.Filter.eq('ADM1_NAME', NOMBRES_GEE[dept_exp])
                ).first().geometry()

                # Coleccion Sentinel-2
                ini = f'{anio_exp}-01-01'
                fin = f'{anio_exp}-12-31'
                col_s2 = get_s2_collection(geom_dept, ini, fin)
                n_s2   = col_s2.size().getInfo()

                if n_s2 < 5:
                    col_ls   = get_landsat_collection(geom_dept, ini, fin)
                    col_usar = col_s2.merge(col_ls)
                    fuente   = f'S2({n_s2}) + Landsat'
                else:
                    col_usar = col_s2
                    fuente   = f'Sentinel-2 ({n_s2} escenas)'

                # Composito mediana
                composito = col_usar.median().clip(geom_dept)

                # ── Capa de probabilidad de cafe ──────────────────────────
                # Mascara: NDVI en rango cafe Y EVI en rango cafe
                # Usando umbrales calibrados del informe
                ndvi = composito.select('NDVI')
                evi  = composito.select('EVI')
                gndvi= composito.select('GNDVI')
                ndre = composito.select('NDRE')
                savi = composito.select('SAVI')

                # Score de cafe: ponderacion de indices
                # Cada condicion suma puntos al score
                cond_ndvi  = ndvi.gte(0.40).And(ndvi.lte(0.75))
                cond_evi   = evi.gte(0.20).And(evi.lte(0.50))
                cond_gndvi = gndvi.gte(0.30).And(gndvi.lte(0.65))
                cond_ndre  = ndre.gte(0.28).And(ndre.lte(0.55))
                cond_savi  = savi.gte(0.25).And(savi.lte(0.55))

                # DEM para altitud
                dem   = ee.Image('USGS/SRTMGL1_003')
                elev  = dem.clip(geom_dept)
                cond_alt = elev.gte(800).And(elev.lte(1800))

                # Score ponderado (0-100)
                score_cafe = (
                    cond_ndvi.multiply(25)
                    .add(cond_evi.multiply(25))
                    .add(cond_gndvi.multiply(15))
                    .add(cond_ndre.multiply(15))
                    .add(cond_savi.multiply(10))
                    .add(cond_alt.multiply(10))
                )

                # Mascara: solo mostrar donde score >= 45
                mascara_cafe = score_cafe.gte(45)
                capa_cafe    = score_cafe.updateMask(mascara_cafe)

                # ── Generar tile URLs para el mapa ────────────────────────
                # Capa de probabilidad de cafe
                viz_cafe = {
                    'min': 45, 'max': 100,
                    'palette': ['#AACBE8', '#2E5FA3', '#1F3864']
                }
                map_id_cafe = capa_cafe.getMapId(viz_cafe)
                tile_url_cafe = map_id_cafe['tile_fetcher'].url_format

                # Capa NDVI para referencia
                viz_ndvi = {
                    'min': 0.2, 'max': 0.9,
                    'palette': ['#d73027','#fee08b','#1a9850']
                }
                map_id_ndvi = ndvi.getMapId(viz_ndvi)
                tile_url_ndvi = map_id_ndvi['tile_fetcher'].url_format

                # RGB natural
                viz_rgb = {'bands':['B4','B3','B2'], 'min':0, 'max':0.3, 'gamma':1.4}
                map_id_rgb = composito.getMapId(viz_rgb)
                tile_url_rgb = map_id_rgb['tile_fetcher'].url_format

                # Guardar en sesion
                st.session_state.tile_url_cafe  = tile_url_cafe
                st.session_state.tile_url_ndvi  = tile_url_ndvi
                st.session_state.tile_url_rgb   = tile_url_rgb
                st.session_state.dept_generado  = dept_exp
                st.session_state.anio_generado  = anio_exp
                st.session_state.capa_generada  = True
                capa_disponible = True

                st.success(
                    f"Capa generada para {dept_exp} | "
                    f"{fuente} | {anio_exp}"
                )

            except Exception as e:
                st.error(f"Error generando capa: {e}")

# Recuperar URLs de sesion si ya se generaron
if st.session_state.capa_generada:
    tile_url_cafe  = st.session_state.tile_url_cafe
    tile_url_ndvi  = st.session_state.tile_url_ndvi
    tile_url_rgb   = st.session_state.tile_url_rgb
    capa_disponible = True


# ════════════════════════════════════════════════════════════════
# MAPA INTERACTIVO
# ════════════════════════════════════════════════════════════════
with col_mapa:

    centro = CENTROIDES_DEPT.get(dept_exp, [14.26, -87.84])
    zoom   = 11 if st.session_state.capa_generada else 10

    m = folium.Map(location=centro, zoom_start=zoom, tiles=None)

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

    # ── Capas GEE (si están disponibles) ─────────────────────────────────────
    if capa_disponible and tile_url_cafe:
        # RGB natural
        if tile_url_rgb:
            folium.TileLayer(
                tiles=tile_url_rgb,
                attr='GEE Sentinel-2',
                name='RGB Sentinel-2',
                overlay=True, control=True,
                opacity=0.9
            ).add_to(m)

        # NDVI
        if tile_url_ndvi:
            folium.TileLayer(
                tiles=tile_url_ndvi,
                attr='GEE NDVI',
                name='NDVI (verde = vegetacion)',
                overlay=True, control=True,
                opacity=0.7, show=False
            ).add_to(m)

        # CAPA PRINCIPAL: Probabilidad de cafe
        folium.TileLayer(
            tiles=tile_url_cafe,
            attr='GEE Cafe Score',
            name='Zonas de cafe (azul)',
            overlay=True, control=True,
            opacity=0.75
        ).add_to(m)

        # Tooltip informativo
        st.markdown(
            f"<div style='background:#e8f5e9;border-left:4px solid #1a7a4a;"
            f"padding:8px 12px;border-radius:6px;font-size:13px;margin-bottom:8px'>"
            f"Capa activa: <b>{st.session_state.dept_generado}</b> | "
            f"Ano: <b>{st.session_state.anio_generado}</b> | "
            f"Las zonas azules tienen alta probabilidad de cafe"
            f"</div>",
            unsafe_allow_html=True
        )
    else:
        # Sin capa — mostrar instruccion
        st.info(
            "Selecciona el departamento y el ano, luego presiona "
            "**Generar capa de cafe** para colorear el mapa."
        )

    # ── Herramientas de dibujo ────────────────────────────────────────────────
    Draw(
        draw_options={
            'polyline':     False,
            'rectangle':    True,
            'circle':       False,
            'circlemarker': False,
            'marker':       True,
            'polygon': {
                'shapeOptions': {
                    'color':       '#FF6B35',
                    'fillColor':   '#FF6B35',
                    'fillOpacity': 0.25,
                    'weight':      3,
                },
            },
        },
        edit_options={'edit': True, 'remove': True},
        export=True,
    ).add_to(m)

    MeasureControl(
        position='topleft',
        primary_area_unit='hectares',
    ).add_to(m)

    MousePosition(
        position='bottomleft', prefix='Lat/Lon:'
    ).add_to(m)

    # Mostrar poligono guardado
    if st.session_state.poligono_explorador:
        folium.GeoJson(
            st.session_state.poligono_explorador,
            name='Mi poligono',
            style_function=lambda x: {
                'fillColor': '#FF6B35',
                'color':     '#CC3D00',
                'weight':    3,
                'fillOpacity': 0.3,
            }
        ).add_to(m)

    folium.LayerControl(position='topright', collapsed=False).add_to(m)

    # Renderizar mapa
    mapa_out = st_folium(
        m,
        height=560,
        width=None,
        returned_objects=["last_active_drawing", "all_drawings"],
        key="mapa_explorador"
    )

    # ── Capturar poligono dibujado ────────────────────────────────────────────
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
        st.session_state.poligono_explorador = poligono_nuevo

        # Calcular area
        try:
            from shapely.geometry import shape
            import geopandas as gpd
            geom_sh = shape(poligono_nuevo["geometry"])
            gdf_tmp = gpd.GeoDataFrame([{'geometry': geom_sh}], crs='EPSG:4326')
            area_ha = gdf_tmp.to_crs(epsg=32616).geometry.area.sum() / 10000
            c       = geom_sh.centroid

            st.success(
                f"Poligono dibujado | "
                f"Area: **{area_ha:.3f} ha** ({area_ha/MZ_TO_HA:.3f} mz) | "
                f"Centro: {c.y:.5f}N, {c.x:.5f}W"
            )

            # Construir GeoJSON con propiedades
            st.session_state.geojson_listo = {
                'type': 'FeatureCollection',
                'features': [{
                    **poligono_nuevo,
                    'properties': {
                        'nombre':       f'Zona {dept_exp}',
                        'area_ha':      round(area_ha, 4),
                        'departamento': dept_exp,
                        'anio':         anio_exp,
                    }
                }]
            }

        except Exception as e:
            st.info(f"Poligono capturado. {e}")

    # ── Guia de uso ──────────────────────────────────────────────────────────
    with st.expander("Como usar este explorador"):
        st.markdown("""
**Paso 1:** Selecciona el departamento y el año en el panel izquierdo

**Paso 2:** Presiona **"Generar capa de cafe"** — el mapa se coloreara
en azul donde hay cafe (~30-60 segundos)

**Paso 3:** Navega el mapa usando scroll o los botones +/-
- Cambia a **Satelite Google** para ver el terreno real
- Las zonas **azul oscuro** tienen alta probabilidad de cafe

**Paso 4:** Dibuja tu poligono con el icono de poligono (barra izquierda)
- Haz clic en cada esquina de tu finca
- Doble clic para cerrar

**Paso 5:** Descarga el GeoJSON con el boton del panel izquierdo
y subelo en **Analisis de Zona** para ver los resultados completos

---
**Capas disponibles en el control de capas:**
- Satelite Google / Hibrido / OpenStreetMap
- RGB Sentinel-2 (color natural)
- NDVI (verde = vegetacion)
- Zonas de cafe (azul = alta probabilidad)
        """)


# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Capa generada con Sentinel-2 SR Harmonized (ESA Copernicus) via Google Earth Engine | "
    "Score de cafe: NDVI [0.40-0.75] + EVI [0.20-0.50] + GNDVI + NDRE + SAVI + Altitud | "
    "Tesis UNAH"
)
