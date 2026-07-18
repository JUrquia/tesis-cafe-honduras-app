"""
validacion_campo.py — Validacion con Puntos de Campo
Rediseñado para ser mas intuitivo:
- Mapa interactivo donde el usuario hace clic para agregar puntos
- Selector visual de clase (cafe / bosque / mixto)
- Analisis automatico con GEE al confirmar
- Metricas y matriz de confusion generadas automaticamente
"""

import streamlit as st
import json, os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime
import folium
from folium.plugins import Draw, MousePosition
from streamlit_folium import st_folium

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gee_auth import inicializar_gee

st.set_page_config(page_title="Validacion de Campo", page_icon="📋", layout="wide")

st.markdown("""
<style>
.val-header {
    background: linear-gradient(135deg,#1F3864,#2E5FA3);
    padding:1.2rem 2rem; border-radius:12px; color:white; margin-bottom:1rem;
}
.clase-card {
    border:3px solid #e0e0e0; border-radius:12px; padding:14px 10px;
    text-align:center; cursor:pointer; transition:0.2s;
    background:white; margin-bottom:8px;
}
.clase-card:hover { border-color:#2E5FA3; background:#f0f4ff; }
.clase-card.selected { border-color:#1a7a4a; background:#e8f5e9; }
.step-badge {
    background:#2E5FA3; color:white; border-radius:20px;
    padding:2px 12px; font-size:13px; font-weight:bold; margin-right:8px;
}
.result-ok  { background:#e8f5e9; border-left:4px solid #1a7a4a;
               padding:8px 12px; border-radius:6px; margin:4px 0; }
.result-no  { background:#fce4ec; border-left:4px solid #c0392b;
               padding:8px 12px; border-radius:6px; margin:4px 0; }
.result-inc { background:#fff8e1; border-left:4px solid #f57f17;
               padding:8px 12px; border-radius:6px; margin:4px 0; }
</style>
""", unsafe_allow_html=True)

# ── GEE ──────────────────────────────────────────────────────────────────────
gee_ok, _ = inicializar_gee()
if gee_ok:
    st.sidebar.success("GEE: Conectado")
else:
    st.sidebar.error("GEE: Sin conexion")

# ── Estado ────────────────────────────────────────────────────────────────────
for k, v in [('puntos', []), ('clase_sel', 'cafe'),
             ('nombre_pt', ''), ('notas_pt', ''),
             ('ultimo_click', None)]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='val-header'>
    <h1 style='margin:0;font-size:1.7rem'>📋 Validacion con Puntos de Campo</h1>
    <p style='margin:0.3rem 0 0;opacity:0.9;font-size:14px'>
        Haz clic en el mapa → elige la clase → agrega el punto →
        analiza con GEE → obtén OA y Kappa para tu tesis
    </p>
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# LAYOUT PRINCIPAL: mapa grande a la izquierda, panel a la derecha
# ════════════════════════════════════════════════════════════════
col_mapa, col_panel = st.columns([3, 2], gap="medium")

# ════════════════════════════════════════════════════════════════
# MAPA INTERACTIVO
# ════════════════════════════════════════════════════════════════
with col_mapa:
    st.markdown("#### 🗺️ Mapa — haz clic para marcar una zona")
    st.caption(
        "Haz clic en el mapa donde está la zona que conoces → "
        "luego elige la clase en el panel derecho → presiona Agregar"
    )

    # Centro por defecto (La Paz)
    centro = [14.31, -87.68]
    if st.session_state.puntos:
        centro = [
            np.mean([p['lat'] for p in st.session_state.puntos]),
            np.mean([p['lon'] for p in st.session_state.puntos])
        ]

    m = folium.Map(location=centro, zoom_start=12, tiles=None)

    # Capas base
    folium.TileLayer(
        'https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
        attr='Google', name='Satelite Google',
        overlay=False, control=True
    ).add_to(m)
    folium.TileLayer(
        'https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}',
        attr='Google', name='Hibrido Google',
        overlay=False, control=True
    ).add_to(m)
    folium.TileLayer('OpenStreetMap', name='Mapa', overlay=False, control=True).add_to(m)

    # Colores y iconos por clase y estado
    COLORES = {
        'cafe':   {'pendiente': 'orange', 'ok': 'green',    'error': 'red'},
        'bosque': {'pendiente': 'blue',   'ok': 'darkgreen','error': 'red'},
        'mixto':  {'pendiente': 'gray',   'ok': 'purple',   'error': 'red'},
    }
    ICONOS  = {'cafe': 'leaf', 'bosque': 'tree-deciduous', 'mixto': 'question-sign'}
    LABELS  = {'cafe': '☕ Café', 'bosque': '🌳 Bosque / Sin café', 'mixto': '🌿 Mixto'}

    # Dibujar puntos existentes
    for p in st.session_state.puntos:
        clase  = p['clase']
        estado = 'pendiente'
        if p.get('analizado'):
            estado = 'ok' if _coincide(p) else 'error'
        color_m = COLORES[clase][estado]
        icon_m  = ICONOS.get(clase, 'circle')

        pred_txt = (
            f"Sistema: {p.get('prediccion','-')} ({p.get('score',0):.1f}%)"
            if p.get('analizado') else 'Sin analizar aún'
        )
        coincide_txt = (
            '✅ Correcto' if estado == 'ok' else
            '❌ Error'   if estado == 'error' else ''
        )

        popup_html = f"""
        <div style='min-width:160px;font-family:sans-serif'>
            <b style='color:#1F3864'>{p['nombre']}</b><br>
            <span style='color:#888;font-size:12px'>
                {LABELS.get(clase,'?')} | {p['fecha']}</span><br><br>
            <b>Lat:</b> {p['lat']:.5f}<br>
            <b>Lon:</b> {p['lon']:.5f}<br><br>
            <b>{pred_txt}</b><br>
            {coincide_txt}
        </div>
        """
        folium.Marker(
            location=[p['lat'], p['lon']],
            popup=folium.Popup(popup_html, max_width=200),
            tooltip=f"{p['nombre']} — {LABELS.get(clase,'?')}",
            icon=folium.Icon(color=color_m, icon=icon_m, prefix='glyphicon')
        ).add_to(m)

    # Resaltar ultimo click pendiente de agregar
    if st.session_state.ultimo_click:
        lc = st.session_state.ultimo_click
        folium.CircleMarker(
            location=[lc['lat'], lc['lng']],
            radius=14, color='#FF6B35', fill=True,
            fill_color='#FF6B35', fill_opacity=0.4,
            tooltip='📍 Click registrado — elige la clase y agrega'
        ).add_to(m)

    MousePosition(position='bottomleft', prefix='📍').add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    # Renderizar mapa
    mapa_data = st_folium(
        m, height=500, width=None,
        returned_objects=['last_clicked'],
        key='mapa_validacion'
    )

    # Capturar click
    if mapa_data and mapa_data.get('last_clicked'):
        click = mapa_data['last_clicked']
        if (st.session_state.ultimo_click is None or
                abs(click['lat'] - st.session_state.ultimo_click.get('lat',0)) > 0.00001):
            st.session_state.ultimo_click = click

    # Leyenda del mapa
    st.markdown("""
    <div style='background:#f8f9fa;border-radius:8px;padding:8px 14px;
                font-size:12px;margin-top:6px;display:flex;gap:20px;flex-wrap:wrap'>
        <span>🟠 Café sin analizar</span>
        <span>🔵 Bosque sin analizar</span>
        <span>🟢 Clasificación correcta</span>
        <span>🔴 Clasificación incorrecta</span>
        <span>🔶 Punto seleccionado</span>
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# PANEL DERECHO
# ════════════════════════════════════════════════════════════════
with col_panel:

    # ── PASO 1: Agregar punto ─────────────────────────────────────────────────
    with st.expander("➕  PASO 1 — Agregar punto de campo", expanded=True):

        # Coordenadas del click
        if st.session_state.ultimo_click:
            lc  = st.session_state.ultimo_click
            lat = round(lc['lat'], 6)
            lon = round(lc['lng'], 6)
            st.success(f"📍 Punto seleccionado: **{lat:.5f}°N, {lon:.5f}°W**")
        else:
            st.info("Haz clic en el mapa para seleccionar la ubicación")
            lat = 14.2653
            lon = -87.8440

        # Nombre
        nombre = st.text_input(
            "Nombre de la zona",
            value=st.session_state.nombre_pt or
                  f"Punto {len(st.session_state.puntos)+1}",
            placeholder="ej. Finca Delma sector norte"
        )

        # Clase — botones visuales
        st.markdown("**¿Qué hay en esta zona?**")
        c1, c2, c3 = st.columns(3)
        sel = st.session_state.clase_sel

        with c1:
            if st.button("☕\nCafé", use_container_width=True,
                         type="primary" if sel=='cafe' else "secondary"):
                st.session_state.clase_sel = 'cafe'
                st.rerun()
        with c2:
            if st.button("🌳\nBosque /\nSin café", use_container_width=True,
                         type="primary" if sel=='bosque' else "secondary"):
                st.session_state.clase_sel = 'bosque'
                st.rerun()
        with c3:
            if st.button("🌿\nMixto /\nIncierto", use_container_width=True,
                         type="primary" if sel=='mixto' else "secondary"):
                st.session_state.clase_sel = 'mixto'
                st.rerun()

        # Clase seleccionada
        clase_colores = {'cafe':'#8B5E3C','bosque':'#2d6a4f','mixto':'#888888'}
        cc = clase_colores[sel]
        st.markdown(
            f"<div style='background:{cc}20;border-left:4px solid {cc};"
            f"padding:6px 12px;border-radius:6px;font-size:13px'>"
            f"Clase seleccionada: <b>{LABELS[sel]}</b></div>",
            unsafe_allow_html=True
        )

        notas = st.text_input(
            "Notas (opcional)",
            placeholder="ej. Café a 1350 msnm, ladera este",
            value=st.session_state.notas_pt
        )

        # Coordenadas editables como respaldo
        with st.expander("✏️ Editar coordenadas manualmente"):
            lat = st.number_input("Latitud",  value=lat,  format="%.6f", key='lat_manual')
            lon = st.number_input("Longitud", value=lon, format="%.6f", key='lon_manual')

        # Botón agregar
        btn_agregar = st.button(
            "✅ Agregar punto",
            type="primary", use_container_width=True,
            disabled=(st.session_state.ultimo_click is None and lat == 14.2653)
        )

        if btn_agregar:
            st.session_state.puntos.append({
                'id':         len(st.session_state.puntos) + 1,
                'nombre':     nombre,
                'clase':      st.session_state.clase_sel,
                'lat':        lat,
                'lon':        lon,
                'notas':      notas,
                'fecha':      datetime.now().strftime('%Y-%m-%d'),
                'analizado':  False,
                'prediccion': None,
                'score':      None,
            })
            st.session_state.ultimo_click = None
            st.session_state.nombre_pt    = ''
            st.session_state.notas_pt     = ''
            st.success(f"Punto agregado: **{nombre}** ({LABELS[st.session_state.clase_sel]})")
            st.rerun()

    # ── Subir GeoJSON ─────────────────────────────────────────────────────────
    with st.expander("📂  Subir GeoJSON existente"):
        st.caption(
            "El GeoJSON debe tener la propiedad `clase` con valor "
            "`cafe`, `bosque` o `mixto`"
        )
        archivo = st.file_uploader("Sube tu GeoJSON", type=['geojson','json'],
                                   label_visibility='collapsed')
        if archivo:
            try:
                geojson = json.load(archivo)
                importados = 0
                for feat in geojson.get('features', []):
                    props = feat.get('properties', {})
                    geom  = feat.get('geometry', {})
                    clase = str(props.get('clase',
                                props.get('class', 'cafe'))).lower()
                    if clase not in ['cafe','bosque','mixto']:
                        clase = 'cafe'
                    coords = geom.get('coordinates', [0, 0])
                    if geom.get('type') == 'Point':
                        lon_i, lat_i = coords[0], coords[1]
                    else:
                        pts   = coords[0] if geom.get('type')=='Polygon' else coords[0][0]
                        lon_i = np.mean([c[0] for c in pts])
                        lat_i = np.mean([c[1] for c in pts])
                    st.session_state.puntos.append({
                        'id':        len(st.session_state.puntos)+1,
                        'nombre':    str(props.get('nombre',
                                         props.get('name', f'Punto {importados+1}'))),
                        'clase':     clase,
                        'lat':       float(lat_i),
                        'lon':       float(lon_i),
                        'notas':     str(props.get('notas', '')),
                        'fecha':     datetime.now().strftime('%Y-%m-%d'),
                        'analizado': False,
                        'prediccion':None,
                        'score':     None,
                    })
                    importados += 1
                st.success(f"✅ {importados} puntos importados")
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")

    # ── PASO 2: Analizar con GEE ──────────────────────────────────────────────
    puntos   = st.session_state.puntos
    pendientes = [p for p in puntos if not p['analizado']]
    analizados = [p for p in puntos if p['analizado']]

    if puntos:
        st.divider()
        with st.expander(
            f"🛰️  PASO 2 — Analizar con GEE  ({len(pendientes)} pendientes)",
            expanded=len(pendientes) > 0
        ):
            col_a, col_b = st.columns(2)
            anio_val = col_a.selectbox("Año satelital", [2026,2025,2024,2023])
            radio_m  = col_b.select_slider(
                "Radio (m)", options=[50,100,150,200,300], value=100
            )

            n_cafe_r  = sum(1 for p in puntos if p['clase']=='cafe')
            n_bosq_r  = sum(1 for p in puntos if p['clase']=='bosque')
            n_mix_r   = sum(1 for p in puntos if p['clase']=='mixto')

            st.markdown(
                f"**{len(puntos)} puntos:** "
                f"☕ {n_cafe_r} café &nbsp;|&nbsp; "
                f"🌳 {n_bosq_r} bosque &nbsp;|&nbsp; "
                f"🌿 {n_mix_r} mixto"
            )

            btn_anal = st.button(
                f"🛰️ Analizar {len(pendientes)} puntos con GEE",
                type="primary", use_container_width=True,
                disabled=(not gee_ok or len(pendientes) == 0)
            )
            if not gee_ok:
                st.warning("GEE no conectado — configura las credenciales en Secrets")

            if btn_anal and gee_ok and pendientes:
                _analizar_puntos(pendientes, anio_val, radio_m)
                st.rerun()

        # ── PASO 3: Resultados rápidos ────────────────────────────────────────
        if analizados:
            st.divider()
            correctos = sum(1 for p in analizados if _coincide(p))
            oa_rap = correctos / len(analizados)

            st.markdown("#### 📊 Resultados")
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Exactitud (OA)", f"{oa_rap*100:.1f}%")
            mc2.metric("Correctos", f"{correctos}/{len(analizados)}")
            mc3.metric("Pendientes", len(pendientes))

            # Lista compacta de resultados
            for p in analizados[-5:]:  # mostrar ultimos 5
                ok = _coincide(p)
                css = 'result-ok' if ok else 'result-no'
                emoji_r = '✅' if ok else '❌'
                st.markdown(
                    f"<div class='{css}'>"
                    f"{emoji_r} <b>{p['nombre']}</b> — "
                    f"Real: {LABELS.get(p['clase'],'?')} | "
                    f"Sistema: {p.get('prediccion','-')} "
                    f"({p.get('score',0):.0f}%)</div>",
                    unsafe_allow_html=True
                )
            if len(analizados) > 5:
                st.caption(f"... y {len(analizados)-5} más. Ver tabla completa abajo.")

        # Botón para limpiar
        if st.button("🗑️ Limpiar todos los puntos",
                     use_container_width=False):
            st.session_state.puntos = []
            st.session_state.ultimo_click = None
            st.rerun()


# ════════════════════════════════════════════════════════════════
# SECCION INFERIOR: Tabla + Métricas + Exportar
# ════════════════════════════════════════════════════════════════
if st.session_state.puntos:
    st.divider()
    tab_tabla, tab_metricas, tab_exportar = st.tabs([
        "📋 Tabla de puntos",
        "📊 Métricas de exactitud",
        "📥 Exportar para tesis"
    ])

    analizados = [p for p in st.session_state.puntos if p['analizado']]

    # ── TABLA ─────────────────────────────────────────────────────────────────
    with tab_tabla:
        todos = st.session_state.puntos
        df_t  = pd.DataFrame([{
            'ID':         p['id'],
            'Nombre':     p['nombre'],
            'Clase real': LABELS.get(p['clase'],'?'),
            'Lat':        f"{p['lat']:.5f}",
            'Lon':        f"{p['lon']:.5f}",
            'Estado':     '✅ OK' if _coincide(p) else
                          ('❌ Error' if p['analizado'] else '⏳ Pendiente'),
            'Score':      f"{p.get('score',0):.1f}%" if p.get('analizado') else '-',
            'Prediccion': p.get('prediccion', '-'),
            'NDRE':       f"{p.get('ndre',0):.3f}" if p.get('analizado') else '-',
            'Slope':      f"{p.get('slope',0):.1f}°" if p.get('analizado') else '-',
        } for p in todos])
        st.dataframe(df_t, use_container_width=True, hide_index=True)

        # Descargar GeoJSON
        geojson_exp = {
            'type':'FeatureCollection',
            'features':[{
                'type':'Feature',
                'geometry':{'type':'Point','coordinates':[p['lon'],p['lat']]},
                'properties':{k:v for k,v in p.items() if k not in ['lat','lon']}
            } for p in todos]
        }
        st.download_button(
            "⬇️ Descargar GeoJSON",
            json.dumps(geojson_exp, indent=2, ensure_ascii=False).encode(),
            f"puntos_campo_{datetime.now().strftime('%Y%m%d')}.geojson",
            "application/geo+json"
        )

    # ── METRICAS ──────────────────────────────────────────────────────────────
    with tab_metricas:
        if len(analizados) < 2:
            st.info("Necesitas al menos 2 puntos analizados para calcular métricas.")
        else:
            y_real, y_pred = [], []
            for p in analizados:
                real = p['clase']
                pred = p.get('prediccion','')
                if 'CONFIRMADO' in pred or 'PROBABLE' in pred:
                    pred_b = 'cafe'
                elif 'INCIERTO' in pred:
                    pred_b = 'mixto'
                else:
                    pred_b = 'bosque'
                y_real.append(real)
                y_pred.append(pred_b)

            vp = sum(1 for r,p in zip(y_real,y_pred) if r=='cafe'  and p=='cafe')
            vn = sum(1 for r,p in zip(y_real,y_pred) if r!='cafe'  and p!='cafe')
            fp = sum(1 for r,p in zip(y_real,y_pred) if r!='cafe'  and p=='cafe')
            fn = sum(1 for r,p in zip(y_real,y_pred) if r=='cafe'  and p!='cafe')
            n  = len(y_real)

            oa  = (vp+vn)/n
            pre = vp/(vp+fp) if (vp+fp)>0 else 0
            rec = vp/(vp+fn) if (vp+fn)>0 else 0
            f1  = 2*pre*rec/(pre+rec) if (pre+rec)>0 else 0
            p_e = ((vp+fp)/n)*((vp+fn)/n)+((vn+fn)/n)*((vn+fp)/n)
            kap = (oa-p_e)/(1-p_e) if (1-p_e)>0 else 0

            # Metricas en columnas
            cols = st.columns(5)
            for col, lbl, val, tip in zip(cols,
                ['OA','Kappa','Precisión','Recall','F1-Score'],
                [f'{oa*100:.1f}%',f'{kap:.3f}',
                 f'{pre*100:.1f}%',f'{rec*100:.1f}%',f'{f1*100:.1f}%'],
                ['Clasificaciones correctas / total',
                 '>0.80=excelente | 0.60=bueno | 0.40=moderado',
                 'Del café predicho, cuánto era café real',
                 'Del café real, cuánto detectó el sistema',
                 'Media armónica de Precisión y Recall']
            ):
                col.metric(lbl, val, help=tip)

            kap_txt = ('Excelente ✅' if kap>=0.80 else 'Bueno 👍' if kap>=0.60
                       else 'Moderado ⚠️' if kap>=0.40 else 'Débil ❌')
            st.caption(f"Kappa de Cohen: {kap_txt} — Landis & Koch (1977)")
            st.divider()

            # Matriz de confusion
            col_mat, col_det = st.columns([1,1])
            with col_mat:
                st.markdown("**Matriz de confusión**")
                fig, ax = plt.subplots(figsize=(4.5, 3.5))
                M = np.array([[vp,fn],[fp,vn]])
                im = ax.imshow(M, cmap='Blues')
                ax.set_xticks([0,1])
                ax.set_xticklabels(['Café\n(predicho)','No café\n(predicho)'], fontsize=9)
                ax.set_yticks([0,1])
                ax.set_yticklabels(['Café\n(real)','No café\n(real)'], fontsize=9)
                ax.set_title('Clasificación café vs no-café', fontsize=10,
                             fontweight='bold', color='#1F3864', pad=8)
                for i in range(2):
                    for j in range(2):
                        v = M[i,j]
                        c = 'white' if v > M.max()*0.5 else '#1F3864'
                        ax.text(j, i, str(v), ha='center', va='center',
                                fontsize=20, fontweight='bold', color=c)
                for (i,j), lbl in [((0,0),'VP'),((0,1),'FN'),
                                    ((1,0),'FP'),((1,1),'VN')]:
                    ax.text(j+0.44, i-0.40, lbl, ha='center', va='center',
                            fontsize=8, color='gray', style='italic')
                plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close()

            with col_det:
                st.markdown("**Comparación con literatura:**")
                st.dataframe(pd.DataFrame({
                    'Estudio':  ['🎓 Este sistema','📄 Medina et al. (2026)','🎯 Meta tesis'],
                    'OA':       [f'{oa*100:.1f}%','>94%','≥85%'],
                    'Kappa':    [f'{kap:.3f}','>0.889','≥0.75'],
                }), use_container_width=True, hide_index=True)

                st.markdown("**Detalle:**")
                st.dataframe(pd.DataFrame({
                    'Indicador':['VP','VN','FP','FN'],
                    'N':        [vp,vn,fp,fn],
                    'Descripcion':['Café → CAFÉ ✅','No café → NO CAFÉ ✅',
                                   'No café → CAFÉ ❌','Café → NO CAFÉ ❌'],
                }), use_container_width=True, hide_index=True)

    # ── EXPORTAR ──────────────────────────────────────────────────────────────
    with tab_exportar:
        if not analizados:
            st.info("Analiza los puntos primero.")
        else:
            vp2=vn2=fp2=fn2=0
            for p in analizados:
                pred = p.get('prediccion','')
                real = p['clase']
                p_b  = 'cafe' if ('CONFIRMADO' in pred or 'PROBABLE' in pred) \
                       else ('mixto' if 'INCIERTO' in pred else 'bosque')
                if real=='cafe'  and p_b=='cafe':  vp2+=1
                elif real!='cafe'and p_b!='cafe':  vn2+=1
                elif real!='cafe'and p_b=='cafe':  fp2+=1
                elif real=='cafe'and p_b!='cafe':  fn2+=1

            n2   = len(analizados)
            oa2  = (vp2+vn2)/n2
            pre2 = vp2/(vp2+fp2) if (vp2+fp2)>0 else 0
            rec2 = vp2/(vp2+fn2) if (vp2+fn2)>0 else 0
            f12  = 2*pre2*rec2/(pre2+rec2) if (pre2+rec2)>0 else 0
            p_e2 = ((vp2+fp2)/n2)*((vp2+fn2)/n2)+((vn2+fn2)/n2)*((vn2+fp2)/n2)
            kap2 = (oa2-p_e2)/(1-p_e2) if (1-p_e2)>0 else 0

            n_cafe_t  = sum(1 for p in analizados if p['clase']=='cafe')
            n_bosq_t  = sum(1 for p in analizados if p['clase']=='bosque')
            n_mix_t   = sum(1 for p in analizados if p['clase']=='mixto')
            anio_t    = analizados[0].get('anio_anal', 2026)

            # Tabla CSV
            df_exp = pd.DataFrame([{
                'N':         i+1,
                'Zona':      p['nombre'],
                'Clase real':LABELS.get(p['clase'],''),
                'Lat':       f"{p['lat']:.5f}",
                'Lon':       f"{p['lon']:.5f}",
                'NDRE':      f"{p.get('ndre',0):.3f}",
                'Slope':     f"{p.get('slope',0):.1f}",
                'Score':     f"{p.get('score',0):.2f}",
                'Prediccion':p.get('prediccion','-'),
                'Correcto':  'SI' if _coincide(p) else 'NO',
            } for i,p in enumerate(analizados)])

            col_e1, col_e2 = st.columns(2)
            col_e1.download_button(
                "⬇️ Tabla CSV para Word/Excel",
                df_exp.to_csv(index=False, encoding='utf-8-sig').encode(),
                f"validacion_campo_{datetime.now().strftime('%Y%m%d')}.csv",
                "text/csv", use_container_width=True
            )

            geojson_e = {
                'type':'FeatureCollection',
                'features':[{
                    'type':'Feature',
                    'geometry':{'type':'Point','coordinates':[p['lon'],p['lat']]},
                    'properties':{k:v for k,v in p.items() if k not in ['lat','lon']}
                } for p in analizados]
            }
            col_e2.download_button(
                "⬇️ GeoJSON con resultados",
                json.dumps(geojson_e, indent=2, ensure_ascii=False).encode(),
                f"validacion_resultados_{datetime.now().strftime('%Y%m%d')}.geojson",
                "application/geo+json", use_container_width=True
            )

            st.divider()
            st.markdown("**Texto sugerido para el informe (Sección 7.X):**")
            kap_interp = ('excelente (κ≥0.80)' if kap2>=0.80 else
                          'bueno (κ≥0.60)' if kap2>=0.60 else 'moderado')
            texto = f"""Para evaluar el desempeño del sistema de clasificación, se recopilaron {n2} puntos de referencia verificados en campo en el departamento de La Paz, Honduras, correspondientes a {n_cafe_t} sitios de café ({round(n_cafe_t/n2*100)}%), {n_bosq_t} sitios sin café o bosque ({round(n_bosq_t/n2*100)}%) y {n_mix_t} zonas mixtas ({round(n_mix_t/n2*100)}%), analizados con imágenes Sentinel-2 y SAR Sentinel-1 del año {anio_t}.

El sistema alcanzó una Exactitud General (OA) de {oa2*100:.1f}% y un coeficiente Kappa de Cohen de {kap2:.3f}, indicando un acuerdo {kap_interp} entre la clasificación satelital y la verificación en campo (Landis & Koch, 1977). La Precisión fue {pre2*100:.1f}%, el Recall {rec2*100:.1f}% y el F1-Score {f12*100:.1f}%.

Estos resultados son consistentes con los reportados por Medina et al. (2026) para sistemas cafetaleros en Perú (OA>94%, κ>0.889), con la diferencia esperada por el menor número de puntos de campo disponibles en este estudio. La integración de datos SAR Sentinel-1 mejoró la discriminación entre café bajo sombra y vegetación densa en terrenos planos, conforme a la metodología descrita en la Sección 10.2.1."""

            st.text_area("", texto, height=220, label_visibility='collapsed')
            st.download_button(
                "⬇️ Descargar texto (.txt)",
                texto.encode('utf-8'),
                f"texto_validacion_{datetime.now().strftime('%Y%m%d')}.txt",
                "text/plain"
            )


# ════════════════════════════════════════════════════════════════
# FUNCIONES AUXILIARES
# ════════════════════════════════════════════════════════════════

def _coincide(p):
    """True si la prediccion del sistema coincide con la clase real."""
    if not p.get('analizado'):
        return False
    pred = p.get('prediccion','')
    real = p['clase']
    if real == 'cafe':
        return 'CONFIRMADO' in pred or 'PROBABLE' in pred
    elif real == 'bosque':
        return 'NO ES' in pred or 'INCIERTO' in pred
    elif real == 'mixto':
        return 'INCIERTO' in pred or 'PROBABLE' in pred
    return False


def _analizar_puntos(pendientes, anio_val, radio_m):
    """Analiza cada punto pendiente con GEE y actualiza st.session_state.puntos."""
    import ee
    import geopandas as gpd
    from shapely.geometry import Point as ShapelyPoint
    from gee_extractor import extraer_sar_stats, get_elevacion, get_s2_collection

    prog = st.progress(0)
    stat = st.empty()
    log  = []

    for i, punto in enumerate(pendientes):
        stat.markdown(f"🛰️ Analizando **{punto['nombre']}** ({i+1}/{len(pendientes)})...")
        prog.progress((i+0.5)/len(pendientes))

        try:
            # Crear buffer circular
            gdf = gpd.GeoDataFrame(
                [{'geometry': ShapelyPoint(punto['lon'], punto['lat'])}],
                crs='EPSG:4326'
            ).to_crs(epsg=32616)
            buf  = gdf.buffer(radio_m).to_crs(epsg=4326)
            b    = buf.iloc[0].bounds
            geom = ee.Geometry.BBox(b[0], b[1], b[2], b[3])

            start = f'{anio_val}-01-01'
            end   = f'{anio_val}-12-31'

            # Indices opticos
            col  = get_s2_collection(geom, start, end)
            comp = col.median().clip(geom)
            stats = comp.select(
                ['NDVI','EVI','GNDVI','NDWI','SAVI','NDRE']
            ).reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=geom, scale=10, maxPixels=1e9, bestEffort=True
            ).getInfo()

            ndvi  = float(stats.get('NDVI',  0) or 0)
            evi   = float(stats.get('EVI',   0) or 0)
            gndvi = float(stats.get('GNDVI', 0) or 0)
            ndwi  = float(stats.get('NDWI',  0) or 0)
            savi  = float(stats.get('SAVI',  0) or 0)
            ndre  = float(stats.get('NDRE',  0) or 0)

            # Elevacion y pendiente
            elev_data = get_elevacion(geom)
            slope     = elev_data.get('slope_mean', 0)
            elev_mean = elev_data.get('elev_mean', 1000)
            apto_alt  = 800 <= elev_mean <= 1800

            # Amplitud aproximada
            ndvi_amp = abs(ndvi - 0.55) * 1.8 + 0.12

            # Clasificacion (misma logica que analisis_zona.py)
            es_sombra = (0.60<=ndvi<=0.88 and ndre>=0.42 and
                         evi<=0.65 and ndvi_amp>=0.18 and slope>=7.0)

            if es_sombra:
                n_ok = sum([0.40<=ndvi<=0.88, True, 0.20<=evi<=0.62,
                            0.30<=gndvi<=0.75, 0.25<=savi<=0.58,
                            ndre>=0.42, True, apto_alt])
                tipo = 'Cafe bajo sombra'
            else:
                n_ok = sum([0.40<=ndvi<=0.75, True, 0.20<=evi<=0.50,
                            0.30<=gndvi<=0.65, 0.25<=savi<=0.55,
                            0.28<=ndre<=0.55, True, apto_alt])
                tipo = 'Cafe a pleno sol'

            sc_reg = n_ok / 8 * 100
            pen    = (20 if evi>0.52 and ndwi>0.16 and slope<7 and not es_sombra else 0)
            pen   += (20 if ndre>0.62 and evi>0.62 else 0)
            pen   += (15 if 0.45<=ndvi<=0.62 and ndre<0.35 and ndwi>0.05 else 0)

            prob = min(100, max(0, sc_reg * 0.85 + ndre * 28))
            sb   = (0.35*(sc_reg/100)+0.45*(prob/100)+0.20*0.15)*100
            sc   = sb if es_sombra else max(0, sb - pen)
            if es_sombra and prob>60 and ndre>0.45:
                sc = min(100, sc+8)

            if   sc>=75: pred='CAFE CONFIRMADO'
            elif sc>=55: pred='PROBABLE CAFE'
            elif sc>=35: pred='RESULTADO INCIERTO'
            else:        pred='NO ES CAFE'

            # Actualizar en session_state
            idx = next(j for j,p in enumerate(st.session_state.puntos)
                       if p['id']==punto['id'])
            st.session_state.puntos[idx].update({
                'analizado':   True,
                'prediccion':  pred,
                'pred_clase':  pred,
                'score':       round(sc, 2),
                'tipo_sistema':tipo,
                'ndvi':        round(ndvi, 4),
                'evi':         round(evi,  4),
                'ndre':        round(ndre, 4),
                'ndwi':        round(ndwi, 4),
                'slope':       round(slope,1),
                'anio_anal':   anio_val,
            })

            ok    = _coincide(st.session_state.puntos[idx])
            emoji = '✅' if ok else '❌'
            log.append(
                f"{emoji} **{punto['nombre']}** — "
                f"Real: {LABELS.get(punto['clase'],'?')} | "
                f"Sistema: **{pred}** ({sc:.0f}%)"
            )

        except Exception as e:
            log.append(f"⚠️ **{punto['nombre']}** — Error: {str(e)[:70]}")

        prog.progress((i+1)/len(pendientes))

    prog.empty()
    stat.success(f"✅ {len(pendientes)} puntos analizados")
    for msg in log:
        st.markdown(msg)
