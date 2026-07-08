"""
gee_extractor.py — Extraccion real de indices satelitales Sentinel-2 via GEE
Implementacion alineada con Seccion 10.2.1 del informe de avance
"""

import ee
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import streamlit as st


# ── Parametros del informe ────────────────────────────────────────────────────
CLOUD_THRESH = 30      # Seccion 9.4
SG_WINDOW    = 7       # Seccion 10.2.1
SG_POLY      = 2
SCALE_FINCA  = 10      # Resolucion Sentinel-2 (metros)
INDICE_COLS  = ['NDVI', 'EVI', 'GNDVI', 'NDWI', 'SAVI', 'NDRE']


# ── Preprocesamiento Sentinel-2 ───────────────────────────────────────────────

def mask_s2_scl(image):
    """Mascara de nubes con SCL — Seccion 9.4."""
    scl  = image.select('SCL')
    mask = (scl.neq(3)
              .And(scl.neq(8))
              .And(scl.neq(9))
              .And(scl.neq(10)))
    return (image.updateMask(mask)
                 .divide(10000)
                 .copyProperties(image, ['system:time_start']))


def add_6_indices(image):
    """
    Calcula los 6 indices espectrales — Seccion 10.2.1 del informe.
    NDVI, EVI, GNDVI, NDWI, SAVI, NDRE
    """
    ndvi  = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    evi   = image.expression(
        '2.5*((NIR-RED)/(NIR+6.0*RED-7.5*BLUE+1.0))',
        {'NIR':image.select('B8'),
         'RED':image.select('B4'),
         'BLUE':image.select('B2')}
    ).rename('EVI')
    gndvi = image.normalizedDifference(['B8', 'B3']).rename('GNDVI')
    ndwi  = image.normalizedDifference(['B8', 'B11']).rename('NDWI')
    savi  = image.expression(
        '((NIR-RED)/(NIR+RED+0.5))*1.5',
        {'NIR':image.select('B8'),
         'RED':image.select('B4')}
    ).rename('SAVI')
    ndre  = image.normalizedDifference(['B8', 'B5']).rename('NDRE')
    return image.addBands([ndvi, evi, gndvi, ndwi, savi, ndre])


def get_s2_collection(geometry, start_date, end_date):
    """Coleccion Sentinel-2 SR Harmonized filtrada."""
    return (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
              .filterBounds(geometry)
              .filterDate(start_date, end_date)
              .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', CLOUD_THRESH))
              .map(mask_s2_scl)
              .map(add_6_indices))


def get_landsat_collection(geometry, start_date, end_date):
    """Landsat 8/9 como respaldo en periodos de alta nubosidad."""
    def mask_ls(image):
        qa = image.select('QA_PIXEL')
        return (image.updateMask(
                    qa.bitwiseAnd(1 << 3).eq(0)
                      .And(qa.bitwiseAnd(1 << 4).eq(0))
                ).multiply(0.0000275).add(-0.2)
                 .copyProperties(image, ['system:time_start']))

    def add_indices_ls(image):
        ndvi  = image.normalizedDifference(['B5', 'B4']).rename('NDVI')
        evi   = image.expression(
            '2.5*((NIR-RED)/(NIR+6.0*RED-7.5*BLUE+1.0))',
            {'NIR':image.select('B5'),
             'RED':image.select('B4'),
             'BLUE':image.select('B2')}
        ).rename('EVI')
        gndvi = image.normalizedDifference(['B5', 'B3']).rename('GNDVI')
        ndwi  = image.normalizedDifference(['B5', 'B6']).rename('NDWI')
        savi  = image.expression(
            '((NIR-RED)/(NIR+RED+0.5))*1.5',
            {'NIR':image.select('B5'),
             'RED':image.select('B4')}
        ).rename('SAVI')
        ndre  = ndvi.rename('NDRE')  # proxy (Landsat no tiene Red Edge)
        return image.addBands([ndvi, evi, gndvi, ndwi, savi, ndre])

    l8 = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
            .filterBounds(geometry).filterDate(start_date, end_date)
            .filter(ee.Filter.lt('CLOUD_COVER', CLOUD_THRESH))
            .map(mask_ls).map(add_indices_ls))
    l9 = (ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
            .filterBounds(geometry).filterDate(start_date, end_date)
            .filter(ee.Filter.lt('CLOUD_COVER', CLOUD_THRESH))
            .map(mask_ls).map(add_indices_ls))
    return l8.merge(l9)


# ── Extraccion de series temporales ──────────────────────────────────────────

def extraer_series_temporales(geometry, anio, progress_callback=None):
    """
    Extrae series temporales reales de Sentinel-2 sobre el poligono.

    Args:
        geometry: ee.Geometry del poligono dibujado
        anio: int, ano de analisis
        progress_callback: funcion para actualizar barra de progreso

    Returns:
        dict con DataFrames de indices y metadatos, o None si falla
    """
    start = f'{anio}-01-01'
    end   = f'{anio}-12-31'

    try:
        if progress_callback:
            progress_callback(0.1, "Cargando coleccion Sentinel-2...")

        col_s2 = get_s2_collection(geometry, start, end)
        n_s2   = col_s2.size().getInfo()

        if progress_callback:
            progress_callback(0.25, f"Sentinel-2: {n_s2} escenas disponibles...")

        # Usar Landsat como complemento si hay pocas escenas S2
        if n_s2 < 5:
            col_ls   = get_landsat_collection(geometry, start, end)
            n_ls     = col_ls.size().getInfo()
            col_usar = col_s2.merge(col_ls)
            fuente   = f'S2({n_s2}) + Landsat({n_ls})'
        else:
            col_usar = col_s2
            fuente   = f'Sentinel-2 ({n_s2} escenas)'

        if progress_callback:
            progress_callback(0.4, f"Extrayendo indices de {fuente}...")

        # Funcion de extraccion por imagen
        def extract_per_image(image):
            stats = image.select(INDICE_COLS).reduceRegion(
                reducer=ee.Reducer.mean()
                          .combine(ee.Reducer.stdDev(), sharedInputs=True),
                geometry=geometry,
                scale=SCALE_FINCA,
                maxPixels=1e9,
                bestEffort=True
            )
            return ee.Feature(None, stats.set(
                'fecha', image.date().format('YYYY-MM-dd')
            ))

        fc_results = col_usar.map(extract_per_image)

        if progress_callback:
            progress_callback(0.6, "Descargando resultados de GEE...")

        # Obtener resultados
        features = fc_results.getInfo()['features']

        if progress_callback:
            progress_callback(0.75, "Procesando series temporales...")

        # Construir DataFrame
        registros = []
        for feat in features:
            p = feat['properties']
            if p.get('NDVI_mean') is not None:
                registros.append(p)

        if not registros:
            return None, "No se encontraron observaciones validas para esta zona y periodo."

        df = pd.DataFrame(registros)
        df['fecha'] = pd.to_datetime(df['fecha'])
        df = df.sort_values('fecha').reset_index(drop=True)

        # Aplicar Savitzky-Golay a cada indice
        for idx in INDICE_COLS:
            col_raw = f'{idx}_mean'
            if col_raw in df.columns:
                s = df[col_raw].copy()
                s = s.interpolate(method='linear', limit_direction='both').ffill().bfill()
                if len(s.dropna()) >= SG_WINDOW:
                    df[f'{idx}_SG'] = savgol_filter(
                        s.values, window_length=SG_WINDOW, polyorder=SG_POLY
                    )
                else:
                    df[f'{idx}_SG'] = s.values

        if progress_callback:
            progress_callback(0.9, "Analisis espectral completado...")

        return df, fuente

    except Exception as e:
        return None, f"Error en extraccion GEE: {str(e)}"


# ── Altitud y pendiente SRTM ─────────────────────────────────────────────────

def get_elevacion(geometry):
    """Extrae estadisticos de elevacion y pendiente con SRTM 30m."""
    try:
        dem   = ee.Image('USGS/SRTMGL1_003')
        slope = ee.Terrain.slope(dem)

        stats = (dem.addBands(slope)
                   .rename(['elevacion', 'pendiente'])
                   .reduceRegion(
                       reducer=ee.Reducer.mean()
                                 .combine(ee.Reducer.min(),    sharedInputs=True)
                                 .combine(ee.Reducer.max(),    sharedInputs=True)
                                 .combine(ee.Reducer.stdDev(), sharedInputs=True),
                       geometry=geometry,
                       scale=30,
                       maxPixels=1e9
                   ).getInfo())

        return {
            'elev_mean':  float(stats.get('elevacion_mean',  0) or 0),
            'elev_min':   float(stats.get('elevacion_min',   0) or 0),
            'elev_max':   float(stats.get('elevacion_max',   0) or 0),
            'elev_std':   float(stats.get('elevacion_stdDev',0) or 0),
            'slope_mean': float(stats.get('pendiente_mean',  0) or 0),
        }
    except Exception as e:
        return {'elev_mean': 0, 'elev_min': 0,
                'elev_max': 0, 'elev_std': 0, 'slope_mean': 0}


# ── Clasificacion pixeles GEE ─────────────────────────────────────────────────

def clasificar_pixeles_gee(geometry, anio):
    """
    Clasifica uso de suelo pixel a pixel dentro del poligono.
    Usa puntos de referencia distribuidos geograficamente en Honduras.
    Retorna imagen clasificada y de probabilidad, o None si falla.
    """
    try:
        start = f'{anio}-01-01'
        end   = f'{anio}-12-31'
        dem   = ee.Image('USGS/SRTMGL1_003')
        slope = ee.Terrain.slope(dem)

        col_s2   = get_s2_collection(geometry, start, end)
        col_ls   = get_landsat_collection(geometry, start, end)
        n_s2     = col_s2.size().getInfo()
        col_usar = col_s2 if n_s2 >= 5 else col_s2.merge(col_ls)

        composito = (col_usar.median()
                             .addBands(dem.rename('elevacion'))
                             .addBands(slope.rename('pendiente'))
                             .clip(geometry))

        BANDAS = ['NDVI', 'EVI', 'GNDVI', 'NDWI', 'SAVI', 'NDRE',
                  'elevacion', 'pendiente']

        # Puntos de referencia fijos en Honduras (verificados)
        pts_cafe = ee.FeatureCollection([
            ee.Feature(ee.Geometry.Point([-87.982, 14.122]), {'clase': 1}),
            ee.Feature(ee.Geometry.Point([-87.975, 14.118]), {'clase': 1}),
            ee.Feature(ee.Geometry.Point([-88.240, 14.920]), {'clase': 1}),
            ee.Feature(ee.Geometry.Point([-88.875, 14.850]), {'clase': 1}),
            ee.Feature(ee.Geometry.Point([-87.640, 14.440]), {'clase': 1}),
            ee.Feature(ee.Geometry.Point([-86.780, 13.870]), {'clase': 1}),
        ])
        pts_bosque = ee.FeatureCollection([
            ee.Feature(ee.Geometry.Point([-86.500, 14.800]), {'clase': 2}),
            ee.Feature(ee.Geometry.Point([-86.200, 15.100]), {'clase': 2}),
            ee.Feature(ee.Geometry.Point([-87.100, 14.200]), {'clase': 2}),
            ee.Feature(ee.Geometry.Point([-88.500, 15.200]), {'clase': 2}),
        ])
        pts_pasto = ee.FeatureCollection([
            ee.Feature(ee.Geometry.Point([-87.200, 13.500]), {'clase': 3}),
            ee.Feature(ee.Geometry.Point([-87.500, 13.300]), {'clase': 3}),
            ee.Feature(ee.Geometry.Point([-88.800, 13.900]), {'clase': 3}),
        ])
        pts_cultivo = ee.FeatureCollection([
            ee.Feature(ee.Geometry.Point([-87.500, 15.400]), {'clase': 4}),
            ee.Feature(ee.Geometry.Point([-86.800, 15.600]), {'clase': 4}),
        ])
        pts_train = (pts_cafe.merge(pts_bosque)
                             .merge(pts_pasto)
                             .merge(pts_cultivo))

        # Imagen de referencia sin clip para muestrear toda Honduras
        ref_region = ee.Geometry.Rectangle([-90.0, 13.0, -83.0, 16.5])
        composito_ref = (get_s2_collection(ref_region, start, end)
                        .median()
                        .addBands(dem.rename('elevacion'))
                        .addBands(slope.rename('pendiente')))

        muestras = composito_ref.select(BANDAS).sampleRegions(
            collection=pts_train,
            properties=['clase'],
            scale=30,
            geometries=True
        )

        # Clasificador RF en GEE
        clf = (ee.Classifier.smileRandomForest(
            numberOfTrees=100, variablesPerSplit=3,
            minLeafPopulation=2, seed=42
        ).train(
            features=muestras,
            classProperty='clase',
            inputProperties=BANDAS
        ))

        img_clasif = (composito.select(BANDAS)
                               .classify(clf)
                               .clip(geometry))

        # Probabilidad de cafe
        clf_prob = (ee.Classifier.smileRandomForest(
            numberOfTrees=100, seed=42
        ).setOutputMode('MULTIPROBABILITY')
          .train(features=muestras,
                 classProperty='clase',
                 inputProperties=BANDAS))

        img_prob = (composito.select(BANDAS)
                             .classify(clf_prob)
                             .arrayGet(0)
                             .clip(geometry))

        return img_clasif, img_prob, composito

    except Exception as e:
        return None, None, None


def get_distribucion_clases(img_clasif, geometry, scale=10):
    """Calcula la distribucion de clases en pixeles y hectareas."""
    try:
        hist = img_clasif.reduceRegion(
            reducer=ee.Reducer.frequencyHistogram(),
            geometry=geometry,
            scale=scale,
            maxPixels=1e9
        ).getInfo().get('classification', {})

        total = sum(hist.values()) if hist else 1
        nombres = {1: 'Cafe', 2: 'Bosque', 3: 'Pasto', 4: 'Cultivo anual'}
        colores = {1: '#8B5E3C', 2: '#2d6a4f', 3: '#f4d03f', 4: '#e67e22'}

        dist = []
        for cid_str, npx in sorted(hist.items(), key=lambda x: int(float(x[0]))):
            cid = int(float(cid_str))
            ha  = (npx * scale * scale) / 10000
            pct = npx / total * 100
            dist.append({
                'clase':  nombres.get(cid, f'Clase {cid}'),
                'pixeles': npx,
                'ha':      round(ha,  3),
                'pct':     round(pct, 2),
                'color':   colores.get(cid, '#888888'),
            })
        return dist
    except:
        return []
