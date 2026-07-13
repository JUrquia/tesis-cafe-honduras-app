"""
predictor_real.py — Prediccion de rendimiento con modelos RF+XGB reales
Carga los modelos .pkl entrenados con datos IHCAFE 2021-2025
Si no existen los .pkl, usa formula calibrada como fallback
"""

import os, json, pickle
import numpy as np
import pandas as pd
import streamlit as st

MODELOS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'modelos')
W_RF  = 0.55
W_XGB = 0.45

IHCAFE_REF = {
    'Comayagua':    21.86, 'Copan':   26.89, 'El Paraiso': 15.80,
    'La Paz':       20.05, 'Santa Barbara': 19.39,
}


@st.cache_resource(show_spinner="Cargando modelos ML entrenados...")
def cargar_modelos():
    """
    Carga modelos .pkl. Retorna (rf, xgb, scaler, features, metadata, modo).
    modo = 'real' con .pkl, 'demo' sin ellos.
    """
    paths = {
        'rf':       os.path.join(MODELOS_DIR, 'modelo_rf.pkl'),
        'xgb':      os.path.join(MODELOS_DIR, 'modelo_xgb.pkl'),
        'scaler':   os.path.join(MODELOS_DIR, 'scaler.pkl'),
        'features': os.path.join(MODELOS_DIR, 'features_lista.pkl'),
        'meta':     os.path.join(MODELOS_DIR, 'metadata_modelo.json'),
    }
    requeridos = ['rf', 'xgb', 'scaler', 'features']
    faltantes  = [k for k in requeridos if not os.path.exists(paths[k])]

    if faltantes:
        return None, None, None, None, {'modo':'demo','razon':f'Faltan: {faltantes}'}, 'demo'

    try:
        with open(paths['rf'],  'rb') as f: rf_m  = pickle.load(f)
        with open(paths['xgb'], 'rb') as f: xgb_m = pickle.load(f)
        with open(paths['scaler'],   'rb') as f: sc   = pickle.load(f)
        with open(paths['features'], 'rb') as f: feats= pickle.load(f)
        meta = {'modo':'real'}
        if os.path.exists(paths['meta']):
            with open(paths['meta']) as f: meta = json.load(f)
            meta['modo'] = 'real'
        return rf_m, xgb_m, sc, feats, meta, 'real'
    except Exception as e:
        return None, None, None, None, {'modo':'demo','razon':str(e)}, 'demo'


def _vector(clasif, clima, elev_mean, df_ts, feat_list):
    """Construye el vector de features en el orden del entrenamiento."""
    def sg(col):
        s = df_ts[col].dropna() if col in df_ts.columns else pd.Series(dtype=float)
        return s

    ndvi = sg('NDVI_SG'); evi  = sg('EVI_SG');  gndvi = sg('GNDVI_SG')
    ndwi = sg('NDWI_SG'); savi = sg('SAVI_SG'); ndre  = sg('NDRE_SG')

    sm  = lambda s: float(s.mean()) if len(s) > 0 else np.nan
    sx  = lambda s: float(s.max())  if len(s) > 0 else np.nan
    sn  = lambda s: float(s.min())  if len(s) > 0 else np.nan
    std = lambda s: float(s.std())  if len(s) > 1 else np.nan
    amp = lambda s: float(s.max()-s.min()) if len(s) > 1 else np.nan

    pk_idx = df_ts['NDVI_SG'].idxmax() \
             if 'NDVI_SG' in df_ts.columns and len(ndvi) > 0 else None
    pk_doy = float(df_ts.loc[pk_idx,'fecha'].dayofyear) \
             if pk_idx is not None else np.nan

    nq = {}
    if 'NDVI_SG' in df_ts.columns and 'fecha' in df_ts.columns:
        for q in [1,2,3,4]:
            sub = df_ts[df_ts['fecha'].dt.quarter==q]['NDVI_SG']
            nq[q] = float(sub.mean()) if len(sub) > 0 else np.nan

    d = {
        'ndvi_max':sm(ndvi)+amp(ndvi)/2 if len(ndvi)>1 else sx(ndvi),
        'ndvi_min':sm(ndvi)-amp(ndvi)/2 if len(ndvi)>1 else sn(ndvi),
        'ndvi_mean':sm(ndvi), 'ndvi_std':std(ndvi),
        'ndvi_amplitude':amp(ndvi),
        'ndvi_auc':float(np.trapz(ndvi.values)) if len(ndvi)>1 else np.nan,
        'ndvi_peak_doy':pk_doy,
        'ndvi_q75_q25':float(ndvi.quantile(.75)-ndvi.quantile(.25))
                        if len(ndvi)>3 else np.nan,
        'evi_mean':sm(evi), 'evi_max':sx(evi),
        'evi_std':std(evi),  'evi_amplitude':amp(evi),
        'gndvi_mean':sm(gndvi), 'gndvi_max':sx(gndvi),
        'ndwi_mean':sm(ndwi),   'ndwi_min':sn(ndwi),
        'savi_mean':sm(savi),   'savi_max':sx(savi),
        'ndre_mean':sm(ndre),   'ndre_max':sx(ndre),
        'ndvi_q1':nq.get(1,np.nan), 'ndvi_q2':nq.get(2,np.nan),
        'ndvi_q3':nq.get(3,np.nan), 'ndvi_q4':nq.get(4,np.nan),
        'elev_mean':float(elev_mean) if elev_mean else np.nan,
        'n_obs':float(len(ndvi)),
        'tmax_mean':    float(clima.get('tmax_mean',26.5)),
        'tmin_mean':    float(clima.get('tmin_mean',16.0)),
        'precip_anual': float(clima.get('precip_anual',1300)),
        'precip_q1':    float(clima.get('precip_q1',80)),
        'precip_q2':    float(clima.get('precip_q2',320)),
        'precip_q3':    float(clima.get('precip_q3',480)),
        'precip_q4':    float(clima.get('precip_q4',200)),
        'tmax_floracion':float(clima.get('tmax_floracion',27.0)),
        'precip_fructic':float(clima.get('precip_fructic',750)),
    }
    vec = np.array([d.get(f, np.nan) for f in feat_list]).reshape(1,-1)
    return np.nan_to_num(vec, nan=0.0)


def _pred_real(rf_m, xgb_m, sc, feats, clasif, clima, elev_mean, df_ts, area_ha, dept):
    vec_sc   = sc.transform(_vector(clasif, clima, elev_mean, df_ts, feats))
    p_rf     = max(3.0, min(45.0, float(rf_m.predict(vec_sc)[0])))
    p_xgb    = max(3.0, min(45.0, float(xgb_m.predict(vec_sc)[0])))
    p_ens    = W_RF*p_rf + W_XGB*p_xgb
    arboles  = np.array([t.predict(vec_sc)[0] for t in rf_m.estimators_])
    ic_lo    = max(3.0,  float(np.percentile(arboles, 10)))
    ic_hi    = min(45.0, float(np.percentile(arboles, 90)))
    base     = IHCAFE_REF.get(dept, 20.0)
    return {
        'pred_rf':round(p_rf,2), 'pred_xgb':round(p_xgb,2),
        'pred_ens':round(p_ens,2), 'ic_lo':round(ic_lo,2), 'ic_hi':round(ic_hi,2),
        'prod_est':round(p_ens*area_ha,0), 'prod_lo':round(ic_lo*area_ha,0),
        'prod_hi':round(ic_hi*area_ha,0),
        'hist_dep':base, 'delta':round(p_ens-base,2), 'modo':'real',
    }


def _pred_demo(clasif, area_ha, dept, clima):
    base  = IHCAFE_REF.get(dept, 20.0)
    tmax  = clima.get('tmax_mean', 26.5) if clima else 26.5
    prec  = clima.get('precip_anual', 1300) if clima else 1300
    ajuste = ((clasif['ndvi_prom']-0.60)*18.0 + (clasif['ndvi_amp']-0.22)*9.0 +
              (clasif['evi_prom'] -0.40)*12.0 + (prec-1300)*0.003 + (tmax-26.0)*(-0.45))
    np.random.seed(int(abs(clasif['ndvi_prom']*10000)) % 2**31)
    p_rf  = round(max(5.0, min(40.0, base+ajuste+np.random.normal(0,.2))),2)
    p_xgb = round(max(5.0, min(40.0, base+ajuste+np.random.normal(0,.2))),2)
    p_ens = round(W_RF*p_rf + W_XGB*p_xgb, 2)
    ic_lo = round(p_ens*0.82, 2); ic_hi = round(p_ens*1.18, 2)
    return {
        'pred_rf':p_rf, 'pred_xgb':p_xgb, 'pred_ens':p_ens,
        'ic_lo':ic_lo, 'ic_hi':ic_hi,
        'prod_est':round(p_ens*area_ha,0), 'prod_lo':round(ic_lo*area_ha,0),
        'prod_hi':round(ic_hi*area_ha,0),
        'hist_dep':IHCAFE_REF.get(dept,20.0),
        'delta':round(p_ens-IHCAFE_REF.get(dept,20.0),2), 'modo':'demo',
    }


def predecir_rendimiento(clasif, area_ha, dept, clima, df_ts=None, elev_mean=None):
    """Punto de entrada principal. Usa modelos reales si existen, demo si no."""
    rf_m, xgb_m, sc, feats, meta, modo = cargar_modelos()
    if modo == 'real' and df_ts is not None:
        try:
            return _pred_real(rf_m, xgb_m, sc, feats,
                              clasif, clima, elev_mean, df_ts, area_ha, dept)
        except Exception as e:
            r = _pred_demo(clasif, area_ha, dept, clima)
            r['error_real'] = str(e)
            return r
    return _pred_demo(clasif, area_ha, dept, clima)


def estado_modelos():
    """Para mostrar el estado en el sidebar de Streamlit."""
    _, _, _, _, meta, modo = cargar_modelos()
    return modo, meta
