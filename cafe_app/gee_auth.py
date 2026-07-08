"""
gee_auth.py — Autenticacion GEE
Soporta: OAuth personal (base64), Service Account (base64), y archivo local
"""

import ee
import json
import base64
import os
import streamlit as st


@st.cache_resource(show_spinner="Conectando con Google Earth Engine...")
def inicializar_gee():
    """
    Inicializa GEE probando metodos en orden de prioridad.
    Retorna (True, mensaje) o (False, error).
    """

    # ── Método 1: OAuth personal en base64 (recomendado) ─────────────────────
    try:
        if "gee_oauth" in st.secrets:
            b64_str    = st.secrets["gee_oauth"]["credentials"]
            creds_str  = base64.b64decode(b64_str.encode()).decode()
            creds_dict = json.loads(creds_str)

            # Escribir credenciales al path que espera ee
            cred_path = os.path.expanduser('~/.config/earthengine/credentials')
            os.makedirs(os.path.dirname(cred_path), exist_ok=True)
            with open(cred_path, 'w') as f:
                json.dump(creds_dict, f)

            ee.Initialize(project='tesis-cafe-honduras')
            # Verificar conexion
            _ = ee.Number(1).getInfo()
            return True, "GEE conectado (cuenta personal OAuth)"
    except Exception as e:
        pass

    # ── Método 2: Service Account en base64 ───────────────────────────────────
    try:
        if "gee_b64" in st.secrets:
            b64_str     = st.secrets["gee_b64"]["credentials"]
            sa_json_str = base64.b64decode(b64_str.encode()).decode()
            sa_dict     = json.loads(sa_json_str)

            credentials = ee.ServiceAccountCredentials(
                email=sa_dict['client_email'],
                key_data=sa_json_str
            )
            ee.Initialize(credentials,
                          project=sa_dict.get('project_id','tesis-cafe-honduras'))
            return True, f"GEE conectado (Service Account)"
    except Exception as e:
        pass

    # ── Método 3: Service Account TOML directo ────────────────────────────────
    try:
        if "gee_service_account" in st.secrets:
            creds_dict = dict(st.secrets["gee_service_account"])
            pk = creds_dict.get('private_key', '')
            if '\\n' in pk and '\n' not in pk:
                pk = pk.replace('\\n', '\n')
            creds_dict['private_key'] = pk

            key_data = json.dumps({
                "type":                        creds_dict.get("type","service_account"),
                "project_id":                  creds_dict.get("project_id"),
                "private_key_id":              creds_dict.get("private_key_id"),
                "private_key":                 pk,
                "client_email":                creds_dict.get("client_email"),
                "client_id":                   creds_dict.get("client_id"),
                "auth_uri":                    creds_dict.get("auth_uri"),
                "token_uri":                   creds_dict.get("token_uri"),
                "auth_provider_x509_cert_url": creds_dict.get("auth_provider_x509_cert_url"),
                "client_x509_cert_url":        creds_dict.get("client_x509_cert_url"),
            })
            credentials = ee.ServiceAccountCredentials(
                email=creds_dict['client_email'],
                key_data=key_data
            )
            ee.Initialize(credentials,
                          project=creds_dict.get('project_id','tesis-cafe-honduras'))
            return True, "GEE conectado (Service Account TOML)"
    except Exception as e:
        pass

    # ── Método 4: Archivo local ───────────────────────────────────────────────
    for path in ["service_account.json",
                 os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "service_account.json")]:
        if os.path.exists(path):
            try:
                with open(path) as f:
                    raw = f.read()
                sa = json.loads(raw)
                credentials = ee.ServiceAccountCredentials(
                    email=sa['client_email'], key_data=raw
                )
                ee.Initialize(credentials,
                              project=sa.get('project_id','tesis-cafe-honduras'))
                return True, f"GEE conectado (archivo local)"
            except Exception:
                continue

    return False, (
        "No se pudo conectar con Earth Engine.\n"
        "Configura [gee_oauth] en Streamlit Secrets."
    )


def verificar_gee():
    ok, msg = inicializar_gee()
    if ok:
        st.sidebar.success("GEE: Conectado")
    else:
        st.sidebar.error("GEE: Sin conexion")
        st.sidebar.caption(msg)
    return ok
