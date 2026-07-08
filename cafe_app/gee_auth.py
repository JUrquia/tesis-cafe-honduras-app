"""
gee_auth.py — Autenticacion GEE con Service Account
Soporta: base64 (recomendado), TOML directo, y archivo local
"""

import ee
import json
import base64
import os
import streamlit as st


@st.cache_resource(show_spinner="Conectando con Google Earth Engine...")
def inicializar_gee():
    """
    Inicializa GEE probando 3 métodos en orden.
    Retorna (True, mensaje) o (False, error).
    """

    # ── Método 1: Base64 (recomendado — evita problemas de formato) ───────────
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
                          project=sa_dict.get('project_id', 'tesis-cafe-honduras'))
            return True, f"GEE via base64 ({sa_dict['client_email']})"
    except Exception as e:
        pass  # intentar siguiente método

    # ── Método 2: TOML directo ─────────────────────────────────────────────────
    try:
        if "gee_service_account" in st.secrets:
            creds_dict = dict(st.secrets["gee_service_account"])

            # Reconstruir private_key con saltos reales si vienen como \n literal
            pk = creds_dict.get('private_key', '')
            if '\\n' in pk and '\n' not in pk:
                pk = pk.replace('\\n', '\n')
            creds_dict['private_key'] = pk

            key_data = json.dumps({
                "type":                        creds_dict.get("type", "service_account"),
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
                          project=creds_dict.get('project_id', 'tesis-cafe-honduras'))
            return True, f"GEE via TOML ({creds_dict['client_email']})"
    except Exception as e:
        pass

    # ── Método 3: Archivo local ────────────────────────────────────────────────
    local_paths = [
        "service_account.json",
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "service_account.json"),
    ]
    for path in local_paths:
        if os.path.exists(path):
            try:
                with open(path) as f:
                    raw = f.read()
                sa  = json.loads(raw)
                credentials = ee.ServiceAccountCredentials(
                    email=sa['client_email'],
                    key_data=raw
                )
                ee.Initialize(credentials,
                              project=sa.get('project_id', 'tesis-cafe-honduras'))
                return True, f"GEE via archivo local ({path})"
            except Exception:
                continue

    return False, (
        "No se pudo conectar con Earth Engine.\n"
        "Configura [gee_b64] en Streamlit Secrets."
    )


def verificar_gee():
    ok, msg = inicializar_gee()
    if ok:
        st.sidebar.success("GEE: Conectado")
    else:
        st.sidebar.error("GEE: Sin conexion")
        st.sidebar.caption(msg)
    return ok
