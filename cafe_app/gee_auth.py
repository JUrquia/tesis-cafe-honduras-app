"""
gee_auth.py — Autenticacion de Google Earth Engine con Service Account
Soporta: Streamlit Cloud (secrets) y desarrollo local (archivo JSON)
"""

import ee
import json
import os
import streamlit as st


@st.cache_resource(show_spinner="Conectando con Google Earth Engine...")
def inicializar_gee():
    """
    Inicializa GEE con Service Account.
    Primero intenta Streamlit Secrets, luego archivo local.
    Retorna (True, mensaje) o (False, error).
    """
    # ── Opción 1: Streamlit Cloud Secrets ────────────────────────────────────
    try:
        if "gee_service_account" in st.secrets:
            creds_dict = dict(st.secrets["gee_service_account"])
            # Reconstruir el JSON que espera ee.ServiceAccountCredentials
            key_data = json.dumps({
                "type":                        creds_dict.get("type", "service_account"),
                "project_id":                  creds_dict.get("project_id"),
                "private_key_id":              creds_dict.get("private_key_id"),
                "private_key":                 creds_dict.get("private_key"),
                "client_email":                creds_dict.get("client_email"),
                "client_id":                   creds_dict.get("client_id"),
                "auth_uri":                    creds_dict.get("auth_uri"),
                "token_uri":                   creds_dict.get("token_uri"),
                "auth_provider_x509_cert_url": creds_dict.get("auth_provider_x509_cert_url"),
                "client_x509_cert_url":        creds_dict.get("client_x509_cert_url"),
            })

            credentials = ee.ServiceAccountCredentials(
                email=creds_dict["client_email"],
                key_data=key_data
            )
            ee.Initialize(credentials, project=creds_dict.get("project_id"))
            return True, f"GEE conectado via Service Account ({creds_dict['client_email']})"

    except Exception as e:
        # No hay secrets configurados o hubo error — intentar siguiente opción
        pass

    # ── Opción 2: Archivo local (desarrollo) ──────────────────────────────────
    local_paths = [
        "service_account.json",
        os.path.join(os.path.dirname(__file__), "service_account.json"),
        os.path.expanduser("~/.config/earthengine/service_account.json"),
    ]
    for path in local_paths:
        if os.path.exists(path):
            try:
                with open(path) as f:
                    creds_dict = json.load(f)
                credentials = ee.ServiceAccountCredentials(
                    email=creds_dict["client_email"],
                    key_data=json.dumps(creds_dict)
                )
                ee.Initialize(credentials,
                              project=creds_dict.get("project_id", "tesis-cafe-honduras"))
                return True, f"GEE conectado (archivo local: {path})"
            except Exception as e:
                continue

    # ── Opción 3: Credenciales de usuario (ultimo recurso) ────────────────────
    try:
        ee.Initialize(project="tesis-cafe-honduras")
        return True, "GEE conectado (credenciales de usuario)"
    except Exception as e:
        return False, (
            "No se pudo conectar con Earth Engine.\n"
            "Configura las credenciales en Streamlit Secrets o "
            "coloca service_account.json en la raiz del proyecto."
        )


def verificar_gee():
    """
    Verifica la conexion con GEE y muestra el estado en la UI.
    Retorna True si esta conectado, False si no.
    """
    ok, msg = inicializar_gee()
    if ok:
        st.sidebar.success(f"GEE: Conectado")
    else:
        st.sidebar.error(f"GEE: Sin conexion")
        st.sidebar.caption(msg)
    return ok
