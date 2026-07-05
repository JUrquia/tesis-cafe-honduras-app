# ☕ Sistema Predictivo de Rendimiento de Café — Honduras
## Tesis UNAH | Interfaz Streamlit

### Despliegue en streamlit.app

1. Sube esta carpeta a un repositorio público en GitHub
2. Ve a https://streamlit.app
3. Haz clic en "New app"
4. Conecta tu repositorio de GitHub
5. Selecciona `app.py` como archivo principal
6. Haz clic en "Deploy!"

### Estructura del proyecto
```
cafe_app/
├── app.py                  # Aplicación principal
├── requirements.txt        # Dependencias Python
├── .streamlit/
│   └── config.toml        # Tema y configuración
└── README.md
```

### Módulos disponibles
- 🏠 Dashboard General
- 📊 Análisis Histórico IHCAFE
- 🛰️ Explorador Satelital
- 🔮 Predicción de Cosecha
- 📋 Validación del Modelo
- 🗺️ Mapa de Departamentos
- ℹ️ Acerca del Sistema

### Integración con Google Colab
Los notebooks de Colab exportan CSV a Google Drive.
Sube esos archivos en los módulos correspondientes de la app.
