# Modelos entrenados — Sistema Predictivo de Café Honduras

Este directorio contiene los modelos ML entrenados con datos reales IHCAFE.

## Archivos requeridos (generados por entrenamiento_modelos_reales.ipynb)

- `modelo_rf.pkl`        — Random Forest (n_estimators=200, Tabla 10.2)
- `modelo_xgb.pkl`       — XGBoost (n_estimators=300, lr=0.05, Tabla 10.2)
- `scaler.pkl`           — StandardScaler ajustado con datos de entrenamiento
- `features_lista.pkl`   — Lista ordenada de los 26 features (Sección 10.2.1)
- `metadata_modelo.json` — Métricas de validación y fecha de entrenamiento

## Para generar los modelos

1. Abre `entrenamiento_modelos_reales.ipynb` en Google Colab
2. Sube `ihcafe_rendimiento_2021_2025.csv`
3. Ejecuta todas las celdas en orden
4. Los .pkl se exportan a Google Drive/tesis_cafe_honduras/modelos/
5. Descárgalos y súbelos a esta carpeta en GitHub

## Estado actual

Sin archivos .pkl → Streamlit usa modelo de demostración (fórmula calibrada)
Con archivos .pkl → Streamlit usa modelos RF+XGB reales entrenados con IHCAFE
