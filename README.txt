CENTINELA — Detección de Fraude en Apertura de Cuentas Bancarias
TFM — Emilio Andrés Navarro Vergara — Máster Data Science, Big Data and Bussines Analytics UCM, 7.ª ed.

CONTENIDO
- Memoria_TFM_Emilio_Navarro.pdf ...... memoria (20 caras)
- Anexo_A_Notebook_TFM.html ........... código completo, EDA y modelado con salidas
- Anexo_B_Dashboard_y_Monitor.html .... panel de operaciones y monitor de deriva (código)
- dashboard/centinela_ops.html ........ panel interactivo (abrir en cualquier navegador)
- powerbi/CENTINELA_Ejecutivo.pbix .... cuadro de mando ejecutivo
- api/ ................................ servicio de predicción (FastAPI) + solicitud de ejemplo
- modelos/centinela_v1.joblib ......... sistema empaquetado (preprocesador+modelo+calibrador+umbral)
- requirements.txt .................... dependencias de Python del proyecto
- video_link.txt ...................... enlace al vídeo

REQUISITOS E INSTALACIÓN
Entorno: Python 3.12; semilla global 42.

1. Crear un entorno virtual (recomendado):
   python -m venv venv
   venv\Scripts\activate          (Windows)
   source venv/bin/activate       (Linux/Mac)

2. Instalar las dependencias:
   pip install -r requirements.txt

REPRODUCCIÓN
Datos: Bank Account Fraud (BAF), Feedzai/NeurIPS 2022 — descarga desde Kaggle
(licencia CC BY-NC-ND: los datos no se redistribuyen en este paquete).

API local: una vez instaladas las dependencias, desde api/:
   uvicorn app:app