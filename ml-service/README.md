# 🎾 Servicio de Predicción de Partidos de Tenis

## Descripción

Este servicio implementa un sistema completo de predicción de partidos de tenis utilizando machine learning avanzado. El sistema incluye recopilación de datos, procesamiento, entrenamiento de modelos y una API REST para predicciones.

## Características Principales

- Sistema ELO personalizado para tenis
- Ingeniería de características avanzada
- Modelos ensemble y redes neuronales
- Recopilación de datos de múltiples fuentes
- API REST robusta
- Sistema de actualización automática
- Monitoreo y alertas

## Estructura del Proyecto

```
ml-service/
├── api.py                    # API REST con FastAPI
├── model_updater.py          # Sistema de actualización automática
├── advanced_data_collector.py # Recopilador de datos avanzado
├── advanced_scraper.py       # Web scraping avanzado
├── model_ensemble.py         # Modelos ensemble y redes neuronales
├── utils.py                  # Utilidades y procesamiento de datos
├── elo_system.py            # Sistema ELO personalizado
├── calculate_elo.py         # Cálculo de ratings ELO
├── train.py                 # Script de entrenamiento
├── config/                  # Configuraciones
├── data/                    # Datos y caché
├── logs/                    # Logs del sistema
├── models/                  # Modelos entrenados
└── tests/                   # Tests unitarios y de integración
```

## Requisitos

- Python 3.8+
- PostgreSQL 12+
- Dependencias listadas en `requirements.txt`

## Instalación

1. Clonar el repositorio:
```bash
git clone https://github.com/tu-usuario/tennis-predictor.git
cd tennis-predictor/ml-service
```

2. Crear entorno virtual:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Instalar dependencias:
```bash
pip install -r requirements.txt
```

4. Configurar variables de entorno:
```bash
cp .env.example .env
# Editar .env con tus configuraciones
```

5. Inicializar la base de datos:
```bash
python init_db.py
```

## Uso

### Iniciar el Servicio

1. Iniciar la API:
```bash
python api.py
```

2. Iniciar el actualizador de modelo:
```bash
python model_updater.py
```

### Endpoints de la API

- `POST /predict`: Predicción básica
- `POST /analyze`: Análisis detallado
- `GET /model-info`: Información del modelo
- `POST /train`: Iniciar entrenamiento

### Ejemplo de Uso

```python
import requests

# Configurar API key
headers = {'X-API-Key': 'tu-api-key'}

# Realizar predicción
data = {
    'tournament_id': '123',
    'player1_id': '456',
    'player2_id': '789',
    'match_date': '2024-03-21T15:00:00',
    'surface': 'Hard'
}

response = requests.post(
    'http://localhost:8000/predict',
    json=data,
    headers=headers
)
```

## Configuración

### Variables de Entorno Principales

- `API_KEY`: Clave para autenticación
- `DB_HOST`: Host de la base de datos
- `DB_NAME`: Nombre de la base de datos
- `MODEL_PATH`: Ruta al modelo entrenado

### Configuración del Modelo

Editar `config/model_updater_config.json` para ajustar:
- Frecuencia de actualización
- Umbrales de rendimiento
- Parámetros de entrenamiento

## Monitoreo

### Logs

Los logs se almacenan en:
- `logs/api.log`: Logs de la API
- `logs/model_updater.log`: Logs del actualizador
- `logs/data_collector.log`: Logs del recopilador

### Métricas

Acceder a métricas en tiempo real:
```bash
curl http://localhost:8000/metrics
```

## Tests

### Ejecutar Tests

```bash
# Tests unitarios
pytest tests/test_*.py

# Tests de integración
pytest tests/integration/test_*.py
```

### Cobertura de Tests

```bash
pytest --cov=. tests/
```

## Mantenimiento

### Actualización del Modelo

El modelo se actualiza automáticamente según la configuración en `model_updater_config.json`.

### Backup

Los backups del modelo se almacenan en `models/backups/`.

## Contribuir

1. Fork el repositorio
2. Crear rama feature (`git checkout -b feature/AmazingFeature`)
3. Commit cambios (`git commit -m 'Add AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abrir Pull Request

## Licencia

Este proyecto está licenciado bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles. 