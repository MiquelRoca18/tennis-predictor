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

## Requisitos del Sistema

### Software
- Python 3.8+ (recomendado 3.10)
- PostgreSQL 12+
- Git

### Hardware Recomendado
- CPU: 4+ cores
- RAM: 8GB+ (16GB recomendado para entrenamiento)
- Almacenamiento: 10GB+ libre

### Dependencias Principales
- TensorFlow 2.12+
- FastAPI
- SQLAlchemy
- Pandas
- NumPy
- XGBoost
- Scikit-learn

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
│   ├── model_config.json    # Configuración del modelo
│   └── api_config.json      # Configuración de la API
├── data/                    # Datos y caché
├── logs/                    # Logs del sistema
├── model/                   # Modelos entrenados
│   └── backups/            # Backups de modelos
├── sql/                     # Scripts SQL
│   ├── db_schema.sql       # Esquema de la base de datos
│   └── initial_data.sql    # Datos iniciales
└── tests/                   # Tests unitarios y de integración
    ├── unit/               # Tests unitarios
    └── integration/        # Tests de integración
```

## Instalación

### 1. Clonar el Repositorio
```bash
git clone https://github.com/tu-usuario/tennis-predictor.git
cd tennis-predictor/ml-service
```

### 2. Inicializar el Proyecto
```bash
python init_project.py
```
Este script realizará automáticamente:
- Creación de la estructura de directorios
- Configuración del entorno virtual
- Instalación de dependencias
- Creación de archivos de configuración iniciales
- Preparación de la base de datos

### 3. Configurar el Entorno

1. Activar el entorno virtual:
```bash
# Linux/Mac
source venv/bin/activate

# Windows
.\\venv\\Scripts\\activate
```

2. Configurar variables de entorno:
```bash
# Copiar el archivo de ejemplo (si no lo hizo init_project.py)
cp .env.example .env

# Editar .env con tu configuración
nano .env
```

Variables de entorno requeridas:
- `DB_HOST`: Host de la base de datos
- `DB_PORT`: Puerto de la base de datos
- `DB_NAME`: Nombre de la base de datos
- `DB_USER`: Usuario de la base de datos
- `DB_PASSWORD`: Contraseña de la base de datos
- `API_KEY`: Clave para autenticación de la API
- `MODEL_PATH`: Ruta al modelo entrenado
- `LOG_LEVEL`: Nivel de logging (DEBUG, INFO, WARNING, ERROR)

### 4. Configurar la Base de Datos

1. Crear la base de datos PostgreSQL:
```bash
createdb tennis_predictor
```

2. Inicializar el esquema:
```bash
psql -U tu_usuario -d tennis_predictor -f sql/db_schema.sql
psql -U tu_usuario -d tennis_predictor -f sql/initial_data.sql
```

## Uso

### Iniciar el Servicio

1. Activar el entorno virtual (si no está activo):
```bash
source venv/bin/activate  # Linux/Mac
.\\venv\\Scripts\\activate  # Windows
```

2. Iniciar la API:
```bash
python api.py
```

3. Iniciar el actualizador de modelo (en otra terminal):
```bash
python model_updater.py
```

### Endpoints de la API

#### Predicción Básica
```http
POST /predict
Content-Type: application/json
X-API-Key: tu-api-key

{
    "tournament_id": "123",
    "player1_id": "456",
    "player2_id": "789",
    "match_date": "2024-03-21T15:00:00",
    "surface": "Hard"
}
```

#### Análisis Detallado
```http
POST /analyze
Content-Type: application/json
X-API-Key: tu-api-key

{
    "match_id": "123",
    "include_stats": true,
    "include_history": true
}
```

## Mantenimiento

### Logs
Los logs se encuentran en el directorio `logs/`:
- `api.log`: Logs de la API
- `model_updater.log`: Logs del actualizador
- `data_collector.log`: Logs del recopilador

### Backups
Los modelos se respaldan automáticamente en `model/backups/` antes de cada actualización.

### Monitoreo
Métricas disponibles en:
```bash
curl http://localhost:8000/metrics
```

## Tests

### Tests Unitarios
```bash
pytest tests/unit/
```

### Tests de Integración
```bash
pytest tests/integration/
```

### Cobertura
```bash
pytest --cov=. tests/
```

## Solución de Problemas

### Problemas Comunes

1. Error de conexión a la base de datos:
   - Verificar que PostgreSQL esté corriendo
   - Comprobar credenciales en `.env`
   - Verificar que la base de datos existe

2. Error al cargar el modelo:
   - Verificar que `MODEL_PATH` en `.env` es correcto
   - Comprobar permisos de lectura

3. Errores de memoria durante el entrenamiento:
   - Reducir `batch_size` en `config/model_config.json`
   - Liberar memoria RAM
   - Considerar usar un equipo con más recursos

## Contribuir

1. Fork el repositorio
2. Crear rama feature (`git checkout -b feature/NuevaCaracteristica`)
3. Commit cambios (`git commit -m 'Añadir nueva característica'`)
4. Push a la rama (`git push origin feature/NuevaCaracteristica`)
5. Abrir Pull Request

### Guía de Estilo
- Seguir PEP 8
- Documentar funciones y clases
- Añadir tests para nuevas características
- Mantener la cobertura de tests > 80%

## Licencia

Este proyecto está licenciado bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles. 