# Predictor de Partidos de Tenis

Un sistema completo para predecir ganadores de partidos de tenis utilizando Machine Learning, con una arquitectura de tres componentes:

- **Backend**: API REST en Java Spring Boot
- **Frontend**: Interfaz de usuario en React.js 
- **Servicio ML**: Modelo de predicción en Python con FastAPI

## Estructura del Proyecto

- `/backend`: API REST en Java Spring Boot
- `/frontend`: Interfaz de usuario en React.js
- `/ml-service`: Servicio de Machine Learning en Python

## Requisitos

- Java 11+
- Node.js 14+
- Python 3.8+
- PostgreSQL

## Instalación y Ejecución

### 1. Base de Datos
```bash
# Instalar PostgreSQL (macOS)
brew install postgresql
brew services start postgresql

# Crear base de datos
psql postgres -c "CREATE DATABASE tennis_predictor;"
```

### 2. Servicio ML
```bash
cd ml-service
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python train.py
python app.py
```

### 3. Backend
```bash
cd backend
mvn clean package
mvn spring-boot:run
```

### 4. Frontend
```bash
cd frontend
npm install
npm start
```

## Uso

1. Accede a la aplicación en [http://localhost:3000](http://localhost:3000)
2. Introduce los datos de los jugadores y la superficie
3. Haz clic en "Predecir" para obtener el resultado

## Licencia

MIT