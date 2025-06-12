# Plataforma de Aprendizaje Supervisado Distribuido

## Instrucciones de Ejecución Rápida

Este documento proporciona instrucciones rápidas para ejecutar el proyecto de Plataforma de Aprendizaje Supervisado Distribuido. Para instrucciones más detalladas, consulta el archivo `INSTALL.md`.

### Requisitos Previos

- Python 3.9+
- Ray 2.9.0+
- Scikit-learn 1.4.0+
- Pandas, NumPy, Matplotlib, etc.

### Instalación Rápida

1. Instala las dependencias:

```bash
pip install -r requirements.txt
```

2. Genera los datos de ejemplo:

```bash
python generate_data.py
```

### Ejecución del Sistema

#### Modo Demo Completo

Para ejecutar todos los componentes del sistema en modo demo:

```bash
python run_demo.py --demo-mode=all
```

#### Ejecutar Solo el Entrenamiento

```bash
python run_demo.py --demo-mode=train
```

#### Ejecutar Solo el Servicio de Modelos

```bash
python run_demo.py --demo-mode=serve
```

#### Ejecutar Solo el Monitoreo

```bash
python run_demo.py --demo-mode=monitor
```

### Ejecutar Manualmente los Componentes

Si prefieres ejecutar los componentes manualmente, usa:

#### Nodo Principal (Head)

```bash
python src/main.py --mode=train --head --config=src/config/config.yaml
```

#### Nodos Trabajadores

```bash
python src/main.py --mode=train --worker --head-address=localhost:6379 --config=src/config/config.yaml
```

#### Servicio de Modelos

```bash
python src/main.py --mode=serve --config=src/config/config.yaml
```

#### Monitoreo

```bash
python src/main.py --mode=monitor --config=src/config/config.yaml
```

### Pruebas Básicas

Para ejecutar pruebas básicas del sistema:

```bash
python tests/test_basic.py
```

### Análisis y Visualización

Para análisis y visualización de datos y resultados, abre el notebook Jupyter:

```bash
jupyter notebook notebooks/analysis_visualization.ipynb
```

### Usando Docker

También puedes ejecutar el sistema usando Docker:

```bash
cd docker
docker-compose up
```

Para ejecutar solo componentes específicos:

```bash
docker-compose up head worker
```

```bash
docker-compose up serving
```

```bash
docker-compose up monitor
```
