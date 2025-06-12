# Sistema de Entrenamiento y Despliegue Distribuido

Este archivo contiene instrucciones detalladas para instalar y ejecutar la plataforma de aprendizaje supervisado distribuido.

## Requisitos previos

- Python 3.9+
- Docker
- Ray 2.9.0+
- Scikit-learn 1.4.0+

## Instalación

1. Clonar el repositorio:

```bash
git clone https://github.com/DJSixHub/scr_pasd_2025.git
cd scr_pasd_2025
```

2. Crear un entorno virtual e instalar dependencias:

```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Configurar el entorno:

Copiar el archivo de configuración de ejemplo:

```bash
cp src/config/config.example.yaml src/config/config.yaml
```

Ajustar los parámetros según sea necesario.

## Ejecución

### Entrenamiento distribuido

Para iniciar el nodo principal:

```bash
python src/main.py --mode=train --head
```

Para añadir nodos trabajadores:

```bash
python src/main.py --mode=train --worker --head-address=<IP_NODO_PRINCIPAL>:6379
```

### Despliegue de modelos (Serving)

```bash
python src/main.py --mode=serve
```

### Monitoreo y visualización

```bash
python src/main.py --mode=monitor
```

### Modo simplificado de ejecución

También puedes usar el script unificado para ejecutar todos los componentes:

```bash
python run.py --modo=todo
```

## Estructura del proyecto

- `src/`: Código fuente del proyecto
  - `training/`: Módulos para entrenamiento distribuido
  - `serving/`: Módulos para servir modelos
  - `monitoring/`: Módulos para monitoreo y visualización
  - `utils/`: Utilidades compartidas
  - `models/`: Definiciones de modelos
  - `config/`: Archivos de configuración
- `data/`: Datos para entrenamiento y evaluación
  - `raw/`: Datos sin procesar
  - `processed/`: Datos procesados
- `notebooks/`: Jupyter notebooks para exploración y análisis
- `tests/`: Pruebas unitarias y de integración
- `docker/`: Archivos Docker para despliegue

## Características

- Entrenamiento distribuido con Ray
- Autodescubrimiento de nodos
- Tolerancia a fallos
- API para interacción con modelos
- Visualización de métricas
