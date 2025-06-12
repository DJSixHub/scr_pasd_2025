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

### Verificación del Sistema

Antes de ejecutar el sistema, puedes verificar que todo esté correctamente configurado:

```powershell
# Para entorno local
.\verificar_sistema.ps1 -entorno local

# Para entorno Docker
.\verificar_sistema.ps1 -entorno docker
```

### Ejecución del Sistema

El proyecto incluye scripts de PowerShell para facilitar la ejecución:

#### Ejecución Local

```powershell
# Modo Completo (todos los componentes)
.\ejecutar_local.ps1 -modo todo

# Solo entrenamiento (nodo principal o trabajador)
.\ejecutar_local.ps1 -modo entrenamiento

# Solo servicio de modelos
.\ejecutar_local.ps1 -modo servicio

# Solo monitoreo
.\ejecutar_local.ps1 -modo monitoreo

# Usando una configuración personalizada
.\ejecutar_local.ps1 -modo todo -configPath "ruta/a/mi_config.yaml"
```

#### Ejecución con Docker

```powershell
# Modo Completo (todos los servicios)
.\ejecutar_docker.ps1 -componente todo

# Solo nodo principal
.\ejecutar_docker.ps1 -componente head

# Solo nodos trabajadores
.\ejecutar_docker.ps1 -componente worker

# Solo servicio de modelos
.\ejecutar_docker.ps1 -componente serving

# Solo monitoreo
.\ejecutar_docker.ps1 -componente monitor

# Usando una configuración personalizada
.\ejecutar_docker.ps1 -componente todo -configPath "ruta/a/mi_config.yaml"
```

#### Ejecución Manual (Avanzada)

```bash
python run.py --modo=monitoreo
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
