#!/usr/bin/env python3
###### Plataforma de Aprendizaje Supervisado Distribuido - Script de Ejecución
###### 
###### Este script permite ejecutar la plataforma para entrenar modelos en múltiples datasets,
###### desplegarlos para servicio y monitorear el sistema.

import os
import sys
import time
import argparse
import subprocess
from pathlib import Path

# Agregar el directorio raíz del proyecto al path de Python
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    parser = argparse.ArgumentParser(description="Plataforma de Aprendizaje Supervisado Distribuido")
    parser.add_argument("--modo", type=str, choices=['todo', 'entrenamiento', 'servicio', 'monitoreo'], 
                        default='todo', help="Modo de ejecución")
    parser.add_argument("--config", type=str, default="src/config/config.yaml",
                        help="Ruta al archivo de configuración")
    return parser.parse_args()


def ejecutar_generacion_datos():
    """Generar datasets de ejemplo."""
    print("\n=== Generando Datasets de Ejemplo ===")
    try:
        from src.utils.generate_data import save_iris, save_diabetes, save_breast_cancer, save_wine
        save_iris()
        save_diabetes()
        save_breast_cancer()
        save_wine()
        print("Datasets de ejemplo generados exitosamente!")
        return True
    except Exception as e:
        print(f"Error generando datasets: {e}")
        return False


def ejecutar_nodo_principal(config_path):
    """Ejecutar el nodo principal para entrenamiento."""
    print("\n=== Iniciando Nodo Principal ===")
    try:
        cmd = ["python", "src/main.py", "--mode=train", "--head", f"--config={config_path}"]
        process = subprocess.Popen(cmd)
        return process
    except Exception as e:
        print(f"Error iniciando nodo principal: {e}")
        return None


def ejecutar_nodo_trabajador(config_path, dir_nodo_principal="localhost:6379"):
    """Ejecutar un nodo trabajador para entrenamiento."""
    print(f"\n=== Iniciando Nodo Trabajador (conectando a {dir_nodo_principal}) ===")
    try:
        cmd = ["python", "src/main.py", "--mode=train", "--worker", 
               f"--head-address={dir_nodo_principal}", f"--config={config_path}"]
        process = subprocess.Popen(cmd)
        return process
    except Exception as e:
        print(f"Error iniciando nodo trabajador: {e}")
        return None


def ejecutar_servicio(config_path):
    """Ejecutar el servicio de API de modelos."""
    print("\n=== Iniciando Servicio de Modelos ===")
    try:
        cmd = ["python", "src/main.py", "--mode=serve", f"--config={config_path}"]
        process = subprocess.Popen(cmd)
        return process
    except Exception as e:
        print(f"Error iniciando servicio de modelos: {e}")
        return None


def ejecutar_monitoreo(config_path):
    """Ejecutar el monitoreo del sistema."""
    print("\n=== Iniciando Monitoreo del Sistema ===")
    try:
        cmd = ["python", "src/main.py", "--mode=monitor", f"--config={config_path}"]
        process = subprocess.Popen(cmd)
        return process
    except Exception as e:
        print(f"Error iniciando monitoreo: {e}")
        return None


def main():
    args = parse_args()
    
    # Verificar si existe el archivo de configuración, si no crear uno predeterminado
    config_path = args.config
    if not os.path.exists(config_path):
        print(f"Archivo de configuración {config_path} no encontrado. Creando configuración predeterminada...")
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        # Crear un archivo de configuración básico
        with open(config_path, 'w') as f:
            f.write("""###### Archivo de configuración para la Plataforma de Aprendizaje Supervisado Distribuido

# Configuración de Entrenamiento
datasets:
  - name: iris
    target_column: species
    test_size: 0.2
    random_state: 42
  
  - name: diabetes
    target_column: target
    test_size: 0.25
    random_state: 42

# Configuración de Modelos
models:
  - type: random_forest
    params:
      n_estimators: 100
      max_depth: 10
      random_state: 42
      task: classification
  
  - type: gradient_boosting
    params:
      n_estimators: 100
      learning_rate: 0.1
      random_state: 42
      task: classification
  
  - type: logistic_regression
    params:
      max_iter: 1000
      random_state: 42
  
  - type: svm
    params:
      kernel: rbf
      C: 1.0
      random_state: 42
      task: classification

# Configuración de Ray
ray:
  head_address: localhost:6379
  redis_password: null
  num_cpus: null  # Usar todas las CPUs disponibles
  num_gpus: 0

# Configuración de Servicio
serving:
  port: 8000
  host: 0.0.0.0

# Configuración de Monitoreo
monitoring:
  interval: 5  # en segundos
  save_plots: true
""")
        print(f"Creada configuración predeterminada en {config_path}")
    
    # Generar datasets de ejemplo si no existen
    data_dir = Path("data/raw")
    if not (data_dir / "iris.csv").exists():
        if not ejecutar_generacion_datos():
            print("Error al generar datasets de ejemplo. Saliendo.")
            return
    
    procesos = []
    
    if args.modo in ['todo', 'entrenamiento']:
        # Iniciar el nodo principal
        proceso_principal = ejecutar_nodo_principal(config_path)
        if proceso_principal:
            procesos.append(proceso_principal)
            
            # Dar tiempo para inicializar el nodo principal
            print("Esperando a que se inicialice el nodo principal...")
            time.sleep(10)
            
            # Iniciar nodos trabajadores
            for i in range(2):  # Iniciar 2 nodos trabajadores
                proceso_trabajador = ejecutar_nodo_trabajador(config_path)
                if proceso_trabajador:
                    procesos.append(proceso_trabajador)
                time.sleep(2)  # Dar tiempo para conectarse
    
    if args.modo in ['todo', 'servicio']:
        # Iniciar servicio de modelos
        proceso_servicio = ejecutar_servicio(config_path)
        if proceso_servicio:
            procesos.append(proceso_servicio)
    
    if args.modo in ['todo', 'monitoreo']:
        # Iniciar monitoreo
        proceso_monitoreo = ejecutar_monitoreo(config_path)
        if proceso_monitoreo:
            procesos.append(proceso_monitoreo)
    
    print("\n=== Sistema en Ejecución ===")
    print("Presiona Ctrl+C para detener...")
    
    try:
        for proceso in procesos:
            proceso.wait()
    except KeyboardInterrupt:
        print("\nDeteniendo todos los procesos...")
        for proceso in procesos:
            proceso.terminate()
        
        # Esperar a que los procesos terminen
        for proceso in procesos:
            try:
                proceso.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proceso.kill()
        
        print("Todos los procesos detenidos.")


if __name__ == "__main__":
    main()
