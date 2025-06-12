###### Gestor de configuración para la Plataforma de Aprendizaje Supervisado Distribuido

import yaml
from pathlib import Path


def load_config(config_path):
    ###### Cargar configuración desde un archivo YAML
    # 
    # Args:
    #    config_path (str): Ruta al archivo de configuración YAML
    #
    # Returns:
    #    dict: Diccionario de configuración
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file {config_path} not found.")
        return {}
    except yaml.YAMLError as e:
        print(f"Error parsing configuration file: {e}")
        return {}


def get_project_root():
    ###### Obtener la ruta absoluta del directorio raíz del proyecto
    #
    # Returns:
    #    Path: Ruta al directorio raíz del proyecto
    return Path(__file__).parent.parent.parent
