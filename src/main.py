#!/usr/bin/env python3
###### Punto de entrada principal para la Plataforma de Aprendizaje Supervisado Distribuido

import argparse
import os
import sys
import ray

# Agregar el directorio raíz del proyecto al path de Python
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_config
from src.training.trainer import DistributedTrainer
from src.serving.api import ModelServing
from src.monitoring.monitor import SystemMonitor


def parse_args():
    parser = argparse.ArgumentParser(description="Plataforma de Aprendizaje Supervisado Distribuido")
    parser.add_argument("--mode", type=str, choices=['train', 'serve', 'monitor'], 
                        required=True, help="Modo de operación")
    parser.add_argument("--config", type=str, default="src/config/config.yaml",
                        help="Ruta al archivo de configuración")
    parser.add_argument("--head", action="store_true", 
                        help="Iniciar como nodo principal (para modo entrenamiento)")
    parser.add_argument("--worker", action="store_true", 
                        help="Iniciar como nodo trabajador (para modo entrenamiento)")
    parser.add_argument("--head-address", type=str, default=None,
                        help="Dirección del nodo principal (para nodos trabajadores)")
    
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    
    # Inicializar Ray según el tipo de nodo
    if args.mode == 'train':
        if args.head:
            ray.init(include_dashboard=True)
            print("Ray inicializado como nodo principal")
        elif args.worker and args.head_address:
            ray.init(address=args.head_address)
            print(f"Ray inicializado como nodo trabajador, conectado a {args.head_address}")
        else:
            print("Error: Para el modo de entrenamiento, especifica --head o --worker con --head-address")
            return
        
        # Iniciar entrenamiento
        trainer = DistributedTrainer(config)
        trainer.train()
        
    elif args.mode == 'serve':
        # Inicializar Ray para servicio
        ray.init()
        serving = ModelServing(config)
        serving.start()
        
    elif args.mode == 'monitor':
        # Inicializar Ray para monitoreo
        ray.init()
        monitor = SystemMonitor(config)
        monitor.start()
        
    else:
        print(f"Modo desconocido: {args.mode}")


if __name__ == "__main__":
    main()
