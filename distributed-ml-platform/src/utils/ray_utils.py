"""
Ray utility functions for distributed computing
"""
import ray
import logging
import time
import os
import socket
import random

logger = logging.getLogger(__name__)

def initialize_ray(address=None, local=True, log_to_driver=True, num_retries=5, retry_interval=5):
    """
    Initialize Ray with the given configuration and retry logic
    
    Args:
        address: Optional address to connect to an existing Ray cluster
        local: Whether to initialize Ray in local mode if no address is provided
        log_to_driver: Whether to log to the driver
        num_retries: Number of retries for connection failures
        retry_interval: Seconds to wait between retries
    
    Returns:
        bool: True if Ray was initialized successfully
    """
    for attempt in range(num_retries):
        try:
            # Check if Ray is already initialized
            if ray.is_initialized():
                logger.info("Ray is already initialized")
                return True
                
            # Set runtime environment for fault tolerance
            runtime_env = {
                "env_vars": {
                    "RAY_ENABLE_AUTO_RECONNECT": "1",  # Auto reconnect workers
                }
            }
                
            if address:
                logger.info(f"Connecting to Ray cluster at {address}")
                ray.init(address=address, log_to_driver=log_to_driver, runtime_env=runtime_env)
            else:
                if local:
                    logger.info("Initializing Ray locally")
                    ray.init(log_to_driver=log_to_driver, runtime_env=runtime_env)
                else:
                    logger.info("Auto-discovering Ray cluster")
                    ray.init(address="auto", log_to_driver=log_to_driver, runtime_env=runtime_env)
                    
            logger.info(f"Ray initialized with resources: {ray.cluster_resources()}")
            return True
        except Exception as e:
            if attempt < num_retries - 1:
                wait_time = retry_interval * (1 + random.random())  # Add jitter
                logger.warning(f"Failed to initialize Ray (attempt {attempt+1}/{num_retries}): {e}. Retrying in {wait_time:.1f}s...")
                time.sleep(wait_time)
            else:
                logger.error(f"Error initializing Ray after {num_retries} attempts: {e}")
                return False

def shutdown_ray():
    """
    Shutdown Ray
    
    Returns:
        bool: True if Ray was shutdown successfully
    """
    try:
        if ray.is_initialized():
            ray.shutdown()
            logger.info("Ray has been shutdown")
            return True
        else:
            logger.info("Ray is not initialized, no need to shutdown")
            return True
    except Exception as e:
        logger.error(f"Error shutting down Ray: {e}")
        return False

def get_cluster_status():
    """
    Get status information about the Ray cluster
    
    Returns:
        dict: Information about the Ray cluster or None if Ray is not initialized
    """
    try:
        if not ray.is_initialized():
            logger.error("Ray is not initialized")
            return None
            
        nodes = ray.nodes()
        
        total_cpus = 0
        total_gpus = 0
        total_memory = 0
        alive_nodes = 0
        
        for node in nodes:
            if node.get('alive', False):
                alive_nodes += 1
                resources = node.get('Resources', {})
                total_cpus += resources.get('CPU', 0)
                total_gpus += resources.get('GPU', 0)
                total_memory += resources.get('memory', 0) / (1024 * 1024 * 1024)  # Convert to GB
                
        return {
            'total_nodes': len(nodes),
            'alive_nodes': alive_nodes,
            'dead_nodes': len(nodes) - alive_nodes,
            'total_cpus': total_cpus,
            'total_gpus': total_gpus,
            'total_memory_gb': round(total_memory, 2),
            'node_ips': [node.get('NodeManagerAddress') for node in nodes if node.get('alive', False)]
        }
    except Exception as e:
        logger.error(f"Error getting cluster status: {e}")
        return None

def check_cluster_health(min_nodes=1, min_cpus=1):
    """
    Check if the Ray cluster is healthy
    
    Args:
        min_nodes: Minimum number of nodes required
        min_cpus: Minimum number of CPUs required
        
    Returns:
        bool: True if the cluster is healthy
    """
    try:
        status = get_cluster_status()
        if not status:
            return False
            
        return status['alive_nodes'] >= min_nodes and status['total_cpus'] >= min_cpus
    except Exception as e:
        logger.error(f"Error checking cluster health: {e}")
        return False

class RayFailoverManager:
    """
    Manager for handling Ray cluster failover scenarios
    """
    def __init__(self, primary_address=None, secondary_addresses=None):
        self.primary_address = primary_address
        self.secondary_addresses = secondary_addresses or []
        self.current_address = None
        self.connected = False
        
    def connect(self, max_retries=3):
        """
        Connect to Ray cluster with failover support
        
        Returns:
            bool: True if connected successfully
        """
        # Try primary address first
        if self.primary_address:
            if self._try_connect(self.primary_address, max_retries=max_retries):
                return True
                
        # Try secondary addresses if primary fails
        for address in self.secondary_addresses:
            if self._try_connect(address, max_retries=1):  # Less retries for secondary
                return True
                
        # Last resort: try local
        logger.warning("All Ray cluster connections failed, falling back to local mode")
        if initialize_ray(local=True):
            self.current_address = "local"
            self.connected = True
            return True
            
        return False
        
    def _try_connect(self, address, max_retries=3):
        """
        Try to connect to a specific Ray address
        
        Args:
            address: Ray cluster address
            max_retries: Number of connection attempts
            
        Returns:
            bool: True if connected successfully
        """
        for i in range(max_retries):
            try:
                if initialize_ray(address=address):
                    self.current_address = address
                    self.connected = True
                    logger.info(f"Successfully connected to Ray at {address}")
                    return True
            except Exception as e:
                wait_time = 2 ** i  # Exponential backoff
                logger.warning(f"Failed to connect to Ray at {address} (attempt {i+1}/{max_retries}): {e}")
                if i < max_retries - 1:
                    logger.info(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    
        return False
        
    def reconnect_if_needed(self):
        """
        Check connection and reconnect if needed
        
        Returns:
            bool: True if connected (either already or after reconnection)
        """
        if not self.connected or not ray.is_initialized():
            logger.warning("Ray connection lost, attempting to reconnect")
            return self.connect()
        return True

def create_fault_tolerant_actor(actor_cls, *args, **kwargs):
    """
    Create a fault-tolerant Ray actor with auto-restart
    
    Args:
        actor_cls: Actor class to create
        *args: Positional arguments for the actor
        **kwargs: Keyword arguments for the actor
        
    Returns:
        ray.actor.ActorHandle: Handle to the created actor
    """
    return ray.remote(
        max_restarts=-1,  # Infinite restarts
        max_task_retries=3,
        num_cpus=kwargs.pop("num_cpus", 0.1),  # Default to minimal CPU
    )(actor_cls).remote(*args, **kwargs)
