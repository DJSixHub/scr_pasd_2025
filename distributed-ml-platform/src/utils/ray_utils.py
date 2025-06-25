"""
Ray utility functions for distributed computing
"""
import ray
import logging

logger = logging.getLogger(__name__)

def initialize_ray(address=None, local=True, log_to_driver=True):
    """
    Initialize Ray with the given configuration
    
    Args:
        address: Optional address to connect to an existing Ray cluster
        local: Whether to initialize Ray in local mode if no address is provided
        log_to_driver: Whether to log to the driver
    
    Returns:
        bool: True if Ray was initialized successfully
    """
    try:
        # Check if Ray is already initialized
        if ray.is_initialized():
            logger.info("Ray is already initialized")
            return True
            
        if address:
            logger.info(f"Connecting to Ray cluster at {address}")
            ray.init(address=address, log_to_driver=log_to_driver)
        else:
            if local:
                logger.info("Initializing Ray locally")
                ray.init(log_to_driver=log_to_driver)
            else:
                logger.info("Auto-discovering Ray cluster")
                ray.init(address="auto", log_to_driver=log_to_driver)
                
        logger.info(f"Ray initialized with resources: {ray.cluster_resources()}")
        return True
    except Exception as e:
        logger.error(f"Error initializing Ray: {e}")
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

def get_num_cpus():
    """
    Get the number of CPUs available in the Ray cluster
    
    Returns:
        int: Number of CPUs available
    """
    if not ray.is_initialized():
        logger.warning("Ray is not initialized, returning 0 CPUs")
        return 0
        
    return int(ray.cluster_resources().get("CPU", 0))

def get_cluster_status():
    """
    Get the status of the Ray cluster
    
    Returns:
        dict: Cluster status information
    """
    if not ray.is_initialized():
        logger.warning("Ray is not initialized")
        return {"status": "not_initialized"}
        
    try:
        nodes = ray.nodes()
        resources = ray.cluster_resources()
        
        return {
            "status": "initialized",
            "num_nodes": len(nodes),
            "resources": resources,
            "nodes": nodes
        }
    except Exception as e:
        logger.error(f"Error getting cluster status: {e}")
        return {"status": "error", "message": str(e)}
