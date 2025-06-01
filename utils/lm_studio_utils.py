import lmstudio as lms
import logging
import time
import os
from datetime import datetime

logger = logging.getLogger(__name__)

# Global variables for storing LM Studio instance
_lm_studio_instance = None
_last_connection_time = None
_connection_timeout = 300  # 5 minutes timeout

def reset_lm_studio_instance():
    """Reset LM Studio instance"""
    global _lm_studio_instance, _last_connection_time
    
    try:
        # If instance exists, try to close connection
        if _lm_studio_instance is not None:
            try:
                # Try to close any open connections
                if hasattr(_lm_studio_instance, 'close'):
                    _lm_studio_instance.close()
            except Exception as e:
                logger.warning(f"Error closing LM Studio instance: {str(e)}")
    finally:
        # Reset instance regardless
        _lm_studio_instance = None
        _last_connection_time = None
        
        # Reset default client
        try:
            from lmstudio.sync_api import _reset_default_client
            _reset_default_client()
        except Exception as e:
            logger.warning(f"Error resetting LM Studio default client: {str(e)}")
        
        # Re-import module to reset client
        try:
            import importlib
            importlib.reload(lms)
        except Exception as e:
            logger.warning(f"Error reinitializing LM Studio client: {str(e)}")

def get_lm_studio_instance(config):
    """Get LM Studio instance (singleton pattern)"""
    global _lm_studio_instance, _last_connection_time
    
    current_time = time.time()
    
    # Check if reconnection is needed
    if (_lm_studio_instance is None or 
        _last_connection_time is None or 
        current_time - _last_connection_time > _connection_timeout):
        
        # Reset existing instance
        reset_lm_studio_instance()
        
        for attempt in range(config['max_retries']):
            try:
                # Configure client
                lms.configure_default_client(config['base_url'])
                model = lms.llm(config['model_name'])
                _lm_studio_instance = model
                _last_connection_time = current_time
                logger.info(f"Connected to LM Studio (Attempt {attempt + 1}/{config['max_retries']})")
                return _lm_studio_instance
            except Exception as e:
                if attempt < config['max_retries'] - 1:
                    wait_time = config['retry_delay'] * (attempt + 1)
                    logger.info(f"Connection failed, retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    reset_lm_studio_instance()
                else:
                    logger.error("Unable to establish connection")
                    return None
    
    return _lm_studio_instance

def generate_prompt(content, prompt_template, config):
    """Generate image prompt"""
    for attempt in range(config['max_retries']):
        try:
            model = get_lm_studio_instance(config)
            if model is None:
                raise Exception("LM Studio model not initialized")

            # Set timeout (seconds)
            timeout = config.get('timeout', 30)
            
            # Use thread to execute generation task
            import threading
            import queue
            
            result_queue = queue.Queue()
            error_queue = queue.Queue()
            
            def generate_task():
                try:
                    response = model.complete(prompt_template.format(content=content))
                    result_queue.put(str(response).strip())
                except Exception as e:
                    error_queue.put(e)
            
            # Create and start generation thread
            thread = threading.Thread(target=generate_task)
            thread.daemon = True
            thread.start()
            
            # Wait for result or timeout
            thread.join(timeout)
            
            if thread.is_alive():
                logger.error(f"Prompt generation timed out ({timeout} seconds)")
                reset_lm_studio_instance()
                raise Exception(f"Prompt generation timed out ({timeout} seconds)")
            
            # Check for errors
            if not error_queue.empty():
                raise error_queue.get()
            
            # Get result
            if result_queue.empty():
                raise Exception("Prompt generation failed: No result received")
                
            result = result_queue.get()
            reset_lm_studio_instance()
            return result
            
        except Exception as e:
            if attempt < config['max_retries'] - 1:
                wait_time = config['retry_delay'] * (attempt + 1)
                logger.info(f"Generation failed, retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                if "ECONNRESET" in str(e) or "connection" in str(e).lower():
                    reset_lm_studio_instance()
            else:
                raise Exception("Unable to generate prompt")

def save_prompt(prompt_text, output_dir):
    """Save generated prompt"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_dir, exist_ok=True)
    
    filename = os.path.join(output_dir, f"prompt_{timestamp}.txt")
    
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(prompt_text)
        
        logger.info(f"Prompt saved to: {filename}")
        return filename
    except Exception as e:
        raise Exception(f"Error saving prompt: {str(e)}") 