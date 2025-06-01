import os
import logging
from datetime import datetime
import config
from utils.audio_utils import monitor_audio, save_audio, list_audio_devices
from utils.whisper_utils import transcribe_audio, save_transcription
from utils.lm_studio_utils import generate_prompt, save_prompt, reset_lm_studio_instance
from utils.comfyui_utils import ComfyUI
import time
import signal
from contextlib import contextmanager
import threading
from queue import Queue
import concurrent.futures

# Configure logging
def setup_logging():
    """Set up unified logging configuration"""
    # Create log directory
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'app.log'), encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    # Set log levels for other modules
    logging.getLogger('websocket').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    
    # Disable logging configuration for other modules
    for module in ['utils.audio_utils', 'utils.whisper_utils', 'utils.lm_studio_utils', 'utils.comfyui_utils']:
        module_logger = logging.getLogger(module)
        module_logger.propagate = True
        module_logger.handlers = []

# Set up logging
setup_logging()
logger = logging.getLogger(__name__)

def setup_directories():
    """Set up necessary directories"""
    for dir_name in [config.OUTPUT_CONFIG['audio_dir'], 
                    config.OUTPUT_CONFIG['text_dir'], 
                    config.OUTPUT_CONFIG['image_dir']]:
        full_path = os.path.join(config.OUTPUT_CONFIG['base_dir'], dir_name)
        os.makedirs(full_path, exist_ok=True)
        logger.info(f"Created directory: {full_path}")

def process_audio_to_image(audio_file):
    """Process the complete flow from audio to image"""
    try:
        # 3. Transcribe audio (add timeout mechanism)
        transcription = None
        timeout_occurred = False
        max_retries = 3
        retry_count = 0

        def timeout_handler():
            nonlocal timeout_occurred
            timeout_occurred = True
            logger.warning("Transcription process timed out, preparing to retry")

        while retry_count < max_retries:
            timeout_occurred = False
            timer = threading.Timer(120, timeout_handler)
            timer.start()

            try:
                logger.info(f"Starting audio transcription... (Attempt {retry_count + 1}/{max_retries})")
                transcription = transcribe_audio(audio_file, config.WHISPER_CONFIG)
                
                if timeout_occurred:
                    raise TimeoutError("Transcription process timed out (exceeded 120 seconds)")
                    
                if not transcription:
                    raise Exception("Audio transcription failed: No transcription result")
                    
                logger.info("Audio transcription completed")
                break  # Successfully completed, exit retry loop
                
            except TimeoutError as te:
                logger.error(f"Transcription timeout error: {str(te)}")
                retry_count += 1
                if retry_count < max_retries:
                    wait_time = 5 * retry_count  # Exponential backoff
                    logger.info(f"Waiting {wait_time} seconds before attempt {retry_count + 1}...")
                    time.sleep(wait_time)
                else:
                    logger.error("Maximum retry attempts reached, transcription failed")
                    raise Exception("Audio transcription failed: Maximum retry attempts exceeded")
            except Exception as e:
                logger.error(f"Error during transcription process: {str(e)}")
                raise
            finally:
                timer.cancel()
                if timeout_occurred:
                    logger.warning(f"Transcription attempt {retry_count + 1} was forcibly terminated")

        if not transcription:
            raise Exception("Audio transcription failed: All retry attempts unsuccessful")
        
        # 4. Save transcription text
        transcription_file = save_transcription(
            transcription,
            audio_file,
            os.path.join(config.OUTPUT_CONFIG['base_dir'], config.OUTPUT_CONFIG['text_dir'])
        )
        logger.info(f"Transcription text saved: {transcription_file}")
        
        # 5. Generate image prompt
        max_prompt_attempts = 3
        prompt = None
        
        for attempt in range(max_prompt_attempts):
            try:
                # Combine story background and transcription text
                content = f"{config.STORY_BACKGROUND}\n\n{transcription}"
                prompt = generate_prompt(content, config.PROMPT_TEMPLATE, config.LM_STUDIO_CONFIG)
                break
            except Exception as e:
                logger.error(f"Failed to generate prompt (Attempt {attempt + 1}/{max_prompt_attempts}): {str(e)}")
                if attempt < max_prompt_attempts - 1:
                    wait_time = 5 * (attempt + 1)  # Exponential backoff
                    logger.info(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                    # Reset LM Studio instance
                    reset_lm_studio_instance()
                else:
                    raise Exception("Unable to generate prompt, maximum retry attempts reached")
        
        if not prompt:
            raise Exception("Unable to generate prompt")
        
        # 6. Save prompt
        prompt_file = save_prompt(
            prompt,
            os.path.join(config.OUTPUT_CONFIG['base_dir'], config.OUTPUT_CONFIG['text_dir'])
        )
        logger.info(f"Prompt saved: {prompt_file}")
        
        # 7. Generate image
        comfy = ComfyUI(config.COMFYUI_CONFIG)
        image_path = comfy.generate_image(
            config.COMFYUI_CONFIG['workflow_path'],
            prompt,
            os.path.join(config.OUTPUT_CONFIG['base_dir'], config.OUTPUT_CONFIG['image_dir']),
            config.COMFYUI_CONFIG
        )
        
        if image_path:
            logger.info(f"Image generated successfully: {image_path}")
            return {
                'audio_file': audio_file,
                'transcription_file': transcription_file,
                'prompt_file': prompt_file,
                'image_file': image_path
            }
        else:
            raise Exception("Image generation failed")
            
    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        raise

def audio_recording_worker(task_queue):
    """Audio recording worker thread"""
    while True:
        try:
            # 1. Monitor audio
            logger.info("Starting audio monitoring...")
            recording, sample_rate = monitor_audio(config.AUDIO_CONFIG)
            
            # 2. Save audio file
            audio_file = save_audio(
                recording, 
                sample_rate, 
                os.path.join(config.OUTPUT_CONFIG['base_dir'], config.OUTPUT_CONFIG['audio_dir'])
            )
            logger.info(f"Audio saved: {audio_file}")
            
            # Add task to queue
            task_queue.put(audio_file)
            
        except Exception as e:
            logger.error(f"Error during recording process: {str(e)}")
            time.sleep(1)  # Brief wait before continuing

def image_generation_worker(task_queue):
    """Image generation worker thread"""
    while True:
        try:
            # Get task from queue
            audio_file = task_queue.get()
            if audio_file is None:
                break
                
            # Process audio to image conversion
            results = process_audio_to_image(audio_file)
            
            # Output result summary
            logger.info("\n=== Processing Complete ===")
            logger.info(f"Audio file: {results['audio_file']}")
            logger.info(f"Transcription file: {results['transcription_file']}")
            logger.info(f"Prompt file: {results['prompt_file']}")
            logger.info(f"Generated image: {results['image_file']}")
            
            task_queue.task_done()
            
        except Exception as e:
            logger.error(f"Error during image generation process: {str(e)}")
            task_queue.task_done()

def main():
    """Main program entry point"""
    try:
        # Display all audio input devices
        list_audio_devices()
        logger.info("=== End of Device List ===\n")
        
        # Set up directories
        setup_directories()
        
        logger.info("Program started, beginning parallel processing of audio to image conversion...")
        logger.info("Press Ctrl+C to stop the program")
        
        # Create task queue
        task_queue = Queue()
        
        # Create and start worker threads
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            # Start audio recording thread
            audio_future = executor.submit(audio_recording_worker, task_queue)
            
            # Start image generation thread
            image_future = executor.submit(image_generation_worker, task_queue)
            
            try:
                # Wait for threads to complete
                audio_future.result()
                image_future.result()
            except KeyboardInterrupt:
                logger.info("\nUser interrupted program execution")
                # Clear queue and add end marker
                while not task_queue.empty():
                    task_queue.get()
                task_queue.put(None)
                return
            
    except Exception as e:
        logger.error(f"Program execution failed: {str(e)}")
        logger.info("Program will automatically restart in 5 seconds...")
        time.sleep(5)
        main()  # Restart main program

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nProgram completely stopped")
    except Exception as e:
        logger.error(f"Program encountered a critical error: {str(e)}")
        logger.info("Program will automatically restart in 5 seconds...")
        time.sleep(5)
        main()  # Restart main program 