import torch
from faster_whisper import WhisperModel
import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)

__all__ = ['transcribe_audio', 'save_transcription', 'load_whisper_model']

def load_whisper_model(config):
    """Load Whisper model"""
    try:
        logger.info(f"Loading Whisper model {config['model_name']}...")
        
        # Only log GPU info on first load
        if not hasattr(load_whisper_model, '_gpu_info_logged'):
            if torch.cuda.is_available():
                logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            load_whisper_model._gpu_info_logged = True
        
        # Initialize Whisper model
        model = WhisperModel(
            config['model_name'],
            device="cuda" if torch.cuda.is_available() else "cpu",
            compute_type="float16",
            device_index=0,
            cpu_threads=4,
            num_workers=4,
            download_root=config['model_dir'],
            local_files_only=False
        )
        
        logger.info("Whisper model loaded successfully")
        return model
        
    except Exception as e:
        logger.error(f"Unable to load Whisper model: {e}")
        return None

def transcribe_audio(audio_path, config):
    """Transcribe audio using Whisper"""
    logger.info("Starting audio transcription...")
    model = load_whisper_model(config)
    if not model:
        logger.error("Error: Unable to load Whisper model")
        return None
    
    try:
        # Use optimized parameters for transcription
        result = model.transcribe(
            audio_path,
            language=config['language'],
            task=config['task'],
            beam_size=8,
            best_of=8,
            patience=2,
            length_penalty=1.0,
            repetition_penalty=1.0,
            no_repeat_ngram_size=0,
            temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.3,
            condition_on_previous_text=True,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=1000),
            chunk_length=30,
            max_new_tokens=128
        )
        
        # Handle different types of return results
        if hasattr(result, 'text'):
            return result.text.strip()
        elif isinstance(result, tuple) and len(result) > 0:
            segments = list(result[0])
            if segments:
                return " ".join(segment.text for segment in segments).strip()
            else:
                logger.warning("No speech segments detected")
                return ""
        elif isinstance(result, dict):
            text = result.get('text', '')
            return text.strip() if text else ""
        else:
            logger.warning(f"Unknown transcription result format: {type(result)}")
            return str(result).strip()
            
    except Exception as e:
        logger.error(f"Error transcribing audio: {e}")
        return None

def save_transcription(text, audio_filename, output_dir):
    """Save transcription text"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_dir, exist_ok=True)
    
    filename = os.path.join(output_dir, f"transcription_{timestamp}.txt")
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"Original audio file: {os.path.basename(audio_filename)}\n")
            f.write(f"Transcription time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("\nTranscription content:\n")
            f.write(text)
        
        logger.info(f"Transcription saved to: {filename}")
        return filename
    except Exception as e:
        raise Exception(f"Error saving transcription: {str(e)}") 