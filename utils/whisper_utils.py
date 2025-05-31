import torch
from faster_whisper import WhisperModel
import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)

__all__ = ['transcribe_audio', 'save_transcription', 'load_whisper_model']

def load_whisper_model(config):
    """載入Whisper模型"""
    try:
        logger.info(f"正在載入Whisper模型 {config['model_name']}...")
        
        # 記錄CUDA信息
        if torch.cuda.is_available():
            logger.info(f"CUDA版本: {torch.version.cuda}")
            logger.info(f"cuDNN版本: {torch.backends.cudnn.version()}")
            logger.info(f"GPU設備: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU內存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # 初始化Whisper模型
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
        
        logger.info("Whisper模型載入成功")
        return model
        
    except Exception as e:
        logger.error(f"無法載入Whisper模型 '{config['model_name']}': {e}")
        logger.error("請確保模型路徑正確且模型文件可訪問")
        return None

def transcribe_audio(audio_path, config):
    """使用Whisper轉錄音頻"""
    logger.info("開始轉錄音頻...")
    model = load_whisper_model(config)
    if not model:
        logger.error("錯誤：無法載入Whisper模型")
        return None
    
    try:
        # 使用優化參數進行轉錄
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
        
        # 處理不同類型的返回結果
        if hasattr(result, 'text'):
            return result.text.strip()
        elif isinstance(result, tuple) and len(result) > 0:
            segments = list(result[0])
            if segments:
                return " ".join(segment.text for segment in segments).strip()
            else:
                logger.warning("沒有檢測到任何語音片段")
                return ""
        elif isinstance(result, dict):
            text = result.get('text', '')
            return text.strip() if text else ""
        else:
            logger.warning(f"未知的轉錄結果格式: {type(result)}")
            return str(result).strip()
            
    except Exception as e:
        logger.error(f"轉錄音頻時出錯: {e}")
        return None

def save_transcription(text, audio_filename, output_dir):
    """保存轉錄文本"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_dir, exist_ok=True)
    
    filename = os.path.join(output_dir, f"transcription_{timestamp}.txt")
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"原始音頻文件: {os.path.basename(audio_filename)}\n")
            f.write(f"轉錄時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("\n轉錄內容:\n")
            f.write(text)
        
        logger.info(f"轉錄文本已保存至: {filename}")
        return filename
    except Exception as e:
        raise Exception(f"保存轉錄文本時出錯: {str(e)}") 