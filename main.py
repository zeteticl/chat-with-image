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

# 配置日誌
def setup_logging():
    """設置統一的日誌配置"""
    # 創建日誌目錄
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # 配置根日誌記錄器
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'app.log'), encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    # 設置其他模塊的日誌級別
    logging.getLogger('websocket').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    
    # 禁用其他模塊的日誌配置
    for module in ['utils.audio_utils', 'utils.whisper_utils', 'utils.lm_studio_utils', 'utils.comfyui_utils']:
        module_logger = logging.getLogger(module)
        module_logger.propagate = True
        module_logger.handlers = []

# 設置日誌
setup_logging()
logger = logging.getLogger(__name__)

def setup_directories():
    """設置必要的目錄"""
    for dir_name in [config.OUTPUT_CONFIG['audio_dir'], 
                    config.OUTPUT_CONFIG['text_dir'], 
                    config.OUTPUT_CONFIG['image_dir']]:
        full_path = os.path.join(config.OUTPUT_CONFIG['base_dir'], dir_name)
        os.makedirs(full_path, exist_ok=True)
        logger.info(f"已創建目錄: {full_path}")

def process_audio_to_image():
    """處理從音頻到圖片的完整流程"""
    try:
        # 1. 監聽音頻
        logger.info("開始監聽音頻...")
        recording, sample_rate = monitor_audio(config.AUDIO_CONFIG)
        
        # 2. 保存音頻文件
        audio_file = save_audio(
            recording, 
            sample_rate, 
            os.path.join(config.OUTPUT_CONFIG['base_dir'], config.OUTPUT_CONFIG['audio_dir'])
        )
        logger.info(f"音頻已保存: {audio_file}")
        
        # 3. 轉錄音頻（添加超時機制）
        transcription = None
        timeout_occurred = False

        def timeout_handler():
            nonlocal timeout_occurred
            timeout_occurred = True

        timer = threading.Timer(300, timeout_handler)  # 5分鐘超時
        timer.start()

        try:
            transcription = transcribe_audio(audio_file, config.WHISPER_CONFIG)
            if timeout_occurred:
                raise TimeoutError("轉錄過程超時")
            if not transcription:
                raise Exception("音頻轉錄失敗")
        finally:
            timer.cancel()
        
        # 4. 保存轉錄文本
        transcription_file = save_transcription(
            transcription,
            audio_file,
            os.path.join(config.OUTPUT_CONFIG['base_dir'], config.OUTPUT_CONFIG['text_dir'])
        )
        logger.info(f"轉錄文本已保存: {transcription_file}")
        
        # 5. 生成圖片提示詞
        max_prompt_attempts = 3
        prompt = None
        
        for attempt in range(max_prompt_attempts):
            try:
                # 組合故事背景和轉錄文本
                content = f"{config.STORY_BACKGROUND}\n\n{transcription}"
                prompt = generate_prompt(content, config.PROMPT_TEMPLATE, config.LM_STUDIO_CONFIG)
                break
            except Exception as e:
                logger.error(f"生成提示詞失敗 (嘗試 {attempt + 1}/{max_prompt_attempts}): {str(e)}")
                if attempt < max_prompt_attempts - 1:
                    wait_time = 5 * (attempt + 1)  # 指數退避
                    logger.info(f"等待 {wait_time} 秒後重試...")
                    time.sleep(wait_time)
                    # 重置 LM Studio 實例
                    reset_lm_studio_instance()
                else:
                    raise Exception("無法生成提示詞，已達到最大重試次數")
        
        if not prompt:
            raise Exception("無法生成提示詞")
        
        # 6. 保存提示詞
        prompt_file = save_prompt(
            prompt,
            os.path.join(config.OUTPUT_CONFIG['base_dir'], config.OUTPUT_CONFIG['text_dir'])
        )
        logger.info(f"提示詞已保存: {prompt_file}")
        
        # 7. 生成圖片
        comfy = ComfyUI(config.COMFYUI_CONFIG)
        image_path = comfy.generate_image(
            config.COMFYUI_CONFIG['workflow_path'],
            prompt,
            os.path.join(config.OUTPUT_CONFIG['base_dir'], config.OUTPUT_CONFIG['image_dir']),
            config.COMFYUI_CONFIG
        )
        
        if image_path:
            logger.info(f"圖片生成成功: {image_path}")
            return {
                'audio_file': audio_file,
                'transcription_file': transcription_file,
                'prompt_file': prompt_file,
                'image_file': image_path
            }
        else:
            raise Exception("圖片生成失敗")
            
    except Exception as e:
        logger.error(f"處理過程中出錯: {str(e)}")
        raise

def main():
    """主程序入口"""
    try:
        # 顯示所有音頻輸入設備
        list_audio_devices()
        logger.info("=== 設備列表結束 ===\n")
        
        # 設置目錄
        setup_directories()
        
        logger.info("程式已啟動，開始循環處理音頻到圖片的轉換...")
        logger.info("按 Ctrl+C 可以停止程式")
        
        consecutive_errors = 0
        max_consecutive_errors = 3
        
        while True:
            try:
                # 執行主流程
                results = process_audio_to_image()
                
                # 重置連續錯誤計數
                consecutive_errors = 0
                
                # 輸出結果摘要
                logger.info("\n=== 處理完成 ===")
                logger.info(f"音頻文件: {results['audio_file']}")
                logger.info(f"轉錄文本: {results['transcription_file']}")
                logger.info(f"提示詞文件: {results['prompt_file']}")
                logger.info(f"生成圖片: {results['image_file']}")
                
                # 添加短暫延遲，避免CPU使用率過高
                time.sleep(1)
                
            except KeyboardInterrupt:
                logger.info("\n使用者中斷程式執行")
                break
            except Exception as e:
                consecutive_errors += 1
                logger.error(f"處理過程中出錯: {str(e)}")
                
                if consecutive_errors >= max_consecutive_errors:
                    logger.error(f"連續錯誤次數達到 {max_consecutive_errors} 次，重置 LM Studio 連接...")
                    reset_lm_studio_instance()
                    consecutive_errors = 0
                
                wait_time = 5 * consecutive_errors  # 指數退避
                logger.info(f"等待 {wait_time} 秒後重試...")
                time.sleep(wait_time)
                continue
        
        logger.info("程式已停止")
        
    except Exception as e:
        logger.error(f"程式執行失敗: {str(e)}")
        raise

if __name__ == "__main__":
    main() 