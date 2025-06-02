import lmstudio as lms
import logging
import time
import os
from datetime import datetime

logger = logging.getLogger(__name__)

# 全局變量用於存儲LM Studio實例
_lm_studio_instance = None
_last_connection_time = None
_connection_timeout = 300  # 5分鐘超時

def reset_lm_studio_instance():
    """重置LM Studio實例"""
    global _lm_studio_instance, _last_connection_time
    
    try:
        # 如果存在實例，嘗試關閉連接
        if _lm_studio_instance is not None:
            try:
                # 嘗試關閉任何打開的連接
                if hasattr(_lm_studio_instance, 'close'):
                    _lm_studio_instance.close()
            except Exception as e:
                logger.warning(f"關閉LM Studio實例時出錯: {str(e)}")
    finally:
        # 無論如何都重置實例
        _lm_studio_instance = None
        _last_connection_time = None
        
        # 重置默認客戶端
        try:
            from lmstudio.sync_api import _reset_default_client
            _reset_default_client()
        except Exception as e:
            logger.warning(f"重置LM Studio默認客戶端時出錯: {str(e)}")
        
        # 重新導入模塊以重置客戶端
        try:
            import importlib
            importlib.reload(lms)
        except Exception as e:
            logger.warning(f"重新初始化LM Studio客戶端時出錯: {str(e)}")

def get_lm_studio_instance(config):
    """獲取LM Studio實例（單例模式）"""
    global _lm_studio_instance, _last_connection_time
    
    current_time = time.time()
    
    # 檢查是否需要重新連接
    if (_lm_studio_instance is None or 
        _last_connection_time is None or 
        current_time - _last_connection_time > _connection_timeout):
        
        # 重置現有實例
        reset_lm_studio_instance()
        
        for attempt in range(config['max_retries']):
            try:
                # 配置客戶端
                lms.configure_default_client(config['base_url'])
                model = lms.llm(config['model_name'])
                _lm_studio_instance = model
                _last_connection_time = current_time
                logger.info(f"已連接到LM Studio (嘗試 {attempt + 1}/{config['max_retries']})")
                return _lm_studio_instance
            except Exception as e:
                if attempt < config['max_retries'] - 1:
                    wait_time = config['retry_delay'] * (attempt + 1)
                    logger.info(f"連接失敗，{wait_time}秒後重試...")
                    time.sleep(wait_time)
                    reset_lm_studio_instance()
                else:
                    logger.error("無法建立連接")
                    return None
    
    return _lm_studio_instance

def generate_prompt(content, prompt_template, config):
    """生成圖片提示詞"""
    for attempt in range(config['max_retries']):
        try:
            model = get_lm_studio_instance(config)
            if model is None:
                raise Exception("LM Studio模型未初始化")

            # 設置超時時間（秒）
            timeout = config.get('timeout', 30)
            
            # 使用線程來執行生成任務
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
            
            # 創建並啟動生成線程
            thread = threading.Thread(target=generate_task)
            thread.daemon = True
            thread.start()
            
            # 等待結果或超時
            thread.join(timeout)
            
            if thread.is_alive():
                logger.error(f"生成提示詞超時 ({timeout}秒)")
                reset_lm_studio_instance()
                raise Exception(f"生成提示詞超時 ({timeout}秒)")
            
            # 檢查是否有錯誤
            if not error_queue.empty():
                raise error_queue.get()
            
            # 獲取結果
            if result_queue.empty():
                raise Exception("生成提示詞失敗：未獲得結果")
                
            result = result_queue.get()
            reset_lm_studio_instance()
            return result
            
        except Exception as e:
            if attempt < config['max_retries'] - 1:
                wait_time = config['retry_delay'] * (attempt + 1)
                logger.info(f"生成失敗，{wait_time}秒後重試...")
                time.sleep(wait_time)
                if "ECONNRESET" in str(e) or "connection" in str(e).lower():
                    reset_lm_studio_instance()
            else:
                raise Exception("無法生成提示詞")

def save_prompt(prompt_text, output_dir):
    """保存生成的提示詞"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_dir, exist_ok=True)
    
    filename = os.path.join(output_dir, f"prompt_{timestamp}.txt")
    
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(prompt_text)
        
        logger.info(f"提示詞已保存至: {filename}")
        return filename
    except Exception as e:
        raise Exception(f"保存提示詞時出錯: {str(e)}") 