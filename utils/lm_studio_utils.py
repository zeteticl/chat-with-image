import lmstudio as lms
import logging
import time
import os
from datetime import datetime

# 配置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('lm_studio.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 全局變量用於存儲LM Studio實例
_lm_studio_instance = None
_last_connection_time = None
_connection_timeout = 300  # 5分鐘超時

def reset_lm_studio_instance():
    """重置LM Studio實例"""
    global _lm_studio_instance, _last_connection_time
    _lm_studio_instance = None
    _last_connection_time = None

def get_lm_studio_instance(config):
    """獲取LM Studio實例（單例模式）"""
    global _lm_studio_instance, _last_connection_time
    
    current_time = time.time()
    
    # 檢查是否需要重新連接
    if (_lm_studio_instance is None or 
        _last_connection_time is None or 
        current_time - _last_connection_time > _connection_timeout):
        
        reset_lm_studio_instance()
        
        for attempt in range(config['max_retries']):
            try:
                # 配置客戶端
                lms.configure_default_client(config['base_url'])
                model = lms.llm(config['model_name'])
                _lm_studio_instance = model
                _last_connection_time = current_time
                logger.info(f"成功連接到LM Studio (嘗試 {attempt + 1}/{config['max_retries']})")
                return _lm_studio_instance
            except Exception as e:
                logger.error(f"連接失敗 (嘗試 {attempt + 1}/{config['max_retries']}): {str(e)}")
                if attempt < config['max_retries'] - 1:
                    wait_time = config['retry_delay'] * (attempt + 1)  # 指數退避
                    logger.info(f"等待 {wait_time} 秒後重試...")
                    time.sleep(wait_time)
                else:
                    logger.error("達到最大重試次數，無法建立連接")
                    return None
    
    return _lm_studio_instance

def generate_prompt(content, prompt_template, config):
    """生成圖片提示詞"""
    for attempt in range(config['max_retries']):
        try:
            model = get_lm_studio_instance(config)
            if model is None:
                raise Exception("LM Studio模型未初始化")

            response = model.complete(prompt_template.format(content=content))
            return str(response).strip()
        except Exception as e:
            logger.error(f"生成提示詞失敗 (嘗試 {attempt + 1}/{config['max_retries']}): {str(e)}")
            # 如果是連接錯誤，重置實例
            if "ECONNRESET" in str(e) or "connection" in str(e).lower():
                reset_lm_studio_instance()
            
            if attempt < config['max_retries'] - 1:
                wait_time = config['retry_delay'] * (attempt + 1)  # 指數退避
                logger.info(f"等待 {wait_time} 秒後重試...")
                time.sleep(wait_time)
            else:
                raise Exception("達到最大重試次數，無法生成提示詞")

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