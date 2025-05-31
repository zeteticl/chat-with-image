import sounddevice as sd
import numpy as np
import wave
import os
from datetime import datetime
import queue
import logging

logger = logging.getLogger(__name__)

__all__ = ['monitor_audio', 'save_audio', 'get_device_info', 'find_stereo_mix_device', 'list_audio_devices']

def get_device_info(device_id):
    """獲取音頻設備信息"""
    try:
        device_info = sd.query_devices(device_id)
        logger.info(f"設備信息: ID:{device_id} {device_info['name']} (輸入通道: {device_info['max_input_channels']}, 輸出通道: {device_info['max_output_channels']}, 採樣率: {device_info['default_samplerate']}Hz)")
        return device_info
    except Exception as e:
        logger.error(f"獲取設備信息時出錯: {str(e)}")
        return None

def find_stereo_mix_device():
    """查找立體聲混音設備"""
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        # 檢查設備名稱中是否包含關鍵字
        if any(keyword in device['name'].lower() for keyword in ['stereo mix', '立體聲混音', 'what u hear', 'what you hear']) and device['max_input_channels'] > 0:
            # 驗證設備是否可用
            try:
                sd.check_input_settings(device=i)
                return i
            except Exception as e:
                logger.warning(f"設備 {device['name']} 不可用: {str(e)}")
                continue
    return None

def monitor_audio(config):
    """監聽音頻輸出"""
    # 獲取所有設備
    devices = sd.query_devices()
    
    # 首先嘗試找到立體聲混音設備
    stereo_mix_id = find_stereo_mix_device()
    
    if stereo_mix_id is None:
        # 如果找不到立體聲混音，則顯示所有輸入設備供選擇
        input_devices = []
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                try:
                    # 驗證設備是否可用
                    sd.check_input_settings(device=i)
                    input_devices.append((i, device))
                except Exception as e:
                    logger.warning(f"設備 {device['name']} 不可用: {str(e)}")
                    continue
        
        if not input_devices:
            raise Exception("找不到任何可用的音頻輸入設備")
        
        logger.info("\n找不到立體聲混音設備，請選擇其他輸入設備:")
        for i, device in input_devices:
            logger.info(f"設備 {i}: {device['name']} (輸入通道: {device['max_input_channels']}, 輸出通道: {device['max_output_channels']}, 採樣率: {device['default_samplerate']}Hz)")
        
        while True:
            try:
                device_id = int(input("\n請選擇設備ID (輸入數字): "))
                if any(i == device_id for i, _ in input_devices):
                    break
                else:
                    logger.warning("無效的設備ID，請重新選擇")
            except ValueError:
                logger.warning("請輸入有效的數字")
    else:
        device_id = stereo_mix_id
        logger.info(f"找到立體聲混音設備: ID:{device_id} {devices[device_id]['name']}")

    # 獲取設備信息
    device_info = get_device_info(device_id)
    if not device_info:
        raise Exception("無法獲取設備信息")

    # 使用設備的默認採樣率
    sample_rate = int(device_info['default_samplerate'])
    channels = config['channels']
    duration = config['duration']

    logger.info(f"開始監聽 {duration} 秒的音頻... 使用設備: ID:{device_id} {device_info['name']} (採樣率: {sample_rate}Hz, 通道數: {channels})")

    # 創建隊列來存儲音頻數據
    q = queue.Queue()
    recording = []

    def audio_callback(indata, frames, time, status):
        """音頻回調函數"""
        if status:
            logger.warning(f"狀態: {status}")
        q.put(indata.copy())

    try:
        # 驗證設備設置
        sd.check_input_settings(
            device=device_id,
            channels=channels,
            samplerate=sample_rate
        )
        
        # 創建輸入流
        with sd.InputStream(
            device=device_id,
            channels=channels,
            samplerate=sample_rate,
            callback=audio_callback
        ):
            logger.info("開始監聽...")
            # 等待指定的時間
            sd.sleep(int(duration * 1000))
            logger.info("監聽完成！")

        # 從隊列中獲取所有數據
        while not q.empty():
            recording.append(q.get())

        # 將數據轉換為numpy數組
        if recording:
            recording = np.concatenate(recording, axis=0)
            return recording, sample_rate
        else:
            raise Exception("沒有捕獲到音頻數據")

    except Exception as e:
        raise Exception(f"監聽音頻時出錯: {str(e)}")

def save_audio(recording, sample_rate, output_dir):
    """保存音頻為WAV文件"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_dir, exist_ok=True)
    
    filename = os.path.join(output_dir, f"recording_{timestamp}.wav")
    
    # 將float32轉換為int16
    audio_data = (recording * 32767).astype(np.int16)
    
    try:
        with wave.open(filename, 'wb') as wf:
            # 設置WAV文件參數
            wf.setparams((
                recording.shape[1] if len(recording.shape) > 1 else 1,  # 通道數
                2,  # 樣本寬度（字節）
                sample_rate,  # 採樣率
                len(audio_data),  # 幀數
                'NONE',  # 壓縮類型
                'NONE'  # 壓縮名稱
            ))
            # 寫入音頻數據
            wf.writeframes(audio_data.tobytes())
        
        logger.info(f"音頻已保存至: {filename}")
        return filename
    except Exception as e:
        raise Exception(f"保存音頻文件時出錯: {str(e)}")

def list_audio_devices():
    """列出所有可用的音頻設備"""
    logger.info("可用的音頻設備:")
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:  # 只顯示有輸入通道的設備
            logger.info(f"設備 {i}: {device['name']}") 