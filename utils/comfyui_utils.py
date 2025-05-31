import json
import os
import time
import websocket
import urllib.request
import urllib.parse
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class ComfyUI:
    def __init__(self, config):
        """初始化ComfyUI連接"""
        self.server_address = config['server_address']
        self.client_id = config['client_id']
        self.ws = None
        logger.info(f"正在連接到ComfyUI服務器: {self.server_address}")

    def open_websocket_connection(self):
        """建立WebSocket連接"""
        self.ws = websocket.WebSocket()
        self.ws.connect(f"ws://{self.server_address}/ws?clientId={self.client_id}")
        return self.ws

    def queue_prompt(self, prompt):
        """發送提示到ComfyUI工作流程"""
        p = {"prompt": prompt, "client_id": self.client_id}
        headers = {'Content-Type': 'application/json'}
        data = json.dumps(p).encode('utf-8')
        req = urllib.request.Request(
            f"http://{self.server_address}/prompt",
            data=data,
            headers=headers
        )
        return json.loads(urllib.request.urlopen(req).read())

    def get_history(self, prompt_id):
        """獲取指定提示ID的歷史記錄"""
        with urllib.request.urlopen(f"http://{self.server_address}/history/{prompt_id}") as response:
            return json.loads(response.read())

    def get_image(self, filename, subfolder, folder_type):
        """從ComfyUI獲取圖片"""
        data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        url_values = urllib.parse.urlencode(data)
        with urllib.request.urlopen(f"http://{self.server_address}/view?{url_values}") as response:
            return response.read()

    def track_progress(self, prompt, prompt_id):
        """追蹤生成進度"""
        node_ids = list(prompt.keys())
        finished_nodes = []
        last_progress = 0

        while True:
            out = self.ws.recv()
            if isinstance(out, str):
                message = json.loads(out)
                if message['type'] == 'progress':
                    data = message['data']
                    current_step = data['value']
                    print(f'K-Sampler進度 -> 步驟: {current_step}/{data["max"]}', end='\r', flush=True)
                if message['type'] == 'execution_cached':
                    data = message['data']
                    for itm in data['nodes']:
                        if itm not in finished_nodes:
                            finished_nodes.append(itm)
                            progress = len(finished_nodes)
                            if progress != last_progress:
                                print(f'進度: {progress}/{len(node_ids)} 任務完成', end='\r', flush=True)
                                last_progress = progress
                if message['type'] == 'executing':
                    data = message['data']
                    if data['node'] not in finished_nodes:
                        finished_nodes.append(data['node'])
                        progress = len(finished_nodes)
                        if progress != last_progress:
                            print(f'進度: {progress}/{len(node_ids)} 任務完成', end='\r', flush=True)
                            last_progress = progress
                    if data['node'] is None and data['prompt_id'] == prompt_id:
                        print()  # 換行
                        break
            else:
                continue

    def generate_image(self, workflow_path, prompt_text, output_dir, config):
        """生成圖片的主函數"""
        try:
            # 讀取工作流程
            with open(workflow_path, 'r', encoding='utf-8') as f:
                workflow = json.load(f)

            # 更新提示文字
            for node_id, node in workflow.items():
                if node.get('class_type') == 'CLIPTextEncode' and node.get('_meta', {}).get('title') == 'CLIP Text Encode (Positive Prompt)':
                    node['inputs']['text'] = prompt_text
                    break

            # 建立輸出目錄
            os.makedirs(output_dir, exist_ok=True)

            # 建立WebSocket連接
            self.open_websocket_connection()

            # 發送提示
            prompt_id = self.queue_prompt(workflow)['prompt_id']

            # 追蹤進度
            self.track_progress(workflow, prompt_id)

            # 獲取生成的圖片
            history = self.get_history(prompt_id)[prompt_id]
            for node_id in history['outputs']:
                node_output = history['outputs'][node_id]
                if 'images' in node_output:
                    for image in node_output['images']:
                        if image['type'] == 'output':
                            image_data = self.get_image(
                                image['filename'],
                                image['subfolder'],
                                image['type']
                            )
                            
                            # 使用時間戳記作為檔名
                            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                            output_path = os.path.join(output_dir, f'comfyui_{timestamp}.png')
                            
                            # 儲存圖片
                            with open(output_path, 'wb') as f:
                                f.write(image_data)
                            logger.info(f'圖片已保存至: {output_path}')
                            return output_path

        finally:
            if self.ws:
                self.ws.close()

def load_workflow_template(workflow_path):
    """載入工作流程模板"""
    try:
        with open(workflow_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"載入工作流程模板時出錯: {e}")
        return None 