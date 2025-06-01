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
        """Initialize ComfyUI connection"""
        self.server_address = config['server_address']
        self.client_id = config['client_id']
        self.ws = None
        logger.info(f"Connecting to ComfyUI server: {self.server_address}")

    def open_websocket_connection(self):
        """Establish WebSocket connection"""
        self.ws = websocket.WebSocket()
        self.ws.connect(f"ws://{self.server_address}/ws?clientId={self.client_id}")
        return self.ws

    def queue_prompt(self, prompt):
        """Send prompt to ComfyUI workflow"""
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
        """Get history for specified prompt ID"""
        with urllib.request.urlopen(f"http://{self.server_address}/history/{prompt_id}") as response:
            return json.loads(response.read())

    def get_image(self, filename, subfolder, folder_type):
        """Get image from ComfyUI"""
        data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        url_values = urllib.parse.urlencode(data)
        with urllib.request.urlopen(f"http://{self.server_address}/view?{url_values}") as response:
            return response.read()

    def track_progress(self, prompt, prompt_id):
        """Track generation progress"""
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
                    print(f'K-Sampler Progress -> Step: {current_step}/{data["max"]}', end='\r', flush=True)
                if message['type'] == 'execution_cached':
                    data = message['data']
                    for itm in data['nodes']:
                        if itm not in finished_nodes:
                            finished_nodes.append(itm)
                            progress = len(finished_nodes)
                            if progress != last_progress:
                                print(f'Progress: {progress}/{len(node_ids)} tasks completed', end='\r', flush=True)
                                last_progress = progress
                if message['type'] == 'executing':
                    data = message['data']
                    if data['node'] not in finished_nodes:
                        finished_nodes.append(data['node'])
                        progress = len(finished_nodes)
                        if progress != last_progress:
                            print(f'Progress: {progress}/{len(node_ids)} tasks completed', end='\r', flush=True)
                            last_progress = progress
                    if data['node'] is None and data['prompt_id'] == prompt_id:
                        print()  # New line
                        break
            else:
                continue

    def generate_image(self, workflow_path, prompt_text, output_dir, config):
        """Main function for image generation"""
        try:
            # Read workflow
            with open(workflow_path, 'r', encoding='utf-8') as f:
                workflow = json.load(f)

            # Update prompt text
            for node_id, node in workflow.items():
                if node.get('class_type') == 'CLIPTextEncode' and node.get('_meta', {}).get('title') == 'CLIP Text Encode (Positive Prompt)':
                    node['inputs']['text'] = prompt_text
                    break

            # Create output directory
            os.makedirs(output_dir, exist_ok=True)

            # Establish WebSocket connection
            self.open_websocket_connection()

            # Send prompt
            prompt_id = self.queue_prompt(workflow)['prompt_id']

            # Track progress
            self.track_progress(workflow, prompt_id)

            # Get generated image
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
                            
                            # Use timestamp as filename
                            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                            output_path = os.path.join(output_dir, f'comfyui_{timestamp}.png')
                            
                            # Save image
                            with open(output_path, 'wb') as f:
                                f.write(image_data)
                            logger.info(f'Image saved to: {output_path}')
                            return output_path

        finally:
            if self.ws:
                self.ws.close()

def load_workflow_template(workflow_path):
    """Load workflow template"""
    try:
        with open(workflow_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading workflow template: {e}")
        return None 