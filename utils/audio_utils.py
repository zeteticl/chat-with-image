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
    """Get audio device information"""
    try:
        device_info = sd.query_devices(device_id)
        logger.info(f"Device info: ID:{device_id} {device_info['name']} (Input channels: {device_info['max_input_channels']}, Output channels: {device_info['max_output_channels']}, Sample rate: {device_info['default_samplerate']}Hz)")
        return device_info
    except Exception as e:
        logger.error(f"Error getting device info: {str(e)}")
        return None

def find_stereo_mix_device():
    """Find stereo mix device"""
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        # Check if device name contains keywords
        if any(keyword in device['name'].lower() for keyword in ['stereo mix', 'what u hear', 'what you hear']) and device['max_input_channels'] > 0:
            # Verify if device is available
            try:
                sd.check_input_settings(device=i)
                return i
            except Exception as e:
                logger.warning(f"Device {device['name']} is not available: {str(e)}")
                continue
    return None

def monitor_audio(config):
    """Monitor audio output"""
    # Get all devices
    devices = sd.query_devices()
    
    # First try to find stereo mix device
    stereo_mix_id = find_stereo_mix_device()
    
    if stereo_mix_id is None:
        # If stereo mix is not found, display all input devices for selection
        input_devices = []
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                try:
                    # Verify if device is available
                    sd.check_input_settings(device=i)
                    input_devices.append((i, device))
                except Exception as e:
                    logger.warning(f"Device {device['name']} is not available: {str(e)}")
                    continue
        
        if not input_devices:
            raise Exception("No available audio input devices found")
        
        logger.info("\nStereo mix device not found, please select another input device:")
        for i, device in input_devices:
            logger.info(f"Device {i}: {device['name']} (Input channels: {device['max_input_channels']}, Output channels: {device['max_output_channels']}, Sample rate: {device['default_samplerate']}Hz)")
        
        while True:
            try:
                device_id = int(input("\nPlease select device ID (enter number): "))
                if any(i == device_id for i, _ in input_devices):
                    break
                else:
                    logger.warning("Invalid device ID, please select again")
            except ValueError:
                logger.warning("Please enter a valid number")
    else:
        device_id = stereo_mix_id
        logger.info(f"Found stereo mix device: ID:{device_id} {devices[device_id]['name']}")

    # Get device information
    device_info = get_device_info(device_id)
    if not device_info:
        raise Exception("Unable to get device information")

    # Use device's default sample rate
    sample_rate = int(device_info['default_samplerate'])
    channels = config['channels']
    duration = config['duration']

    logger.info(f"Starting to monitor audio for {duration} seconds... Using device: ID:{device_id} {device_info['name']} (Sample rate: {sample_rate}Hz, Channels: {channels})")

    # Create queue to store audio data
    q = queue.Queue()
    recording = []

    def audio_callback(indata, frames, time, status):
        """Audio callback function"""
        if status:
            logger.warning(f"Status: {status}")
        q.put(indata.copy())

    try:
        # Verify device settings
        sd.check_input_settings(
            device=device_id,
            channels=channels,
            samplerate=sample_rate
        )
        
        # Create input stream
        with sd.InputStream(
            device=device_id,
            channels=channels,
            samplerate=sample_rate,
            callback=audio_callback
        ):
            logger.info("Starting monitoring...")
            # Wait for specified duration
            sd.sleep(int(duration * 1000))
            logger.info("Monitoring complete!")

        # Get all data from queue
        while not q.empty():
            recording.append(q.get())

        # Convert data to numpy array
        if recording:
            recording = np.concatenate(recording, axis=0)
            return recording, sample_rate
        else:
            raise Exception("No audio data captured")

    except Exception as e:
        raise Exception(f"Error monitoring audio: {str(e)}")

def save_audio(recording, sample_rate, output_dir):
    """Save audio as WAV file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_dir, exist_ok=True)
    
    filename = os.path.join(output_dir, f"recording_{timestamp}.wav")
    
    # Convert float32 to int16
    audio_data = (recording * 32767).astype(np.int16)
    
    try:
        with wave.open(filename, 'wb') as wf:
            # Set WAV file parameters
            wf.setparams((
                recording.shape[1] if len(recording.shape) > 1 else 1,  # Number of channels
                2,  # Sample width (bytes)
                sample_rate,  # Sample rate
                len(audio_data),  # Number of frames
                'NONE',  # Compression type
                'NONE'  # Compression name
            ))
            # Write audio data
            wf.writeframes(audio_data.tobytes())
        
        logger.info(f"Audio saved to: {filename}")
        return filename
    except Exception as e:
        raise Exception(f"Error saving audio file: {str(e)}")

def list_audio_devices():
    """List all available audio devices"""
    logger.info("Available audio devices:")
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:  # Only show devices with input channels
            logger.info(f"Device {i}: {device['name']}") 