# Chat with Image

A powerful Python application that converts audio recordings into AI-generated images through a multi-step process involving speech recognition, text processing, and image generation.

## Features

- **Audio Recording**: Captures system audio with configurable duration and quality settings
- **Speech Recognition**: Uses Whisper AI for accurate speech-to-text conversion
- **Text Processing**: Leverages LM Studio for intelligent text analysis and prompt generation
- **Image Generation**: Integrates with ComfyUI for high-quality AI image generation
- **Parallel Processing**: Implements multi-threading for efficient audio recording and image generation
- **Robust Error Handling**: Includes comprehensive error handling and automatic retry mechanisms
- **Configurable Settings**: Easy-to-customize configuration for all components

## Prerequisites

- Python 3.8 or higher
- NVIDIA GPU with CUDA support (recommended for optimal performance)
- Audio input device (microphone or system audio capture)
- ComfyUI server running locally
- LM Studio server running locally

## Installation

1. Clone the repository:

```bash
git clone https://github.com/zeteticl/chat-with-image.git
cd chat-with-image
```

2. Install required Python packages:

```bash
pip install -r requirements.txt
```

3. Download required models:

- Whisper model (specified in config.py)
- LM Studio model (specified in config.py)

4. Configure ComfyUI:

- Ensure ComfyUI server is running on the specified address
- Place your workflow JSON file in the comfyui directory

## Configuration

Edit `config.py` to customize:

- Audio recording settings (duration, channels, sample rate)
- Whisper model settings (model name, language, task)
- LM Studio settings (API address, model name)
- ComfyUI settings (server address, workflow path)
- Output directory settings
- File deletion preferences
- Story background and prompt template

## Usage

1. Start the application:

```bash
python main.py
```

2. Select your audio input device when prompted
3. The application will:

   - Record audio for the specified duration
   - Convert speech to text using Whisper
   - Generate an image prompt using LM Studio
   - Create an image using ComfyUI
   - Save all outputs in their respective directories
4. Press Ctrl+C to stop the program

## Project Structure

```
chat-with-image/
├── main.py                 # Main application entry point
├── config.py              # Configuration settings
├── utils/
│   ├── audio_utils.py     # Audio recording and processing
│   ├── whisper_utils.py   # Speech recognition
│   ├── lm_studio_utils.py # Text processing and prompt generation
│   └── comfyui_utils.py   # Image generation
├── models/                # AI model storage
├── output/
│   ├── audio/            # Recorded audio files
│   ├── text/             # Transcriptions and prompts
│   └── images/           # Generated images
└── logs/                 # Application logs
```

## Error Handling

The application includes robust error handling for:

- Audio device selection and recording
- Speech recognition failures
- Text processing timeouts
- Image generation errors
- File system operations

Automatic retry mechanisms are implemented for critical operations.

## Logging

Comprehensive logging is implemented throughout the application:

- Log files are stored in the `logs` directory
- Console output for real-time monitoring
- Detailed error tracking and debugging information

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Whisper AI](https://github.com/openai/whisper) for speech recognition
- [LM Studio](https://lmstudio.ai/) for text processing
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) for image generation
