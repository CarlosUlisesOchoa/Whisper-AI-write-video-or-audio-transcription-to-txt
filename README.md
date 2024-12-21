# Whisper Audio/Video Transcription Tool 🎙️

A powerful Python-based transcription tool that leverages OpenAI's Whisper model to transcribe both audio and video files with GPU acceleration support.

## 🌟 Features

- Supports both audio and video file transcription
- Automatic audio extraction from video files
- GPU acceleration with CUDA support
- Timestamp-based transcription output
- Multi-language support
- Easy-to-use command line interface
- Saves transcriptions to text files

## 🔧 Usage Examples

Run the script with various arguments:

```bash
# Basic usage with a video file
python main.py --input video.mp4 --output transcript.txt

# Specify a different language (default is English)
python main.py --input audio.mp3 --language spanish

# Complete example with all options
python main.py --input interview.mp4 --output transcript.txt --language english
```

Available arguments:

- `--input`: Path to input audio/video file (required)
- `--output`: Path to output text file (optional, defaults to input filename + .txt)
- `--language`: Input language (optional, defaults to English)

## 🔧 Requirements

- Python 3.7+
- FFmpeg
- CUDA-compatible GPU (optional, for faster processing)
- Required Python packages (see `requirements.txt`)

## 🚀 Installation

1. Clone the repository:

```bash
git clone https://github.com/CarlosUlisesOchoa/Whisper-AI-write-video-or-audio-transcription-to-txt.git
```

2. Install the required Python packages:

```bash
pip install -r requirements.txt
```

3. Run the script:

```bash
py main.py --input "D:\files\myfile.mp4" --language en
```

You'll find the transcription in the same folder with the same name as the input file but with a .txt extension.

## 🔑 License

- [GPL-3.0 license](https://github.com/CarlosUlisesOchoa/Whisper-AI-write-video-or-audio-transcription-to-txt/blob/main/LICENSE)
