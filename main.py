"""
Video to Text Conversion Tool
Copyright (C) 2024  CarlosUlisesOchoa @Carlos8aDev
"""

import subprocess
import argparse
import os
import torch
import whisper

def extract_audio_from_video(video_path, output_audio_path):
    """Extracts audio from a video file using FFmpeg."""
    try:
        subprocess.run([
            "ffmpeg", "-i", video_path, "-q:a", "0", "-map", "a", output_audio_path, "-y"
        ], check=True)
        print(f"Audio extracted to {output_audio_path}")
    except subprocess.CalledProcessError:
        print("Failed to extract audio from video.")
        raise

def transcribe_audio(audio_path, output_path=None, language=None):
    """Transcribes audio file to text using Whisper."""
    # Get default output path if not specified
    if output_path is None:
        output_path = audio_path.rsplit('.', 1)[0] + '.txt'

    # Check if CUDA is available
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        print(f"CUDA is available. Using GPU: {gpu_name}")
    else:
        device = "cpu"
        print("CUDA is not available. Using CPU.")

    # Load Whisper model
    model_name = "turbo"
    print(f"Loading Whisper {model_name} model...")
    model = whisper.load_model(model_name, device=device)

    # Transcribe
    print("Starting transcription...")
    result = model.transcribe(audio_path, language=language)

    # Save transcription
    print("Transcription completed. Saving to:", output_path)
    with open(output_path, 'w', encoding='utf-8') as f:
        for segment in result["segments"]:
            start = segment["start"]
            end = segment["end"]
            text = segment["text"]
            line = f"[{start:.2f}s - {end:.2f}s] {text}\n"
            f.write(line)
            print(line, end='')

def main():
    parser = argparse.ArgumentParser(description="Convert video to text through audio extraction and transcription.")
    parser.add_argument("--input", type=str, required=True, help="Path to the video file")
    parser.add_argument("--output", type=str, help="Path to save the text file (default: same as input with .txt extension)")
    parser.add_argument("--language", type=str, default=None, help="Language of the audio")
    parser.add_argument("--keep-audio", action="store_true", help="Keep the intermediate audio file")
    args = parser.parse_args()

    # Check if input file exists
    if not os.path.exists(args.input):
        print("Input file does not exist.")
        return

    # Check if input is video
    file_extension = os.path.splitext(args.input)[1].lower()
    if file_extension not in [".mp4", ".mkv", ".avi"]:
        print("Input file must be a video file (mp4, mkv, or avi).")
        return

    # Set default output path if not specified
    if args.output is None:
        args.output = args.input.rsplit('.', 1)[0] + '.txt'

    # Create temporary audio file path
    temp_audio_path = args.input.rsplit('.', 1)[0] + '_temp.wav'

    try:
        # Extract audio
        extract_audio_from_video(args.input, temp_audio_path)

        # Transcribe audio
        transcribe_audio(temp_audio_path, args.output, args.language)

    finally:
        # Clean up temporary audio file if not keeping it
        if not args.keep_audio and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
            print(f"Removed temporary audio file: {temp_audio_path}")

if __name__ == "__main__":
    main()
