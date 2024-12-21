import torch
import whisper
import argparse

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Transcribe audio using Whisper model.")
    parser.add_argument("audio_file", type=str, help="Path to the audio file")
    parser.add_argument("--language", type=str, default=None, help="Language of the audio")
    args = parser.parse_args()

    # Step 1: Check if CUDA (GPU) is available and print which device will be used
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        print(f"CUDA is available. Using GPU: {gpu_name}")
    else:
        device = "cpu"
        print("CUDA is not available. Using CPU.")

    # Step 2: Load the Whisper "turbo" model using the appropriate device (GPU or CPU)
    model_name = "turbo"  # Always using the "turbo" version
    model = whisper.load_model(model_name, device=device)

    # Step 3: Transcribe the audio file
    print("Starting transcription...")
    result = model.transcribe(args.audio_file, language=args.language)

    # Step 4: Print each segment of the transcription with timestamps
    print("Transcription completed:")
    for segment in result["segments"]:
        start = segment["start"]
        end = segment["end"]
        text = segment["text"]
        print(f"[{start:.2f}s - {end:.2f}s] {text}")

if __name__ == "__main__":
    main()
