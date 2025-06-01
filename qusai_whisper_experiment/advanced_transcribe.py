#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import argparse
import json
import torch
import whisper
from pathlib import Path
from tqdm import tqdm

def setup_args():
    """Set up command-line arguments."""
    parser = argparse.ArgumentParser(description='Advanced audio transcription using OpenAI Whisper.')
    parser.add_argument(
        'audio_path', 
        type=str, 
        help='Path to the audio file to transcribe'
    )
    parser.add_argument(
        '--model', 
        type=str, 
        default='medium', 
        choices=['tiny', 'base', 'small', 'medium', 'large'],
        help='Whisper model to use (default: medium)'
    )
    parser.add_argument(
        '--language', 
        type=str, 
        default=None, 
        help='Language code (ar for Arabic, en for English). If not specified, Whisper will auto-detect.'
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default='transcriptions', 
        help='Directory to save transcription files (default: transcriptions)'
    )
    parser.add_argument(
        '--task', 
        type=str, 
        default='transcribe', 
        choices=['transcribe', 'translate'],
        help='Task to perform. "transcribe" keeps original language, "translate" translates to English (default: transcribe)'
    )
    parser.add_argument(
        '--device', 
        type=str, 
        default=None, 
        help='Device to use for inference (cuda for GPU, cpu for CPU). Defaults to GPU if available.'
    )
    parser.add_argument(
        '--verbose', 
        action='store_true', 
        help='Show verbose output during transcription'
    )
    return parser.parse_args()

def detect_audio_language(audio_path, model):
    """
    Detect the language of an audio file.
    
    Args:
        audio_path (str): Path to the audio file
        model: Loaded Whisper model
        
    Returns:
        str: Detected language code
    """
    print("Detecting audio language...")
    
    # Load audio
    audio = whisper.load_audio(audio_path)
    
    # Ensure audio is an appropriate length for language detection
    audio = whisper.pad_or_trim(audio)
    
    # Convert audio to log-Mel spectrogram
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    
    # Detect language
    _, probs = model.detect_language(mel)
    detected_language = max(probs, key=probs.get)
    
    print(f"Detected language: {detected_language}")
    return detected_language

def transcribe_audio_advanced(
    audio_path, 
    model_name='medium', 
    language=None, 
    output_dir='transcriptions',
    task='transcribe',
    device=None,
    verbose=False
):
    """
    Advanced transcription of an audio file using OpenAI's Whisper.
    
    Args:
        audio_path (str): Path to the audio file
        model_name (str): Name of the Whisper model to use
        language (str): Language code (ar for Arabic, en for English)
        output_dir (str): Directory to save transcription files
        task (str): Task to perform ('transcribe' or 'translate')
        device (str): Device to use ('cuda' or 'cpu')
        verbose (bool): Whether to show verbose output
        
    Returns:
        dict: Transcription results including text and segments
    """
    # Check if the audio file exists
    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found at {audio_path}")
        sys.exit(1)
    
    # Setup device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Using device: {device}")
    
    # Load the specified model
    print(f"Loading Whisper model: {model_name}")
    start_time = time.time()
    try:
        model = whisper.load_model(model_name, device=device)
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        sys.exit(1)
        
    load_time = time.time() - start_time
    print(f"Model loaded in {load_time:.2f} seconds")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get base filename without extension
    input_filename = os.path.basename(audio_path)
    base_filename = os.path.splitext(input_filename)[0]
    
    # Detect language if not specified
    if language is None:
        language = detect_audio_language(audio_path, model)
    
    # Set up transcription options
    options = {
        "task": task,
        "verbose": verbose,
    }
    
    if language:
        options["language"] = language
    
    # Transcribe the audio
    print(f"Transcribing audio: {audio_path}")
    print(f"Task: {task}")
    start_time = time.time()
    try:
        result = model.transcribe(audio_path, **options)
    except Exception as e:
        print(f"Error during transcription: {str(e)}")
        sys.exit(1)
    
    transcription_time = time.time() - start_time
    print(f"Transcription completed in {transcription_time:.2f} seconds")
    
    # Create output files
    output_files = {}
    
    # Save full text transcription
    text_file = os.path.join(output_dir, f"{base_filename}_{language}.txt")
    with open(text_file, 'w', encoding='utf-8') as f:
        f.write(result["text"])
    output_files["text"] = text_file
    
    # Save segments with timestamps as JSON
    segments_file = os.path.join(output_dir, f"{base_filename}_{language}_segments.json")
    with open(segments_file, 'w', encoding='utf-8') as f:
        json.dump(result["segments"], f, ensure_ascii=False, indent=2)
    output_files["segments"] = segments_file
    
    # Save segments with timestamps as SRT format for subtitles
    srt_file = os.path.join(output_dir, f"{base_filename}_{language}.srt")
    with open(srt_file, 'w', encoding='utf-8') as f:
        for i, segment in enumerate(result["segments"], start=1):
            # Format timestamps as SRT format (HH:MM:SS,mmm)
            start_time_str = format_timestamp(segment["start"])
            end_time_str = format_timestamp(segment["end"])
            
            f.write(f"{i}\n")
            f.write(f"{start_time_str} --> {end_time_str}\n")
            f.write(f"{segment['text'].strip()}\n\n")
    output_files["srt"] = srt_file
    
    print("\nTranscription files saved:")
    for file_type, file_path in output_files.items():
        print(f"- {file_type}: {file_path}")
    
    return result

def format_timestamp(seconds):
    """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)"""
    milliseconds = int((seconds % 1) * 1000)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

def display_segments(segments, max_segments=10):
    """
    Display a sample of transcription segments with timestamps.
    
    Args:
        segments (list): List of transcription segments
        max_segments (int): Maximum number of segments to display
    """
    print("\nSample segments with timestamps:")
    segments_to_show = segments[:max_segments]
    
    for i, segment in enumerate(segments_to_show, start=1):
        start_time = format_timestamp(segment["start"])
        end_time = format_timestamp(segment["end"])
        text = segment["text"].strip()
        
        print(f"{i}. [{start_time} --> {end_time}] {text}")
    
    if len(segments) > max_segments:
        print(f"... and {len(segments) - max_segments} more segments")

def main():
    """Main function."""
    args = setup_args()
    
    # Get the full path to the audio file
    audio_path = os.path.abspath(args.audio_path)
    
    # Get full path to output directory
    output_dir = os.path.abspath(args.output_dir)
    
    # Transcribe the audio file
    result = transcribe_audio_advanced(
        audio_path=audio_path,
        model_name=args.model,
        language=args.language,
        output_dir=output_dir,
        task=args.task,
        device=args.device,
        verbose=args.verbose
    )
    
    # Display sample segments
    display_segments(result["segments"])
    
    # Print a preview of the transcription
    print("\nTranscription Preview:")
    preview_length = min(300, len(result["text"]))
    print(f"{result['text'][:preview_length]}...")
    
    if len(result["text"]) > preview_length:
        print("...")

if __name__ == "__main__":
    main() 