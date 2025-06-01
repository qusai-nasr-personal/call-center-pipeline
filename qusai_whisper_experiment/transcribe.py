#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import whisper
from pathlib import Path

def setup_args():
    """Set up command-line arguments."""
    parser = argparse.ArgumentParser(description='Transcribe audio files using OpenAI Whisper.')
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
        '--output', 
        type=str, 
        default=None, 
        help='Output file to save transcription (default: input_file_name.txt)'
    )
    return parser.parse_args()

def transcribe_audio(audio_path, model_name='medium', language=None, output_path=None):
    """
    Transcribe an audio file using OpenAI's Whisper.
    
    Args:
        audio_path (str): Path to the audio file
        model_name (str): Name of the Whisper model to use
        language (str): Language code (ar for Arabic, en for English)
        output_path (str): Path to save the transcription
        
    Returns:
        str: Transcribed text
    """
    # Check if the audio file exists
    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found at {audio_path}")
        sys.exit(1)
    
    # Load the specified model
    print(f"Loading Whisper model: {model_name}")
    try:
        model = whisper.load_model(model_name)
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        sys.exit(1)
    
    # Set up transcription options
    options = {}
    if language:
        options["language"] = language
    
    # Transcribe the audio
    print(f"Transcribing audio: {audio_path}")
    try:
        result = model.transcribe(audio_path, **options)
    except Exception as e:
        print(f"Error during transcription: {str(e)}")
        sys.exit(1)
    
    # Get the transcribed text
    transcription = result["text"]
    
    # Save to output file if specified
    if output_path:
        output_file = output_path
    else:
        # Default output file is the input file name with .txt extension
        input_path = Path(audio_path)
        output_file = str(input_path.with_suffix('.txt'))
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(transcription)
    
    print(f"Transcription saved to: {output_file}")
    
    # Return the transcribed text
    return transcription

def main():
    """Main function."""
    args = setup_args()
    
    # Get the full path to the audio file
    audio_path = os.path.abspath(args.audio_path)
    
    # Process output path
    output_path = args.output
    if output_path:
        output_path = os.path.abspath(output_path)
    
    # Transcribe the audio file
    transcription = transcribe_audio(
        audio_path=audio_path,
        model_name=args.model,
        language=args.language,
        output_path=output_path
    )
    
    # Print a preview of the transcription
    print("\nTranscription Preview:")
    preview_length = min(200, len(transcription))
    print(f"{transcription[:preview_length]}...")

if __name__ == "__main__":
    main() 