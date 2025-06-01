#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import re
import json
import whisper
from pathlib import Path

def setup_args():
    """Set up command-line arguments."""
    parser = argparse.ArgumentParser(description='Specialized Arabic audio transcription using OpenAI Whisper.')
    parser.add_argument(
        'audio_path', 
        type=str, 
        help='Path to the Arabic audio file to transcribe'
    )
    parser.add_argument(
        '--model', 
        type=str, 
        default='medium', 
        choices=['tiny', 'base', 'small', 'medium', 'large'],
        help='Whisper model to use (default: medium)'
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default='transcriptions', 
        help='Directory to save transcription files (default: transcriptions)'
    )
    parser.add_argument(
        '--translate', 
        action='store_true', 
        help='Translate Arabic audio to English'
    )
    parser.add_argument(
        '--verbose', 
        action='store_true', 
        help='Show verbose output during transcription'
    )
    return parser.parse_args()

def normalize_arabic_text(text):
    """
    Normalize Arabic text by:
    1. Removing diacritics
    2. Normalizing hamzas
    3. Removing extra spaces
    4. Fixing common transcription errors
    
    Args:
        text (str): Arabic text to normalize
        
    Returns:
        str: Normalized Arabic text
    """
    if not text:
        return text
    
    # Remove diacritics (tashkeel)
    text = re.sub(r'[\u064B-\u065F\u0670]', '', text)
    
    # Normalize alef with hamza forms to plain alef
    text = re.sub(r'[إأآا]', 'ا', text)
    
    # Normalize different hamza forms
    text = re.sub(r'[ؤئ]', 'ء', text)
    
    # Remove extra spaces and punctuation
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    # Common error corrections in Arabic transcription
    replacements = {
        # Fix common Whisper transcription errors for Arabic
        'اه ': 'اة ',  # ta marbuta fix
        ' ه ': ' ة ',  # ta marbuta fix
        'هذه': 'هذا',  # common phrase correction
        'انا': 'أنا',   # add hamza back for common pronouns
        'انت': 'أنت',   # add hamza back for common pronouns
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    return text

def post_process_arabic_segments(segments):
    """
    Post-process Arabic transcription segments.
    
    Args:
        segments (list): List of transcription segments
        
    Returns:
        list: Post-processed segments
    """
    processed_segments = []
    
    for segment in segments:
        # Create a copy of the segment to modify
        processed_segment = segment.copy()
        
        # Normalize the text
        processed_segment['text'] = normalize_arabic_text(segment['text'])
        
        processed_segments.append(processed_segment)
    
    return processed_segments

def transcribe_arabic_audio(
    audio_path, 
    model_name='medium', 
    output_dir='transcriptions',
    translate=False,
    verbose=False
):
    """
    Transcribe Arabic audio using OpenAI Whisper with specialized post-processing.
    
    Args:
        audio_path (str): Path to the audio file
        model_name (str): Name of the Whisper model to use
        output_dir (str): Directory to save transcription files
        translate (bool): Whether to translate Arabic to English
        verbose (bool): Whether to show verbose output
        
    Returns:
        dict: Transcription results including text and segments
    """
    # Check if the audio file exists
    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found at {audio_path}")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the specified model
    print(f"Loading Whisper model: {model_name}")
    try:
        model = whisper.load_model(model_name)
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        sys.exit(1)
    
    # Set up transcription options
    options = {
        "language": "ar",  # Explicitly set to Arabic
        "task": "translate" if translate else "transcribe",
        "verbose": verbose,
    }
    
    # Transcribe the audio
    print(f"Transcribing Arabic audio: {audio_path}")
    try:
        result = model.transcribe(audio_path, **options)
    except Exception as e:
        print(f"Error during transcription: {str(e)}")
        sys.exit(1)
    
    # Get base filename without extension
    input_filename = os.path.basename(audio_path)
    base_filename = os.path.splitext(input_filename)[0]
    
    # Create language suffix
    lang_suffix = "en" if translate else "ar"
    
    # Post-process the segments if not translating
    if not translate:
        result["segments"] = post_process_arabic_segments(result["segments"])
        
        # Create a normalized full text from the processed segments
        full_text = " ".join([segment["text"] for segment in result["segments"]])
        result["text"] = full_text
    
    # Save full text transcription
    text_file = os.path.join(output_dir, f"{base_filename}_{lang_suffix}.txt")
    with open(text_file, 'w', encoding='utf-8') as f:
        f.write(result["text"])
    
    # Save segments with timestamps as JSON
    segments_file = os.path.join(output_dir, f"{base_filename}_{lang_suffix}_segments.json")
    with open(segments_file, 'w', encoding='utf-8') as f:
        json.dump(result["segments"], f, ensure_ascii=False, indent=2)
    
    print(f"\nTranscription saved to: {text_file}")
    print(f"Segments saved to: {segments_file}")
    
    return result

def display_results(result, translate=False, max_preview=200):
    """
    Display transcription results.
    
    Args:
        result (dict): Transcription results
        translate (bool): Whether this was a translation task
        max_preview (int): Maximum length for text preview
    """
    if translate:
        print("\nArabic to English Translation Preview:")
    else:
        print("\nArabic Transcription Preview:")
    
    # For Arabic text, ensure proper display with RTL markers if in a terminal
    if not translate:
        # Add RTL mark for proper display in terminals
        text_preview = "\u202B" + result["text"][:max_preview] + "\u202C"
    else:
        text_preview = result["text"][:max_preview]
    
    print(f"{text_preview}...")
    
    # Display some segments
    print("\nSample segments:")
    max_segments = 5
    for i, segment in enumerate(result["segments"][:max_segments], start=1):
        start_time = format_timestamp(segment["start"])
        end_time = format_timestamp(segment["end"])
        
        if not translate:
            # Add RTL mark for proper display in terminals
            segment_text = "\u202B" + segment["text"] + "\u202C"
        else:
            segment_text = segment["text"]
        
        print(f"{i}. [{start_time} --> {end_time}] {segment_text}")
    
    if len(result["segments"]) > max_segments:
        print(f"... and {len(result['segments']) - max_segments} more segments")

def format_timestamp(seconds):
    """Convert seconds to timestamp format (MM:SS.mmm)"""
    minutes = int(seconds // 60)
    seconds = seconds % 60
    return f"{minutes:02d}:{seconds:06.3f}"

def main():
    """Main function."""
    args = setup_args()
    
    # Get the full path to the audio file
    audio_path = os.path.abspath(args.audio_path)
    
    # Get full path to output directory
    output_dir = os.path.abspath(args.output_dir)
    
    # Transcribe the audio file
    result = transcribe_arabic_audio(
        audio_path=audio_path,
        model_name=args.model,
        output_dir=output_dir,
        translate=args.translate,
        verbose=args.verbose
    )
    
    # Display results
    display_results(
        result=result,
        translate=args.translate
    )

if __name__ == "__main__":
    main() 