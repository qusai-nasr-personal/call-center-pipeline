#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Example script demonstrating how to use the transcription tools programmatically.
"""

import os
import sys
from pathlib import Path

# Import our transcription modules
from transcribe import transcribe_audio
from advanced_transcribe import transcribe_audio_advanced
from arabic_transcribe import transcribe_arabic_audio

def main():
    """Demonstrate the usage of transcription tools."""
    # Ensure we have an audio file to work with
    if len(sys.argv) < 2:
        print("Usage: python example.py <path_to_audio_file>")
        print("Please provide a path to an audio file.")
        sys.exit(1)
    
    audio_path = os.path.abspath(sys.argv[1])
    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found at {audio_path}")
        sys.exit(1)
    
    # Create output directory
    output_dir = "example_output"
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 50)
    print("OpenAI Whisper Transcription Examples")
    print("=" * 50)
    
    # Example 1: Basic transcription
    print("\n1. Basic Transcription Example")
    print("-" * 30)
    basic_result = transcribe_audio(
        audio_path=audio_path,
        model_name="base",  # Using a smaller model for faster execution
        output_path=os.path.join(output_dir, "basic_transcription.txt")
    )
    print(f"Basic transcription preview: {basic_result[:100]}...")
    
    # Example 2: Advanced transcription with automatic language detection
    print("\n2. Advanced Transcription Example")
    print("-" * 30)
    advanced_result = transcribe_audio_advanced(
        audio_path=audio_path,
        model_name="base",  # Using a smaller model for faster execution
        output_dir=output_dir,
        verbose=True
    )
    
    # Check if the detected language is Arabic
    is_arabic = False
    for segment in advanced_result["segments"]:
        if segment.get("language") == "ar":
            is_arabic = True
            break
    
    # Example 3: Arabic-specific transcription (if applicable)
    if is_arabic:
        print("\n3. Arabic-Specific Transcription Example")
        print("-" * 30)
        arabic_result = transcribe_arabic_audio(
            audio_path=audio_path,
            model_name="base",  # Using a smaller model for faster execution
            output_dir=output_dir,
            verbose=True
        )
        
        # Example 4: Arabic to English translation
        print("\n4. Arabic to English Translation Example")
        print("-" * 30)
        translation_result = transcribe_arabic_audio(
            audio_path=audio_path,
            model_name="base",  # Using a smaller model for faster execution
            output_dir=output_dir,
            translate=True,
            verbose=True
        )
    
    print("\n" + "=" * 50)
    print("Examples completed! Output files saved to:", output_dir)
    print("=" * 50)
    
    # List all generated files
    print("\nGenerated files:")
    for file_path in Path(output_dir).glob("*"):
        print(f"- {file_path}")

if __name__ == "__main__":
    main() 