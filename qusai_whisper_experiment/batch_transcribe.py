#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import time
import glob
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# Import our transcription module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from advanced_transcribe import transcribe_audio_advanced

def setup_args():
    """Set up command-line arguments."""
    parser = argparse.ArgumentParser(description='Batch process audio files for transcription using OpenAI Whisper.')
    parser.add_argument(
        'input_dir', 
        type=str, 
        help='Directory containing audio files to transcribe'
    )
    parser.add_argument(
        '--pattern', 
        type=str, 
        default='*.mp3,*.wav,*.m4a,*.ogg,*.flac',
        help='Comma-separated patterns for audio files to process (default: "*.mp3,*.wav,*.m4a,*.ogg,*.flac")'
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
        '--workers', 
        type=int, 
        default=1, 
        help='Number of parallel workers (default: 1)'
    )
    parser.add_argument(
        '--verbose', 
        action='store_true', 
        help='Show verbose output during transcription'
    )
    return parser.parse_args()

def find_audio_files(input_dir, pattern):
    """
    Find all audio files in the input directory matching the pattern.
    
    Args:
        input_dir (str): Directory to search
        pattern (str): Comma-separated file patterns
        
    Returns:
        list: List of audio file paths
    """
    audio_files = []
    patterns = pattern.split(',')
    
    for p in patterns:
        pattern_path = os.path.join(input_dir, p.strip())
        matching_files = glob.glob(pattern_path)
        audio_files.extend(matching_files)
    
    return sorted(audio_files)

def transcribe_worker(args):
    """
    Worker function for parallel transcription.
    
    Args:
        args (dict): Arguments for transcription
        
    Returns:
        tuple: (file_path, success, message)
    """
    file_path = args['audio_path']
    try:
        result = transcribe_audio_advanced(
            audio_path=file_path,
            model_name=args['model_name'],
            language=args['language'],
            output_dir=args['output_dir'],
            task=args['task'],
            device=args['device'],
            verbose=args['verbose']
        )
        return (file_path, True, "Success")
    except Exception as e:
        return (file_path, False, str(e))

def main():
    """Main function."""
    args = setup_args()
    
    # Get full paths
    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)
    
    # Find audio files
    print(f"Searching for audio files in {input_dir} with pattern {args.pattern}")
    audio_files = find_audio_files(input_dir, args.pattern)
    
    if not audio_files:
        print(f"No audio files found in {input_dir} matching the pattern {args.pattern}")
        sys.exit(1)
    
    print(f"Found {len(audio_files)} audio files to process")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process files
    start_time = time.time()
    results = []
    
    # Use single process if workers=1, otherwise use ProcessPoolExecutor
    if args.workers == 1:
        print("Processing files sequentially...")
        for file_path in audio_files:
            print(f"\nProcessing {file_path}...")
            worker_args = {
                'audio_path': file_path,
                'model_name': args.model,
                'language': args.language,
                'output_dir': output_dir,
                'task': args.task,
                'device': args.device,
                'verbose': args.verbose
            }
            result = transcribe_worker(worker_args)
            results.append(result)
    else:
        print(f"Processing files in parallel with {args.workers} workers...")
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_to_file = {}
            
            for file_path in audio_files:
                worker_args = {
                    'audio_path': file_path,
                    'model_name': args.model,
                    'language': args.language,
                    'output_dir': output_dir,
                    'task': args.task,
                    'device': args.device,
                    'verbose': args.verbose
                }
                future = executor.submit(transcribe_worker, worker_args)
                future_to_file[future] = file_path
            
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    result = future.result()
                    results.append(result)
                    print(f"Completed {file_path}")
                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")
                    results.append((file_path, False, str(e)))
    
    # Summarize results
    total_time = time.time() - start_time
    
    print("\n==== Transcription Summary ====")
    print(f"Total files processed: {len(audio_files)}")
    print(f"Total time: {total_time:.2f} seconds")
    
    successful = [r for r in results if r[1]]
    failed = [r for r in results if not r[1]]
    
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if failed:
        print("\nFailed files:")
        for file_path, _, error in failed:
            print(f"- {file_path}: {error}")
    
    print("\nTranscription complete!")
    print(f"Results saved in: {output_dir}")

if __name__ == "__main__":
    main() 