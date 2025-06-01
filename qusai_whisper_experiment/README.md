# Voice Note Transcription with OpenAI Whisper

This project provides tools for transcribing voice notes in multiple languages (with special support for Arabic and English) using OpenAI's Whisper speech recognition model.

## Features

- Transcribe audio files in Arabic, English, and many other languages
- Automatic language detection
- Generate transcripts in text, JSON, and SRT (subtitle) formats
- Process files individually or in batch
- Option to translate audio to English from any source language
- Parallel processing for batch transcription

## Requirements

- Python 3.8 or higher
- FFmpeg installed on your system
- Virtual environment (recommended)

## Setup Instructions

1. Activate the virtual environment:

```bash
source grad-venv/bin/activate
```

2. Install required Python packages:

```bash
pip install ffmpeg-python torch openai-whisper tqdm
```

3. Ensure FFmpeg is installed on your system:

```bash
# For Ubuntu/Debian:
sudo apt update && sudo apt install ffmpeg

# For macOS:
brew install ffmpeg

# For Windows:
# Install via Chocolatey: choco install ffmpeg
```

## Usage

### Basic Transcription

For basic transcription of a single audio file:

```bash
python transcribe.py path/to/audio_file.mp3
```

Options:

- `--model`: Whisper model to use (`tiny`, `base`, `small`, `medium`, `large`). Default: `medium`
- `--language`: Language code (e.g., `ar` for Arabic, `en` for English). If not specified, Whisper will auto-detect.
- `--output`: Output file path for the transcription

### Advanced Transcription

For advanced transcription with more options and output formats:

```bash
python advanced_transcribe.py path/to/audio_file.mp3 --model medium --language ar
```

Additional options:

- `--output_dir`: Directory to save transcription files. Default: `transcriptions`
- `--task`: Task to perform (`transcribe` or `translate`). Default: `transcribe`
- `--device`: Device to use (`cuda` for GPU, `cpu` for CPU). Default: auto-detect
- `--verbose`: Show verbose output during transcription

### Batch Processing

To process multiple audio files at once:

```bash
python batch_transcribe.py path/to/audio_directory --model medium --language ar
```

Additional options:

- `--pattern`: Comma-separated patterns for audio files. Default: `*.mp3,*.wav,*.m4a,*.ogg,*.flac`
- `--workers`: Number of parallel workers. Default: `1`

## Language Support

Whisper supports a wide range of languages. For Arabic and English specifically:

- Arabic: Use language code `ar`
- English: Use language code `en`

If you don't specify a language, Whisper will automatically detect it.

## Model Selection

Whisper provides different model sizes with varying accuracy and speed:

| Model | Parameters | Required VRAM | Relative Speed | Accuracy |
|-------|------------|---------------|----------------|----------|
| tiny  | 39M        | ~1 GB         | ~32x           | Lowest   |
| base  | 74M        | ~1 GB         | ~16x           | Low      |
| small | 244M       | ~2 GB         | ~6x            | Medium   |
| medium| 769M       | ~5 GB         | ~2x            | High     |
| large | 1550M      | ~10 GB        | 1x             | Highest  |

The `medium` model provides a good balance between accuracy and speed for most use cases.

## Output Formats

The advanced transcription script generates multiple output files:

- `.txt`: Plain text transcription
- `.json`: JSON file with segments and timestamps
- `.srt`: SubRip subtitle format compatible with video players

## Examples

### Transcribe an Arabic voice note:

```bash
python advanced_transcribe.py arabic_voice_note.mp3 --language ar
```

### Translate an Arabic voice note to English:

```bash
python advanced_transcribe.py arabic_voice_note.mp3 --language ar --task translate
```

### Batch process all voice notes in a directory:

```bash
python batch_transcribe.py voice_notes_directory/ --model medium
```

## Troubleshooting

1. **"FFmpeg not found" error**:
   - Ensure FFmpeg is installed and in your system PATH
   - For WSL users, install FFmpeg with `sudo apt install ffmpeg`

2. **Memory issues with large models**:
   - Use a smaller model like `base` or `small`
   - Close other applications to free up memory

3. **Slow processing**:
   - Use a GPU if available
   - Select a smaller model for faster processing
   - For batch processing, increase the number of workers if you have multiple CPUs

## License

This project uses OpenAI Whisper, which is released under the MIT License. 