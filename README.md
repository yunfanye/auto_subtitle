# auto_subtitle

Automatic subtitle generation tool that extracts audio from video files and generates subtitles using Google's Gemini AI models.

## Features

- **Audio Extraction**: Automatically extracts audio from video files using ffmpeg
- **AI Transcription**: Uses Gemini 2.5 models (Flash or Pro) for accurate speech-to-text with precise timestamps
- **Multiple Output Formats**:
  - `.srt` subtitle files
  - Videos with embedded subtitles (toggleable)
  - Videos with burnt-in subtitles (permanent)
  - Extracted audio files (`.wav`)
- **Model Selection**: Choose between Gemini Flash 2.5 (faster) or Pro 2.5 (more accurate)
- **Structured Output**: Uses Pydantic models for reliable JSON parsing
- **Environment Variables**: Supports `.env` files for API key management

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   
   Or install manually:
   ```bash
   pip install google-genai pydantic python-dotenv
   ```

2. **Install ffmpeg** (required for audio/video processing):
   - macOS: `brew install ffmpeg`
   - Ubuntu: `sudo apt install ffmpeg`
   - Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html)

3. **Set up Gemini API Key**:
   Create a `.env` file in the project directory:
   ```
   GEMINI_API_KEY=your_api_key_here
   ```

## Usage

### Basic Usage
```bash
# Process default video (test.mp4) with Flash model
python auto_subtitle.py

# Process specific video file
python auto_subtitle.py my_video.mp4

# Use Pro model for better accuracy
python auto_subtitle.py my_video.mp4 --model pro
```

### Command Line Options
```bash
python auto_subtitle.py [video_file] [options]

Arguments:
  video_file              Video file to process (default: test.mp4)

Options:
  --api-key API_KEY       Gemini API key (or set GEMINI_API_KEY env var)
  --output, -o OUTPUT     Output SRT file path (default: video_name.srt)
  --model {flash,pro}     Gemini model: flash (faster) or pro (more accurate)
```

## Output Files

For input video `my_video.mp4`, the tool generates:

- `my_video.srt` - Standard subtitle file
- `my_video.wav` - Extracted audio file
- `my_video_embedded.mp4` - Video with embedded subtitles (can be toggled on/off)
- `my_video_captioned.mp4` - Video with burnt-in subtitles (always visible)

## Model Comparison

| Model | Speed | Accuracy | Cost | Best For |
|-------|-------|----------|------|----------|
| Flash | Fast | Good | Lower | Quick processing, bulk videos |
| Pro | Slower | Better | Higher | High-quality transcription, important content |

## Requirements

- Python 3.8+
- ffmpeg
- Google Gemini API key
- Required Python packages: `google-genai`, `pydantic`, `python-dotenv`
