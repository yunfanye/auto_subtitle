#!/usr/bin/env python3
"""
Auto Subtitle Generator

Extracts audio from video files and generates SRT subtitles using Gemini Flash 2.5 API.
"""

import os
import sys
import argparse
from typing import List, Tuple, Optional
import subprocess
import json
from datetime import timedelta
from google import genai
from google.genai import types
from pydantic import BaseModel

from dotenv import load_dotenv
load_dotenv()


class TranscriptSegment(BaseModel):
    start_time: float  # Time in seconds (e.g., 1.5 for 1.5 seconds)
    end_time: float    # Time in seconds (e.g., 5.2 for 5.2 seconds)
    text: str          # Transcript text for this segment


class AudioExtractor:
    """Handles audio extraction from video files using ffmpeg."""
    
    @staticmethod
    def extract_audio(video_path: str, output_path: str) -> bool:
        """
        Extract audio from video file to WAV format.
        
        Args:
            video_path: Path to input video file
            output_path: Path for output audio file
            
        Returns:
            True if extraction successful, False otherwise
        """
        try:
            cmd = [
                'ffmpeg', '-i', video_path,
                '-ac', '1',  # mono channel
                '-ar', '16000',  # 16kHz sample rate
                '-y',  # overwrite output file
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            return result.returncode == 0
            
        except FileNotFoundError:
            print("Error: ffmpeg not found. Please install ffmpeg.")
            return False
        except Exception as e:
            print(f"Error extracting audio: {e}")
            return False


class GeminiTranscriber:
    """Handles speech-to-text using Gemini API."""
    
    def __init__(self, api_key: str, model: str = 'gemini-2.5-flash-preview-05-20'):
        """
        Initialize with Gemini API key and model.
        
        Args:
            api_key: Gemini API key
            model: Model to use for transcription
        """
        self.api_key = api_key
        self.model = model
        self.client = genai.Client(api_key=api_key)
    
    def transcribe_audio(self, audio_path: str) -> List[Tuple[float, float, str]]:
        """
        Transcribe audio file to text with timestamps.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            List of tuples (start_time, end_time, text)
        """
        try:
            # Read audio file
            with open(audio_path, 'rb') as f:
                audio_bytes = f.read()
            
            # Generate content using new API with structured output
            response = self.client.models.generate_content(
                model=self.model,
                contents=[
                    'Please transcribe this audio file and provide precise timestamps for each segment. Return start_time and end_time as floating point numbers representing seconds (e.g., 1.5 for 1.5 seconds, 10.25 for 10.25 seconds).',
                    types.Part.from_bytes(
                        data=audio_bytes,
                        mime_type='audio/wav',
                    )
                ],
                config={
                    "response_mime_type": "application/json",
                    "response_schema": list[TranscriptSegment],
                }
            )
            
            # Parse structured response
            segments = self._parse_structured_response(response)
            return segments
            
        except Exception as e:
            print(f"Error transcribing audio: {e}")
            return []
    
    def _parse_structured_response(self, response) -> List[Tuple[float, float, str]]:
        """Parse structured Gemini API response."""
        segments = []
        
        try:
            # Use the parsed structured response
            transcript_segments: list[TranscriptSegment] = response.parsed
            
            for segment in transcript_segments:
                if segment.text.strip():
                    segments.append((segment.start_time, segment.end_time, segment.text.strip()))
                    
        except Exception as e:
            print(f"Error parsing structured response: {e}")
            # Fallback to parsing text response
            try:
                segments = self._parse_fallback_response(response.text)
            except Exception as fallback_e:
                print(f"Fallback parsing also failed: {fallback_e}")
                
        return segments
    
    def _parse_fallback_response(self, text: str) -> List[Tuple[float, float, str]]:
        """Fallback parser for when structured parsing fails."""
        segments = []
        
        try:
            # Extract JSON from the response text
            json_start = text.find('[')
            json_end = text.rfind(']') + 1
            
            if json_start != -1 and json_end > json_start:
                json_text = text[json_start:json_end]
                
                try:
                    data = json.loads(json_text)
                    
                    # Handle if data is a list of segments
                    if isinstance(data, list):
                        for segment in data:
                            if isinstance(segment, dict):
                                start_time = float(segment.get('start_time', 0))
                                end_time = float(segment.get('end_time', 0))
                                text_content = segment.get('text', '').strip()
                                if text_content:
                                    segments.append((start_time, end_time, text_content))
                                
                except json.JSONDecodeError:
                    # Final fallback to estimated timing
                    segments = self._create_estimated_segments(text)
            else:
                # No JSON found, use estimated timing
                segments = self._create_estimated_segments(text)
                        
        except Exception as e:
            print(f"Error in fallback parsing: {e}")
            
        return segments
    
    def _create_estimated_segments(self, text: str) -> List[Tuple[float, float, str]]:
        """Create estimated segments when JSON parsing fails."""
        segments = []
        
        # Clean up the text by removing JSON artifacts
        lines = text.split('\n')
        clean_text = []
        
        for line in lines:
            line = line.strip()
            # Skip lines that look like JSON structure
            if (not line or line.startswith('{') or line.startswith('}') or 
                line.startswith('[') or line.startswith(']') or
                line.startswith('"start_time"') or line.startswith('"end_time"') or
                line.startswith('"text"') or line == ',' or line == '```json' or line == '```'):
                continue
            
            # Remove quotes from text content
            line = line.strip('"').strip(',')
            if line:
                clean_text.append(line)
        
        full_text = ' '.join(clean_text)
        words = full_text.split()
        
        if not words:
            return segments
            
        words_per_second = 2.5  # Average speaking rate
        segment_duration = 4  # 4-second segments for better readability
        words_per_segment = int(segment_duration * words_per_second)
        
        current_time = 0
        
        for i in range(0, len(words), words_per_segment):
            segment_words = words[i:i + words_per_segment]
            segment_text = ' '.join(segment_words)
            
            if segment_text.strip():
                start_time = current_time
                end_time = current_time + segment_duration
                segments.append((start_time, end_time, segment_text))
                current_time = end_time
        
        return segments


class SRTGenerator:
    """Generates SRT subtitle files from transcript segments."""
    
    @staticmethod
    def format_timestamp(seconds: float) -> str:
        """
        Format seconds to SRT timestamp format (HH:MM:SS,mmm).
        
        Args:
            seconds: Time in seconds
            
        Returns:
            Formatted timestamp string
        """
        td = timedelta(seconds=seconds)
        hours = int(td.total_seconds() // 3600)
        minutes = int((td.total_seconds() % 3600) // 60)
        secs = int(td.total_seconds() % 60)
        millisecs = int((td.total_seconds() % 1) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"
    
    @staticmethod
    def generate_srt(segments: List[Tuple[float, float, str]], output_path: str) -> bool:
        """
        Generate SRT file from transcript segments.
        
        Args:
            segments: List of (start_time, end_time, text) tuples
            output_path: Path for output SRT file
            
        Returns:
            True if generation successful, False otherwise
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for i, (start_time, end_time, text) in enumerate(segments, 1):
                    start_ts = SRTGenerator.format_timestamp(start_time)
                    end_ts = SRTGenerator.format_timestamp(end_time)
                    
                    f.write(f"{i}\n")
                    f.write(f"{start_ts} --> {end_ts}\n")
                    f.write(f"{text}\n\n")
            
            return True
            
        except Exception as e:
            print(f"Error generating SRT file: {e}")
            return False


class VideoSubtitleEmbedder:
    """Embeds SRT subtitles into video files using ffmpeg."""
    
    @staticmethod
    def embed_subtitles(video_path: str, srt_path: str, output_path: str) -> bool:
        """
        Embed SRT subtitles into video file.
        
        Args:
            video_path: Path to input video file
            srt_path: Path to SRT subtitle file
            output_path: Path for output video with embedded subtitles
            
        Returns:
            True if embedding successful, False otherwise
        """
        try:
            cmd = [
                'ffmpeg',
                '-i', video_path,          # Input video
                '-i', srt_path,            # Input SRT file
                '-c', 'copy',              # Copy video/audio streams without re-encoding
                '-c:s', 'mov_text',        # Subtitle codec (for MP4)
                '-y',                      # Overwrite output file
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            return result.returncode == 0
            
        except FileNotFoundError:
            print("Error: ffmpeg not found. Please install ffmpeg.")
            return False
        except Exception as e:
            print(f"Error embedding subtitles: {e}")
            return False
    
    @staticmethod
    def burn_subtitles(video_path: str, srt_path: str, output_path: str) -> bool:
        """
        Burn SRT subtitles into video (hard-coded subtitles).
        
        Args:
            video_path: Path to input video file
            srt_path: Path to SRT subtitle file
            output_path: Path for output video with burned subtitles
            
        Returns:
            True if burning successful, False otherwise
        """
        try:
            # Use absolute path for SRT file and escape special characters
            abs_srt_path = os.path.abspath(srt_path).replace('\\', '\\\\').replace(':', '\\:')
            
            cmd = [
                'ffmpeg',
                '-i', video_path,
                '-vf', f"subtitles='{abs_srt_path}'",  # Video filter to burn subtitles
                '-y',                                  # Overwrite output file
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            return result.returncode == 0
            
        except FileNotFoundError:
            print("Error: ffmpeg not found. Please install ffmpeg.")
            return False
        except Exception as e:
            print(f"Error burning subtitles: {e}")
            return False


def main():
    """Main function to orchestrate the subtitle generation process."""
    parser = argparse.ArgumentParser(description='Generate subtitles from video files using Gemini Flash 2.5')
    parser.add_argument('video_file', nargs='?', default='test.mp4', 
                       help='Video file to process (default: test.mp4)')
    parser.add_argument('--api-key', required=False,
                       help='Gemini API key (or set GEMINI_API_KEY env var)')
    parser.add_argument('--output', '-o', 
                       help='Output SRT file path (default: video_name.srt)')
    parser.add_argument('--model', choices=['flash', 'pro'], default='flash',
                       help='Gemini model to use: flash (2.5-flash) or pro (2.5-pro) - default: flash')
    
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key or os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("Error: Gemini API key required. Use --api-key or set GEMINI_API_KEY environment variable.")
        sys.exit(1)
    
    # Determine model name
    if args.model == 'pro':
        model_name = 'gemini-2.5-pro-preview-05-06'
    else:
        model_name = 'gemini-2.5-flash-preview-05-20'
    
    # Check if video file exists
    if not os.path.exists(args.video_file):
        print(f"Error: Video file '{args.video_file}' not found.")
        sys.exit(1)
    
    # Determine output paths
    if args.output:
        srt_output_path = args.output
    else:
        base_name = os.path.splitext(args.video_file)[0]
        srt_output_path = f"{base_name}.srt"
    
    # Video and audio output paths
    base_name = os.path.splitext(args.video_file)[0]
    extension = os.path.splitext(args.video_file)[1]
    
    embedded_video_path = f"{base_name}_embedded{extension}"
    burnt_video_path = f"{base_name}_captioned{extension}"
    audio_output_path = f"{base_name}.wav"
    
    print(f"Processing video: {args.video_file}")
    print(f"Output SRT: {srt_output_path}")
    print(f"Output audio: {audio_output_path}")
    print(f"Output embedded video: {embedded_video_path}")
    print(f"Output burnt video: {burnt_video_path}")
    
    try:
        # Step 1: Extract audio
        print("Extracting audio from video...")
        extractor = AudioExtractor()
        if not extractor.extract_audio(args.video_file, audio_output_path):
            print("Failed to extract audio from video.")
            sys.exit(1)
        
        # Step 2: Transcribe audio
        print(f"Transcribing audio with Gemini {args.model.title()}...")
        transcriber = GeminiTranscriber(api_key, model_name)
        segments = transcriber.transcribe_audio(audio_output_path)
        
        if not segments:
            print("Failed to transcribe audio.")
            sys.exit(1)
        
        print(f"Generated {len(segments)} subtitle segments.")
        
        # Step 3: Generate SRT file
        print("Generating SRT file...")
        generator = SRTGenerator()
        if not generator.generate_srt(segments, srt_output_path):
            print("Failed to generate SRT file.")
            sys.exit(1)
        
        print(f"Successfully generated subtitles: {srt_output_path}")
        
        # Step 4: Create both embedded and burnt videos by default
        embedder = VideoSubtitleEmbedder()
        
        # Create embedded video (toggleable subtitles)
        print("Creating embedded subtitles video...")
        embedded_success = embedder.embed_subtitles(args.video_file, srt_output_path, embedded_video_path)
        
        if embedded_success:
            print(f"Successfully embedded subtitles into video: {embedded_video_path}")
        else:
            print("Failed to embed subtitles into video.")
            sys.exit(1)
        
        # Create burnt video (permanent subtitles)
        print("Creating burnt subtitles video...")
        burnt_success = embedder.burn_subtitles(args.video_file, srt_output_path, burnt_video_path)
        
        if burnt_success:
            print(f"Successfully burned subtitles into video: {burnt_video_path}")
        else:
            print("Failed to burn subtitles into video.")
            sys.exit(1)
        
        print(f"Successfully saved extracted audio: {audio_output_path}")


if __name__ == '__main__':
    main()