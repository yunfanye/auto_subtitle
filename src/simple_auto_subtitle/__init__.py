"""
Simple Auto Subtitle - Automatic subtitle generation using Gemini AI

A Python tool that automatically generates subtitles for videos using Google's Gemini AI.
Extracts audio, transcribes speech with precise timestamps, and outputs SRT files plus 
videos with embedded or burnt-in captions.
"""

__version__ = "0.1.0"
__author__ = "yunfanye"
__email__ = "your.email@example.com"
__description__ = "Automatic subtitle generation using Gemini AI"

from .main import main, GeminiTranscriber, AudioExtractor, SRTGenerator, VideoSubtitleEmbedder

__all__ = [
    "main",
    "GeminiTranscriber", 
    "AudioExtractor",
    "SRTGenerator", 
    "VideoSubtitleEmbedder"
]