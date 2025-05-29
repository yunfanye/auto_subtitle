# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an automatic subtitle generation tool that:
1. Accepts video files (defaults to test.mp4)
2. Extracts audio from the video
3. Uses Gemini Flash 2.5 API to generate captions
4. Outputs subtitles in .srt format

## Project Structure

Currently minimal - the main implementation needs to be built. The repository contains:
- `test.mp4` - sample video file for testing
- `README.md` - project description

## Development Notes

- The project is designed to work with Gemini Flash 2.5 for audio-to-text processing
- Default input file is `test.mp4`
- Output format should be .srt subtitle files
- Implementation will likely need audio extraction capabilities and Gemini API integration