"""
Stage 2: Gemini Video Analysis

Uploads the video to Google's Files API, then uses Gemini 2.5 Pro's native
video understanding to analyze the football incident like an expert referee.
"""

import time
import json
from google import genai
from google.genai import types
from .models import GeminiVisualAnalysis
from .prompts import VIDEO_ANALYSIS_PROMPT


class GeminiVideoAnalyzer:
    """Analyzes football video clips using Gemini's multimodal video understanding."""

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash"):
        """
        Initialize the Gemini video analyzer.

        Args:
            api_key: Google AI Studio API key.
            model: Gemini model to use for video analysis.
        """
        self.client = genai.Client(api_key=api_key)
        self.model = model
        print(f"[GEMINI] Initialized with model: {model}")

    def analyze(self, video_path: str) -> GeminiVisualAnalysis:
        """
        Upload a video and run Gemini visual analysis on it.

        Args:
            video_path: Path to the video file.

        Returns:
            GeminiVisualAnalysis with structured scene understanding.
        """
        # Step 1: Upload the video
        print(f"[GEMINI] Uploading video: {video_path}")
        video_file = self.client.files.upload(file=video_path)
        print(f"[GEMINI] File uploaded: {video_file.name}")

        # Step 2: Wait for processing
        print("[GEMINI] Waiting for video processing...")
        while video_file.state == "PROCESSING":
            time.sleep(5)
            video_file = self.client.files.get(name=video_file.name)
            print(f"[GEMINI]   State: {video_file.state}")

        if video_file.state == "FAILED":
            raise ValueError(f"Gemini video processing failed: {video_file.state}")

        print(f"[GEMINI] Video ready. Running analysis...")

        # Step 3: Generate content with structured output
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=[
                    video_file,
                    VIDEO_ANALYSIS_PROMPT,
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=GeminiVisualAnalysis,
                    temperature=0.2,  # Low temperature for factual analysis
                ),
            )

            # Parse the structured response
            result = GeminiVisualAnalysis.model_validate_json(response.text)
            print(f"[GEMINI] Analysis complete:")
            print(f"  Scene: {result.scene_description[:100]}...")
            print(f"  Challenge: {result.challenge_type}")
            print(f"  Contact: {result.initial_contact_point}")

        finally:
            # Step 4: Clean up — delete the uploaded file
            try:
                self.client.files.delete(name=video_file.name)
                print(f"[GEMINI] Cleaned up uploaded file.")
            except Exception as e:
                print(f"[GEMINI] Warning: Could not delete file: {e}")

        return result
