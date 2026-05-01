"""
Stage 3: LLM Rules Engine

Combines YOLO structured data and Gemini visual analysis,
then applies FIFA Law 12 to make a final VAR foul decision.
"""

import json
from google import genai
from google.genai import types
from .models import (
    YOLOAnalysisResult,
    GeminiVisualAnalysis,
    VARDecision,
)
from .prompts import RULES_ENGINE_SYSTEM_PROMPT, RULES_ENGINE_USER_PROMPT


class RulesEngine:
    """
    The VAR decision-maker. Takes all evidence from Stage 1 (YOLO) and
    Stage 2 (Gemini visual), applies FIFA Law 12, and outputs a structured decision.
    """

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash"):
        """
        Initialize the rules engine.

        Args:
            api_key: Google AI Studio API key.
            model: Gemini model to use for the rules engine.
        """
        self.client = genai.Client(api_key=api_key)
        self.model = model
        print(f"[RULES] Initialized Rules Engine with model: {model}")

    def decide(
        self,
        yolo_analysis: YOLOAnalysisResult,
        gemini_analysis: GeminiVisualAnalysis,
    ) -> VARDecision:
        """
        Make a VAR decision based on all available evidence.

        Args:
            yolo_analysis: Structured player detection and tracking data from YOLO.
            gemini_analysis: Visual scene understanding from Gemini.

        Returns:
            VARDecision with the final call, reasoning, and evidence.
        """
        # Build the user prompt with all evidence
        user_prompt = RULES_ENGINE_USER_PROMPT.format(
            yolo_summary=yolo_analysis.get_summary(),
            scene_description=gemini_analysis.scene_description,
            ball_possession=gemini_analysis.ball_possession,
            challenge_type=gemini_analysis.challenge_type,
            initial_contact_point=gemini_analysis.initial_contact_point,
            contact_body_area=gemini_analysis.contact_body_area,
            challenge_direction=gemini_analysis.challenge_direction,
            force_assessment=gemini_analysis.force_assessment,
            studs_showing=gemini_analysis.studs_showing,
            two_footed=gemini_analysis.two_footed,
            simulation_suspected=gemini_analysis.simulation_suspected,
            ball_playing_distance=gemini_analysis.ball_playing_distance,
            attacking_position=gemini_analysis.attacking_position,
            additional_observations=gemini_analysis.additional_observations,
        )

        print(f"[RULES] Sending evidence to rules engine ({self.model})...")
        print(f"[RULES] Evidence length: {len(user_prompt)} chars")

        # Call the LLM with structured output
        response = self.client.models.generate_content(
            model=self.model,
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part(text=RULES_ENGINE_SYSTEM_PROMPT + "\n\n" + user_prompt)
                    ],
                ),
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=VARDecision,
                temperature=0.1,  # Very low temperature for consistent, precise decisions
            ),
        )

        # Parse the structured response
        decision = VARDecision.model_validate_json(response.text)

        print(f"[RULES] ===========================================")
        print(f"[RULES] VAR DECISION: {'>> FOUL' if decision.is_foul else '>> NO FOUL'}")
        if decision.is_foul:
            print(f"[RULES]   Type: {decision.foul_type}")
            print(f"[RULES]   Severity: {decision.severity}")
            print(f"[RULES]   Card: {decision.card_recommendation}")
        print(f"[RULES]   Confidence: {decision.confidence:.0%}")
        print(f"[RULES] ===========================================")

        return decision
