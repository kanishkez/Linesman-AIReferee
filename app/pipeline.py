"""
Pipeline Orchestrator

Coordinates the 3-stage analysis pipeline:
  Stage 1: YOLOv8 Pose Detection & Tracking
  Stage 2: Gemini Video Understanding
  Stage 3: LLM Rules Engine (FIFA Law 12)
"""

import os
import time
import asyncio
from .models import AnalysisResult, AnalysisStatus
from .yolo_analyzer import YOLOAnalyzer
from .gemini_video_analyzer import GeminiVideoAnalyzer
from .rules_engine import RulesEngine


class VARPipeline:
    """Orchestrates the full VAR analysis pipeline."""

    def __init__(self, api_key: str, output_base_dir: str = "outputs"):
        """
        Initialize all pipeline stages.

        Args:
            api_key: Google AI Studio API key for Gemini.
            output_base_dir: Base directory for output files.
        """
        self.output_base_dir = output_base_dir
        os.makedirs(output_base_dir, exist_ok=True)

        # Initialize stages
        print("=" * 60)
        print("  FOOTBALL AI VAR -- Initializing Pipeline")
        print("=" * 60)

        self.yolo = YOLOAnalyzer(model_name="yolov8x-pose.pt")
        self.gemini = GeminiVideoAnalyzer(api_key=api_key)
        self.rules = RulesEngine(api_key=api_key)

        print("=" * 60)
        print("  Pipeline ready. Awaiting video input.")
        print("=" * 60)

    def run(self, video_path: str, job_id: str, result: AnalysisResult) -> AnalysisResult:
        """
        Run the full 3-stage VAR analysis pipeline.

        Args:
            video_path: Path to the uploaded video file.
            job_id: Unique identifier for this analysis job.
            result: The AnalysisResult object to update in-place (for status tracking).

        Returns:
            Updated AnalysisResult with all analysis data.
        """
        start_time = time.time()
        output_dir = os.path.join(self.output_base_dir, job_id)
        os.makedirs(output_dir, exist_ok=True)

        try:
            # ─── Stage 1: YOLO Pose Detection & Tracking ────────────────
            print("\n" + "=" * 60)
            print("  STAGE 1: YOLOv8 Player Detection & Pose Tracking")
            print("=" * 60)
            result.status = AnalysisStatus.YOLO_PROCESSING

            yolo_result = self.yolo.analyze(
                video_path=video_path,
                output_dir=output_dir,
                sample_rate=2,  # Process every 2nd frame for speed
            )
            result.yolo_analysis = yolo_result
            result.annotated_video_path = yolo_result.annotated_video_path

            print(f"[PIPELINE] Stage 1 complete: {yolo_result.max_players_detected} max players, "
                  f"{len(yolo_result.key_contact_frames)} contact frames")

            # ─── Stage 2: Gemini Video Analysis ─────────────────────────
            print("\n" + "=" * 60)
            print("  STAGE 2: Gemini Visual Scene Analysis")
            print("=" * 60)
            result.status = AnalysisStatus.GEMINI_ANALYZING

            gemini_result = self.gemini.analyze(video_path=video_path)
            result.gemini_analysis = gemini_result

            print(f"[PIPELINE] Stage 2 complete: {gemini_result.challenge_type}, "
                  f"contact={gemini_result.initial_contact_point}")

            # ─── Stage 3: Rules Engine Decision ─────────────────────────
            print("\n" + "=" * 60)
            print("  STAGE 3: LLM Rules Engine -- FIFA Law 12")
            print("=" * 60)
            result.status = AnalysisStatus.RULES_ENGINE

            var_decision = self.rules.decide(
                yolo_analysis=yolo_result,
                gemini_analysis=gemini_result,
            )
            result.var_decision = var_decision

            # ─── Done ───────────────────────────────────────────────────
            elapsed = time.time() - start_time
            result.status = AnalysisStatus.COMPLETED
            result.processing_time_sec = round(elapsed, 2)

            print("\n" + "=" * 60)
            print(f"  PIPELINE COMPLETE -- {elapsed:.1f}s total")
            print(f"  Decision: {'FOUL' if var_decision.is_foul else 'NO FOUL'}")
            if var_decision.is_foul:
                print(f"  Type: {var_decision.foul_type} | Severity: {var_decision.severity}")
                print(f"  Card: {var_decision.card_recommendation}")
            print(f"  Confidence: {var_decision.confidence:.0%}")
            print("=" * 60)

        except Exception as e:
            result.status = AnalysisStatus.ERROR
            result.error_message = str(e)
            result.processing_time_sec = round(time.time() - start_time, 2)
            print(f"[PIPELINE] ERROR: {e}")
            import traceback
            traceback.print_exc()

        return result
