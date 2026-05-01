"""
Pydantic data models for the Football AI VAR system.
Defines structured types for YOLO output, Gemini analysis, and final VAR decisions.
"""

from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


# ─── Enums ────────────────────────────────────────────────────────────────────

class FoulType(str, Enum):
    TRIPPING = "tripping"
    KICKING = "kicking"
    PUSHING = "pushing"
    CHARGING = "charging"
    STRIKING = "striking"
    HOLDING = "holding"
    HANDBALL = "handball"
    SLIDING_TACKLE = "sliding_tackle"
    DANGEROUS_PLAY = "dangerous_play"
    OBSTRUCTION = "obstruction"
    SIMULATION = "simulation"
    OTHER = "other"
    NONE = "none"


class Severity(str, Enum):
    CARELESS = "careless"
    RECKLESS = "reckless"
    EXCESSIVE_FORCE = "excessive_force"
    NONE = "none"


class CardRecommendation(str, Enum):
    NONE = "none"
    YELLOW = "yellow"
    RED = "red"


class ChallengeDirection(str, Enum):
    FROM_BEHIND = "from_behind"
    FROM_SIDE = "from_side"
    HEAD_ON = "head_on"
    AERIAL = "aerial"
    UNKNOWN = "unknown"


# ─── YOLOv8 Analysis Models ──────────────────────────────────────────────────

class Keypoint(BaseModel):
    """A single pose keypoint (x, y, confidence)."""
    x: float
    y: float
    confidence: float


class PlayerDetection(BaseModel):
    """A single player detected in a single frame."""
    track_id: Optional[int] = None
    bbox: list[float] = Field(description="[x1, y1, x2, y2] bounding box")
    confidence: float
    center_x: float
    center_y: float
    keypoints: list[Keypoint] = Field(default_factory=list, description="17 COCO keypoints")
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    speed: float = 0.0


class ContactZone(BaseModel):
    """Two players detected in close proximity — potential contact."""
    player_a_id: Optional[int]
    player_b_id: Optional[int]
    distance: float = Field(description="Pixel distance between players")
    frame_index: int


class FrameAnalysis(BaseModel):
    """YOLO analysis for a single video frame."""
    frame_index: int
    timestamp_sec: float
    players: list[PlayerDetection] = Field(default_factory=list)
    contact_zones: list[ContactZone] = Field(default_factory=list)
    num_players: int = 0


class YOLOAnalysisResult(BaseModel):
    """Complete YOLO analysis output across all frames."""
    total_frames: int
    fps: float
    duration_sec: float
    frame_analyses: list[FrameAnalysis] = Field(default_factory=list)
    key_contact_frames: list[int] = Field(
        default_factory=list,
        description="Frame indices where significant player contact was detected"
    )
    max_players_detected: int = 0
    annotated_video_path: Optional[str] = None

    def get_summary(self) -> str:
        """Generate a text summary of the YOLO analysis for the LLM."""
        lines = []
        lines.append(f"=== YOLO PLAYER DETECTION SUMMARY ===")
        lines.append(f"Video: {self.total_frames} frames, {self.fps:.1f} FPS, {self.duration_sec:.2f}s duration")
        lines.append(f"Max players detected in single frame: {self.max_players_detected}")
        lines.append(f"Key contact frames: {self.key_contact_frames}")
        lines.append("")

        # Summarize contact zones
        all_contacts = []
        for fa in self.frame_analyses:
            for cz in fa.contact_zones:
                all_contacts.append(cz)

        if all_contacts:
            lines.append(f"CONTACT EVENTS DETECTED: {len(all_contacts)}")
            for cz in all_contacts[:20]:  # Limit to 20 most relevant
                lines.append(
                    f"  Frame {cz.frame_index}: Player {cz.player_a_id} <-> Player {cz.player_b_id}, "
                    f"distance={cz.distance:.1f}px"
                )
        else:
            lines.append("NO SIGNIFICANT CONTACT DETECTED")

        lines.append("")

        # Summarize key frames with player data
        for frame_idx in self.key_contact_frames[:10]:
            matching = [fa for fa in self.frame_analyses if fa.frame_index == frame_idx]
            if matching:
                fa = matching[0]
                lines.append(f"--- Frame {fa.frame_index} (t={fa.timestamp_sec:.2f}s) ---")
                for p in fa.players:
                    kp_summary = ""
                    if p.keypoints:
                        # Report key body part positions
                        kp_names = [
                            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
                            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                            "left_wrist", "right_wrist", "left_hip", "right_hip",
                            "left_knee", "right_knee", "left_ankle", "right_ankle"
                        ]
                        important_kps = [5, 6, 11, 12, 13, 14, 15, 16]  # shoulders, hips, knees, ankles
                        parts = []
                        for ki in important_kps:
                            if ki < len(p.keypoints) and p.keypoints[ki].confidence > 0.3:
                                kp = p.keypoints[ki]
                                parts.append(f"{kp_names[ki]}=({kp.x:.0f},{kp.y:.0f})")
                        kp_summary = " | " + ", ".join(parts) if parts else ""

                    lines.append(
                        f"  Player {p.track_id}: pos=({p.center_x:.0f},{p.center_y:.0f}), "
                        f"speed={p.speed:.1f}px/f, conf={p.confidence:.2f}{kp_summary}"
                    )
                lines.append("")

        return "\n".join(lines)


# ─── Gemini Visual Analysis Models ───────────────────────────────────────────

class GeminiVisualAnalysis(BaseModel):
    """Structured output from Gemini's video understanding (Stage 2)."""
    scene_description: str = Field(description="Brief description of the scene — what is happening in the clip")
    ball_possession: str = Field(description="Which player/team appears to have ball possession")
    challenge_type: str = Field(description="Type of challenge: standing tackle, sliding tackle, aerial, shoulder charge, etc.")
    initial_contact_point: str = Field(description="What is contacted first: ball, player's legs, player's body, or simultaneous")
    contact_body_area: str = Field(description="Where on the body contact is made: ankles, shins, knees, thighs, torso, head")
    challenge_direction: str = Field(description="Direction of the challenge: from behind, from the side, head-on, aerial")
    force_assessment: str = Field(description="Assessment of force used: minimal, moderate, significant, excessive")
    studs_showing: bool = Field(description="Whether studs are visible / raised during the challenge")
    two_footed: bool = Field(description="Whether this is a two-footed challenge")
    simulation_suspected: bool = Field(description="Whether the fouled player appears to exaggerate or simulate")
    ball_playing_distance: bool = Field(description="Whether the ball is within playing distance during the challenge")
    attacking_position: str = Field(description="Was the fouled player in a promising attacking position or DOGSO situation")
    additional_observations: str = Field(description="Any other relevant observations about the incident")


# ─── Final VAR Decision Models ───────────────────────────────────────────────

class VARDecision(BaseModel):
    """The final VAR decision — output of the rules engine (Stage 3)."""
    is_foul: bool = Field(description="Whether the incident constitutes a foul under FIFA Law 12")
    foul_type: str = Field(description="Type of foul: tripping, kicking, pushing, charging, holding, etc. 'none' if no foul")
    severity: str = Field(description="Severity: careless, reckless, excessive_force, or none")
    card_recommendation: str = Field(description="Card recommendation: none, yellow, or red")
    confidence: float = Field(description="Confidence in this decision from 0.0 to 1.0")
    reasoning: str = Field(description="Detailed reasoning explaining the decision, like a VAR review explanation")
    key_factors: list[str] = Field(description="List of key evidence factors that led to this decision")
    alternative_interpretation: str = Field(description="What the counter-argument or alternative interpretation could be")
    fifa_law_reference: str = Field(description="Specific FIFA Law 12 clause that applies")
    free_kick_recommendation: str = Field(description="Direct free kick, indirect free kick, penalty kick, or none")
    advantage_consideration: str = Field(description="Whether advantage could/should be played")


# ─── Combined Result ─────────────────────────────────────────────────────────

class AnalysisStatus(str, Enum):
    PENDING = "pending"
    YOLO_PROCESSING = "yolo_processing"
    GEMINI_ANALYZING = "gemini_analyzing"
    RULES_ENGINE = "rules_engine"
    COMPLETED = "completed"
    ERROR = "error"


class AnalysisResult(BaseModel):
    """Combined result from all three pipeline stages."""
    job_id: str
    status: AnalysisStatus = AnalysisStatus.PENDING
    video_filename: str = ""
    error_message: Optional[str] = None

    # Stage outputs
    yolo_analysis: Optional[YOLOAnalysisResult] = None
    gemini_analysis: Optional[GeminiVisualAnalysis] = None
    var_decision: Optional[VARDecision] = None

    # Metadata
    processing_time_sec: Optional[float] = None
    annotated_video_path: Optional[str] = None
