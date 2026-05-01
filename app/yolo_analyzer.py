"""
Stage 1: YOLOv8 Pose Detection & Player Tracking

Detects all players in each frame, extracts 17-point pose keypoints,
tracks player IDs across frames with ByteTrack, and identifies contact zones.
"""

import cv2
import numpy as np
import math
import os
from ultralytics import YOLO
from .models import (
    Keypoint,
    PlayerDetection,
    ContactZone,
    FrameAnalysis,
    YOLOAnalysisResult,
)


# Distance threshold (pixels) to consider two players as being in a "contact zone"
CONTACT_DISTANCE_THRESHOLD = 120

# Minimum confidence for a detection to be considered
MIN_DETECTION_CONFIDENCE = 0.3

# COCO keypoint names for reference
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]


class YOLOAnalyzer:
    """Runs YOLOv8 pose estimation and tracking on football video clips."""

    def __init__(self, model_name: str = "yolov8x-pose.pt"):
        """
        Initialize the YOLO analyzer.

        Args:
            model_name: YOLOv8 pose model variant. Options:
                - yolov8n-pose.pt (nano — fastest, least accurate)
                - yolov8s-pose.pt (small)
                - yolov8m-pose.pt (medium)
                - yolov8l-pose.pt (large)
                - yolov8x-pose.pt (extra large — slowest, most accurate)
        """
        print(f"[YOLO] Loading model: {model_name}")
        self.model = YOLO(model_name)
        self.previous_positions: dict[int, tuple[float, float]] = {}

    def analyze(
        self,
        video_path: str,
        output_dir: str,
        sample_rate: int = 2,
    ) -> YOLOAnalysisResult:
        """
        Run full YOLO analysis on a video file.

        Args:
            video_path: Path to the input video file.
            output_dir: Directory to save annotated output video.
            sample_rate: Process every Nth frame (1 = every frame, 2 = every other frame).

        Returns:
            YOLOAnalysisResult with all frame analyses and contact detections.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / fps
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"[YOLO] Video: {total_frames} frames, {fps:.1f} FPS, {duration_sec:.1f}s, {width}x{height}")

        # Set up annotated video writer
        os.makedirs(output_dir, exist_ok=True)
        annotated_path = os.path.join(output_dir, "annotated.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(annotated_path, fourcc, fps, (width, height))

        frame_analyses: list[FrameAnalysis] = []
        key_contact_frames: list[int] = []
        max_players = 0
        self.previous_positions = {}
        frame_index = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            timestamp_sec = frame_index / fps

            if frame_index % sample_rate == 0:
                # Run tracking with pose estimation
                results = self.model.track(
                    source=frame,
                    persist=True,
                    conf=MIN_DETECTION_CONFIDENCE,
                    verbose=False,
                    tracker="bytetrack.yaml",
                )

                frame_analysis = self._process_frame(results, frame_index, timestamp_sec)
                frame_analyses.append(frame_analysis)

                # Track max players
                if frame_analysis.num_players > max_players:
                    max_players = frame_analysis.num_players

                # Flag key contact frames
                if frame_analysis.contact_zones:
                    key_contact_frames.append(frame_index)

                # Write annotated frame
                annotated_frame = results[0].plot() if results else frame
                # Add contact zone highlights
                annotated_frame = self._draw_contact_zones(annotated_frame, frame_analysis)
                writer.write(annotated_frame)
            else:
                writer.write(frame)

            frame_index += 1

            # Progress logging
            if frame_index % 50 == 0:
                print(f"[YOLO] Processed frame {frame_index}/{total_frames}")

        cap.release()
        writer.release()
        print(f"[YOLO] Analysis complete. {len(frame_analyses)} frames analyzed, "
              f"{len(key_contact_frames)} contact frames found.")

        return YOLOAnalysisResult(
            total_frames=total_frames,
            fps=fps,
            duration_sec=duration_sec,
            frame_analyses=frame_analyses,
            key_contact_frames=key_contact_frames,
            max_players_detected=max_players,
            annotated_video_path=annotated_path,
        )

    def _process_frame(
        self,
        results,
        frame_index: int,
        timestamp_sec: float,
    ) -> FrameAnalysis:
        """Process YOLO results for a single frame."""
        players: list[PlayerDetection] = []

        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            keypoints_data = results[0].keypoints if results[0].keypoints is not None else None

            for i in range(len(boxes)):
                box = boxes[i]
                conf = float(box.conf[0])

                # Bounding box
                xyxy = box.xyxy[0].cpu().numpy()
                bbox = [float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])]
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2

                # Track ID
                track_id = None
                if box.id is not None:
                    track_id = int(box.id[0])

                # Pose keypoints
                kps: list[Keypoint] = []
                if keypoints_data is not None and i < len(keypoints_data):
                    kp_data = keypoints_data[i]
                    xy = kp_data.xy[0].cpu().numpy() if kp_data.xy is not None else None
                    kp_conf = kp_data.conf[0].cpu().numpy() if kp_data.conf is not None else None

                    if xy is not None:
                        for j in range(len(xy)):
                            c = float(kp_conf[j]) if kp_conf is not None and j < len(kp_conf) else 0.0
                            kps.append(Keypoint(x=float(xy[j][0]), y=float(xy[j][1]), confidence=c))

                # Compute velocity
                vel_x, vel_y, speed = 0.0, 0.0, 0.0
                if track_id is not None and track_id in self.previous_positions:
                    prev_x, prev_y = self.previous_positions[track_id]
                    vel_x = center_x - prev_x
                    vel_y = center_y - prev_y
                    speed = math.sqrt(vel_x ** 2 + vel_y ** 2)

                # Update position history
                if track_id is not None:
                    self.previous_positions[track_id] = (center_x, center_y)

                players.append(PlayerDetection(
                    track_id=track_id,
                    bbox=bbox,
                    confidence=conf,
                    center_x=center_x,
                    center_y=center_y,
                    keypoints=kps,
                    velocity_x=vel_x,
                    velocity_y=vel_y,
                    speed=speed,
                ))

        # Detect contact zones — pairs of players that are very close
        contact_zones = self._detect_contact_zones(players, frame_index)

        return FrameAnalysis(
            frame_index=frame_index,
            timestamp_sec=timestamp_sec,
            players=players,
            contact_zones=contact_zones,
            num_players=len(players),
        )

    def _detect_contact_zones(
        self,
        players: list[PlayerDetection],
        frame_index: int,
    ) -> list[ContactZone]:
        """Find pairs of players within contact distance."""
        contacts: list[ContactZone] = []

        for i in range(len(players)):
            for j in range(i + 1, len(players)):
                dist = math.sqrt(
                    (players[i].center_x - players[j].center_x) ** 2 +
                    (players[i].center_y - players[j].center_y) ** 2
                )

                if dist < CONTACT_DISTANCE_THRESHOLD:
                    contacts.append(ContactZone(
                        player_a_id=players[i].track_id,
                        player_b_id=players[j].track_id,
                        distance=dist,
                        frame_index=frame_index,
                    ))

        return contacts

    def _draw_contact_zones(self, frame: np.ndarray, analysis: FrameAnalysis) -> np.ndarray:
        """Draw red highlights around contact zones on the frame."""
        for cz in analysis.contact_zones:
            # Find the two players
            player_a = None
            player_b = None
            for p in analysis.players:
                if p.track_id == cz.player_a_id:
                    player_a = p
                elif p.track_id == cz.player_b_id:
                    player_b = p

            if player_a and player_b:
                # Draw a red line between the two players
                pt1 = (int(player_a.center_x), int(player_a.center_y))
                pt2 = (int(player_b.center_x), int(player_b.center_y))
                cv2.line(frame, pt1, pt2, (0, 0, 255), 3)

                # Draw a red circle at the midpoint
                mid_x = int((player_a.center_x + player_b.center_x) / 2)
                mid_y = int((player_a.center_y + player_b.center_y) / 2)
                cv2.circle(frame, (mid_x, mid_y), 20, (0, 0, 255), 3)

                # Label the contact
                cv2.putText(
                    frame,
                    f"CONTACT {cz.distance:.0f}px",
                    (mid_x - 50, mid_y - 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )

        return frame
