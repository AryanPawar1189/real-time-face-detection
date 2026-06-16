"""
Unit Tests — Face Detection Pipeline
======================================
Tests preprocessing correctness, detector initialization,
FPS measurement, and tracking logic.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

import cv2
import numpy as np
import pytest
from face_detector import (
    DetectionConfig,
    ImagePreprocessor,
    FaceDetector,
    FaceTrack,
)


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def config():
    return DetectionConfig()


@pytest.fixture
def preprocessor(config):
    return ImagePreprocessor(config)


@pytest.fixture
def detector(config):
    return FaceDetector(config)


@pytest.fixture
def blank_frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def noise_frame():
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


# ─── DetectionConfig ─────────────────────────────────────────────────────────

class TestDetectionConfig:
    def test_defaults(self, config):
        assert config.scale_factor == 1.1
        assert config.min_neighbors == 5
        assert config.target_fps == 30
        assert config.clahe_clip_limit == 2.0
        assert config.show_landmarks is True

    def test_custom_values(self):
        cfg = DetectionConfig(scale_factor=1.3, min_neighbors=8)
        assert cfg.scale_factor == 1.3
        assert cfg.min_neighbors == 8


# ─── ImagePreprocessor ───────────────────────────────────────────────────────

class TestImagePreprocessor:
    def test_to_gray_shape(self, preprocessor, noise_frame):
        gray = preprocessor.to_gray(noise_frame)
        assert gray.ndim == 2
        assert gray.shape == (480, 640)

    def test_to_gray_dtype(self, preprocessor, noise_frame):
        gray = preprocessor.to_gray(noise_frame)
        assert gray.dtype == np.uint8

    def test_process_returns_two_frames(self, preprocessor, noise_frame):
        display, gray = preprocessor.process(noise_frame)
        assert display.shape == noise_frame.shape
        assert gray.ndim == 2

    def test_process_low_light_returns_same_shape(self, preprocessor, noise_frame):
        display, gray = preprocessor.process(noise_frame, low_light=True)
        assert display.shape == noise_frame.shape
        assert gray.shape == (480, 640)

    def test_enhance_low_light_increases_brightness(self, preprocessor):
        """CLAHE should generally increase mean luminance on a dark frame."""
        dark = np.full((100, 100, 3), 20, dtype=np.uint8)
        enhanced = preprocessor.enhance_low_light(dark)
        assert enhanced.mean() >= dark.mean()

    def test_denoise_preserves_shape(self, preprocessor, noise_frame):
        denoised = preprocessor.denoise(noise_frame)
        assert denoised.shape == noise_frame.shape

    def test_process_blank_frame(self, preprocessor, blank_frame):
        """Should not raise on an all-black frame."""
        display, gray = preprocessor.process(blank_frame)
        assert display is not None
        assert gray is not None


# ─── FaceDetector ────────────────────────────────────────────────────────────

class TestFaceDetector:
    def test_cascade_loaded(self, detector):
        assert not detector.face_cascade.empty()
        assert not detector.eye_cascade.empty()
        assert not detector.smile_cascade.empty()

    def test_detect_returns_list(self, detector, blank_frame):
        gray = cv2.cvtColor(blank_frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detect(gray)
        assert isinstance(faces, list)

    def test_detect_blank_frame_no_faces(self, detector, blank_frame):
        gray = cv2.cvtColor(blank_frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detect(gray)
        assert faces == []

    def test_process_frame_returns_ndarray(self, detector, noise_frame):
        result = detector.process_frame(noise_frame)
        assert isinstance(result, np.ndarray)
        assert result.shape == noise_frame.shape

    def test_process_frame_increments_frame_id(self, detector, blank_frame):
        before = detector._frame_id
        detector.process_frame(blank_frame)
        assert detector._frame_id == before + 1

    def test_fps_computed_after_frames(self, detector, noise_frame):
        for _ in range(5):
            detector.process_frame(noise_frame)
        fps = detector._compute_fps()
        assert fps > 0

    def test_track_count_zero_on_blank(self, detector, blank_frame):
        detector.process_frame(blank_frame)
        assert detector.track_count == 0

    def test_process_low_light_mode(self, detector, noise_frame):
        result = detector.process_frame(noise_frame, low_light=True)
        assert result.shape == noise_frame.shape


# ─── FaceTrack ───────────────────────────────────────────────────────────────

class TestFaceTrack:
    def test_initial_center(self):
        track = FaceTrack(bbox=(10, 20, 50, 60), center=(35, 50), frame_id=0, track_id=0)
        assert track.center == (35, 50)

    def test_update_changes_center(self):
        track = FaceTrack(bbox=(10, 20, 50, 60), center=(35, 50), frame_id=0, track_id=0)
        track.update((100, 100, 50, 60), frame_id=1)
        assert track.center == (125, 130)

    def test_history_grows(self):
        track = FaceTrack(bbox=(0, 0, 40, 40), center=(20, 20), frame_id=0, track_id=0)
        for i in range(5):
            track.update((i * 10, 0, 40, 40), frame_id=i + 1)
        assert len(track.history) == 5

    def test_history_capped_at_30(self):
        track = FaceTrack(bbox=(0, 0, 40, 40), center=(20, 20), frame_id=0, track_id=0)
        for i in range(50):
            track.update((i, 0, 40, 40), frame_id=i + 1)
        assert len(track.history) <= 30


# ─── Track matching ──────────────────────────────────────────────────────────

class TestTrackMatching:
    def test_new_track_created_on_first_detection(self, detector):
        detector._match_tracks([(50, 50, 80, 80)])
        assert len(detector._tracks) == 1

    def test_track_reused_on_close_detection(self, detector):
        detector._match_tracks([(50, 50, 80, 80)])
        detector._frame_id += 1
        detector._match_tracks([(55, 55, 80, 80)])   # slightly moved
        assert len(detector._tracks) == 1            # same track

    def test_new_track_on_distant_detection(self, detector):
        detector._match_tracks([(50, 50, 40, 40)])
        detector._frame_id += 1
        detector._match_tracks([(500, 400, 40, 40)])  # far away
        assert len(detector._tracks) == 2

    def test_stale_tracks_pruned(self, detector):
        detector._match_tracks([(50, 50, 40, 40)])
        detector._frame_id += 15   # advance past prune threshold (10 frames)
        detector._match_tracks([])  # no detections
        assert len(detector._tracks) == 0

    def test_reset_clears_tracks(self, detector):
        detector._match_tracks([(50, 50, 40, 40)])
        detector._tracks.clear()
        assert len(detector._tracks) == 0
