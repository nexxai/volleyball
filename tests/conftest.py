"""
Volleyball AI Analysis System - Shared Test Fixtures
Common fixtures used across all test files
"""

import pytest
import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from io import BytesIO

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
sys.path.insert(0, str(PROJECT_ROOT / "ai_core"))


# ============================================================================
# Common Fixtures
# ============================================================================

@pytest.fixture
def client():
    """Create a FastAPI test client"""
    from fastapi.testclient import TestClient
    from main import app
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def sample_video_dict():
    """Sample video data dictionary"""
    return {
        "id": "test-video-123",
        "filename": "test_match.mp4",
        "file_path": "data/uploads/test_match.mp4",
        "upload_time": "2025-12-10T12:00:00",
        "status": "uploaded",
        "file_size": 1024000
    }


@pytest.fixture
def sample_analysis_results():
    """Sample analysis results"""
    return {
        "video_info": {
            "width": 1920,
            "height": 1080,
            "fps": 30.0,
            "total_frames": 900,
            "duration": 30.0
        },
        "ball_tracking": {
            "trajectory": [
                {"frame": 0, "timestamp": 0.0, "center": [100, 200], "bbox": [90, 190, 110, 210], "confidence": 0.95}
            ],
            "detected_frames": 1,
            "total_frames": 900
        },
        "action_recognition": {
            "actions": [],
            "action_detections": [],
            "action_counts": {"spike": 5, "set": 10},
            "total_actions": 15
        },
        "analysis_time": 45.5
    }


@pytest.fixture
def temp_video_file(tmp_path):
    """Create a temporary mock video file"""
    video_file = tmp_path / "test_video.mp4"
    video_file.write_bytes(b'\x00\x00\x00\x1c\x66\x74\x79\x70\x69\x73\x6f\x6d' + b'\x00' * 1000)
    return video_file


@pytest.fixture
def temp_results_file(tmp_path):
    """Create a temporary results JSON file"""
    results = {
        "video_info": {"width": 1920, "height": 1080, "fps": 30.0, "total_frames": 900, "duration": 30.0},
        "ball_tracking": {"trajectory": [], "detected_frames": 0, "total_frames": 900},
        "action_recognition": {"actions": [], "action_detections": [], "action_counts": {}, "total_actions": 0},
        "analysis_time": 10.5
    }
    results_file = tmp_path / "test_video_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f)
    return results_file


@pytest.fixture
def mock_frame():
    """Create a mock video frame (RGB image)"""
    return np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)


@pytest.fixture
def sample_frame():
    """Create a sample frame (numpy array)"""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_trajectory():
    """Create a sample ball trajectory"""
    return [
        {"frame": 0, "timestamp": 0.0, "center": [100, 200], "confidence": 0.9},
        {"frame": 1, "timestamp": 0.033, "center": [105, 205], "confidence": 0.85},
        {"frame": 2, "timestamp": 0.066, "center": [110, 210], "confidence": 0.8},
        {"frame": 3, "timestamp": 0.1, "center": [115, 215], "confidence": 0.75},
        {"frame": 4, "timestamp": 0.133, "center": [120, 220], "confidence": 0.7},
    ]


@pytest.fixture
def sample_players():
    """Create sample player detections"""
    return [
        {
            "bbox": [100, 100, 200, 300],
            "confidence": 0.9,
            "class_id": 0,
            "label": "player"
        },
        {
            "bbox": [300, 150, 400, 350],
            "confidence": 0.85,
            "class_id": 0,
            "label": "player"
        }
    ]


@pytest.fixture
def sample_video_in_db(tmp_path, client):
    """Create a sample video in database"""
    from database import get_database
    db = get_database()
    
    video_id = "test-video-extended-123"
    video_data = {
        "id": video_id,
        "filename": "test_extended.mp4",
        "file_path": str(tmp_path / "test_extended.mp4"),
        "upload_time": datetime.now().isoformat(),
        "status": "uploaded",
        "file_size": 1024000
    }
    
    # Create the file
    Path(video_data["file_path"]).touch()
    
    db.add_video(video_data)
    yield video_id, video_data
    # Cleanup
    try:
        db.delete_video(video_id)
    except:
        pass


@pytest.fixture
def analyzer():
    """Create analyzer instance without models"""
    from processor import VolleyballAnalyzer
    return VolleyballAnalyzer()
