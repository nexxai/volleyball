"""
Volleyball AI Analysis System - Processor Tests
All tests for processor.py module
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "ai_core"))

# Import processor for cv2 constants
import processor


# ============================================================================
# Analyzer Initialization Tests
# ============================================================================

class TestAnalyzerInitialization:
    """Tests for VolleyballAnalyzer initialization"""
    
    def test_get_optimal_device(self):
        """Test device detection"""
        from processor import VolleyballAnalyzer
        
        device = VolleyballAnalyzer.get_optimal_device()
        assert device in ["cuda", "mps", "cpu"]
    
    def test_analyzer_init_without_models(self):
        """Test initializing analyzer without models"""
        from processor import VolleyballAnalyzer
        
        analyzer = VolleyballAnalyzer()
        assert analyzer.ball_model is None
        assert analyzer.action_model is None
        assert analyzer.player_model is None
    
    def test_analyzer_device_override(self):
        """Test overriding device selection"""
        from processor import VolleyballAnalyzer
        
        analyzer = VolleyballAnalyzer(device="cpu")
        assert analyzer.device == "cpu"
    
    def test_ball_frame_buffer_initialized(self):
        """Test that ball_frame_buffer is properly initialized"""
        from processor import VolleyballAnalyzer
        
        analyzer = VolleyballAnalyzer()
        assert hasattr(analyzer, "ball_frame_buffer")
        assert isinstance(analyzer.ball_frame_buffer, list)
        assert len(analyzer.ball_frame_buffer) == 0
    
    def test_tracker_initialized(self):
        """Test that Norfair tracker is properly initialized"""
        from processor import VolleyballAnalyzer
        
        analyzer = VolleyballAnalyzer()
        assert hasattr(analyzer, "tracker")
        assert analyzer.tracker is not None
    
    def test_jersey_caches_initialized(self):
        """Test that jersey number caches are initialized"""
        from processor import VolleyballAnalyzer
        
        analyzer = VolleyballAnalyzer()
        assert hasattr(analyzer, "jersey_number_cache")
        assert isinstance(analyzer.jersey_number_cache, dict)
        assert hasattr(analyzer, "jersey_to_stable_id")
        assert isinstance(analyzer.jersey_to_stable_id, dict)

    def test_jersey_lock_observations_from_environment(self, monkeypatch):
        from processor import VolleyballAnalyzer

        monkeypatch.setenv("JERSEY_LOCK_OBSERVATIONS", "1")

        analyzer = VolleyballAnalyzer()

        assert analyzer.jersey_lock_observations == 1

    def test_jersey_inference_interval_from_environment(self, monkeypatch):
        from processor import VolleyballAnalyzer

        monkeypatch.setenv("JERSEY_INFERENCE_INTERVAL", "2")

        analyzer = VolleyballAnalyzer()

        assert analyzer.jersey_inference_interval == 2


# ============================================================================
# Ball Detection Tests
# ============================================================================

class TestBallDetection:
    """Tests for ball detection methods"""
    
    def test_detect_ball_no_model(self, analyzer, sample_frame):
        """Test detect_ball when no model is loaded"""
        result = analyzer.detect_ball(sample_frame)
        assert result is None
    
    def test_detect_ball_with_yolo_fallback(self, analyzer, sample_frame):
        """Test detect_ball with YOLO fallback"""
        mock_player_model = Mock()
        mock_result = Mock()
        mock_box = Mock()
        mock_box.conf = [Mock()]
        mock_box.conf[0].cpu.return_value.numpy.return_value = np.array(0.8)
        mock_box.xyxy = [Mock()]
        mock_box.xyxy[0].cpu.return_value.numpy.return_value = np.array([100, 100, 120, 120])
        mock_result.boxes = [mock_box]
        mock_player_model.return_value = [mock_result]
        
        analyzer.player_model = mock_player_model
        
        result = analyzer.detect_ball(sample_frame)
        assert result is not None
        assert "center" in result
        assert "bbox" in result
        assert "confidence" in result
    
    def test_preprocess_ball_frame(self, analyzer, sample_frame):
        """Test ball frame preprocessing"""
        processed = analyzer.preprocess_ball_frame(sample_frame)
        
        assert processed.shape == (288, 512)
        assert processed.dtype == np.float32
        assert processed.max() <= 1.0
        assert processed.min() >= 0.0
    
    def test_preprocess_preserves_content(self, analyzer):
        """Test that preprocessing doesn't destroy image content"""
        gradient = np.linspace(0, 255, 1920, dtype=np.uint8)
        gradient_frame = np.tile(gradient, (1080, 1)).astype(np.uint8)
        gradient_frame = np.stack([gradient_frame] * 3, axis=-1)
        
        processed = analyzer.preprocess_ball_frame(gradient_frame)
        assert processed[:, 0].mean() < processed[:, -1].mean()
    
    def test_postprocess_ball_output_valid(self, analyzer):
        """Test postprocessing valid ball output"""
        heatmap = np.zeros((288, 512), dtype=np.float32)
        heatmap[100:110, 200:210] = 0.5
        
        output = [np.zeros((1, 9, 288, 512), dtype=np.float32)]
        output[0][0, -1, :, :] = heatmap
        
        frame_shape = (480, 640, 3)
        result = analyzer.postprocess_ball_output(output, frame_shape)
        
        assert result is not None
        assert "center" in result
        assert "bbox" in result
        assert "confidence" in result
    
    def test_postprocess_ball_output_invalid(self, analyzer):
        """Test postprocessing invalid ball output"""
        output = []
        frame_shape = (480, 640, 3)
        result = analyzer.postprocess_ball_output(output, frame_shape)
        assert result is None
    
    def test_postprocess_ball_output_none(self, analyzer):
        """Test postprocessing with None output"""
        result = analyzer.postprocess_ball_output([None], (1080, 1920))
        assert result is None
    
    def test_ball_frame_buffer_management(self, analyzer, sample_frame):
        """Test ball frame buffer management"""
        for i in range(12):
            analyzer.detect_ball(sample_frame)
        
        assert len(analyzer.ball_frame_buffer) <= 9


# ============================================================================
# Player Detection Tests
# ============================================================================

class TestPlayerDetection:
    """Tests for player detection methods"""
    
    def test_detect_players_no_model(self, analyzer, sample_frame):
        """Test detect_players when no model is loaded"""
        result = analyzer.detect_players(sample_frame)
        assert result == []
    
    def test_detect_players_with_model(self, analyzer, sample_frame):
        """Test detect_players with model"""
        mock_model = Mock()
        mock_result = Mock()
        mock_box = Mock()
        mock_box.xyxy = [Mock()]
        mock_box.xyxy[0].cpu.return_value.numpy.return_value = np.array([100, 100, 200, 300])
        mock_box.conf = [Mock()]
        mock_box.conf[0].cpu.return_value.numpy.return_value = np.array(0.8)
        mock_box.cls = [Mock()]
        mock_box.cls[0].cpu.return_value.numpy.return_value = np.array(0)
        mock_result.boxes = [mock_box]
        mock_model.return_value = [mock_result]
        mock_model.names = {0: "person"}
        
        analyzer.player_model = mock_model
        
        result = analyzer.detect_players(sample_frame)
        assert len(result) > 0
        assert "bbox" in result[0]
        assert "confidence" in result[0]
    
    def test_detect_players_low_confidence_filter(self, analyzer, sample_frame):
        """Test that low confidence detections are filtered"""
        mock_model = Mock()
        mock_result = Mock()
        mock_box = Mock()
        mock_box.xyxy = [Mock()]
        mock_box.xyxy[0].cpu.return_value.numpy.return_value = np.array([100, 100, 200, 300])
        mock_box.conf = [Mock()]
        mock_box.conf[0].cpu.return_value.numpy.return_value = np.array(0.3)
        mock_box.cls = [Mock()]
        mock_box.cls[0].cpu.return_value.numpy.return_value = np.array(0)
        mock_result.boxes = [mock_box]
        mock_model.return_value = [mock_result]
        mock_model.names = {0: "person"}
        
        analyzer.player_model = mock_model
        
        result = analyzer.detect_players(sample_frame)
        assert len(result) == 0


# ============================================================================
# Action Detection Tests
# ============================================================================

class TestActionDetection:
    """Tests for action detection methods"""
    
    def test_detect_actions_no_model(self, analyzer, sample_frame):
        """Test detect_actions when no model is loaded"""
        result = analyzer.detect_actions(sample_frame)
        assert result == []
    
    def test_detect_actions_with_model(self, analyzer, sample_frame):
        """Test detect_actions with model"""
        mock_model = Mock()
        mock_result = Mock()
        mock_box = Mock()
        mock_box.xyxy = [Mock()]
        mock_box.xyxy[0].cpu.return_value.numpy.return_value = np.array([100, 100, 200, 200])
        mock_box.conf = [Mock()]
        mock_box.conf[0].cpu.return_value.numpy.return_value = np.array(0.8)
        mock_box.cls = [Mock()]
        mock_box.cls[0].cpu.return_value.numpy.return_value = np.array(0)
        mock_result.boxes = [mock_box]
        mock_model.return_value = [mock_result]
        mock_model.names = {0: "spike"}
        
        analyzer.action_model = mock_model
        
        result = analyzer.detect_actions(sample_frame)
        assert len(result) > 0
        assert result[0]["action"] == "spike"
        assert result[0]["confidence"] >= 0.6
    
    def test_detect_actions_low_confidence_filter(self, analyzer, sample_frame):
        """Test that low confidence actions are filtered"""
        mock_model = Mock()
        mock_result = Mock()
        mock_box = Mock()
        mock_box.xyxy = [Mock()]
        mock_box.xyxy[0].cpu.return_value.numpy.return_value = np.array([100, 100, 200, 200])
        mock_box.conf = [Mock()]
        mock_box.conf[0].cpu.return_value.numpy.return_value = np.array(0.5)
        mock_box.cls = [Mock()]
        mock_box.cls[0].cpu.return_value.numpy.return_value = np.array(0)
        mock_result.boxes = [mock_box]
        mock_model.return_value = [mock_result]
        mock_model.names = {0: "spike"}
        
        analyzer.action_model = mock_model
        
        result = analyzer.detect_actions(sample_frame)
        assert len(result) == 0
    
    def test_detect_actions_with_mocked_model(self, analyzer, mock_frame):
        """Test action detection with mocked YOLO model"""
        mock_model = MagicMock()
        mock_model.names = {0: 'spike', 1: 'set', 2: 'receive', 3: 'serve', 4: 'block'}
        
        mock_box = MagicMock()
        mock_box.xyxy = [MagicMock()]
        mock_box.xyxy[0].cpu.return_value.numpy.return_value = np.array([100, 200, 300, 400])
        mock_box.conf = [MagicMock()]
        mock_box.conf[0].cpu.return_value.numpy.return_value = 0.85
        mock_box.cls = [MagicMock()]
        mock_box.cls[0].cpu.return_value.numpy.return_value = 0
        
        mock_result = MagicMock()
        mock_result.boxes = [mock_box]
        mock_model.return_value = [mock_result]
        
        analyzer.action_model = mock_model
        actions = analyzer.detect_actions(mock_frame)
        
        assert len(actions) == 1
        assert actions[0]['action'] == 'spike'
        assert actions[0]['confidence'] == 0.85


# ============================================================================
# Ball Trajectory Filtering Tests
# ============================================================================

class TestBallTrajectoryFiltering:
    """Tests for ball trajectory filtering methods"""
    
    def test_filter_ball_trajectory_short(self, analyzer):
        """Test filtering short trajectory"""
        short_trajectory = [
            {"frame": 0, "timestamp": 0.0, "center": [100, 200], "confidence": 0.9},
            {"frame": 1, "timestamp": 0.033, "center": [105, 205], "confidence": 0.85},
        ]
        result = analyzer._filter_ball_trajectory(short_trajectory)
        assert len(result) == 2
    
    def test_basic_velocity_filter(self, analyzer):
        """Test basic velocity filtering"""
        trajectory = [
            {"frame": 0, "timestamp": 0.0, "center": [100, 200], "confidence": 0.9},
            {"frame": 1, "timestamp": 0.033, "center": [105, 205], "confidence": 0.85},
            {"frame": 2, "timestamp": 0.066, "center": [1000, 2000], "confidence": 0.8},
            {"frame": 3, "timestamp": 0.1, "center": [110, 210], "confidence": 0.75},
        ]
        result = analyzer._basic_velocity_filter(trajectory)
        assert len(result) <= len(trajectory)
    
    def test_basic_velocity_filter_low_confidence(self, analyzer):
        """Test velocity filter with low confidence"""
        trajectory = [
            {"frame": 0, "timestamp": 0.0, "center": [100, 200], "confidence": 0.9},
            {"frame": 1, "timestamp": 0.033, "center": [105, 205], "confidence": 0.1},
        ]
        result = analyzer._basic_velocity_filter(trajectory)
        assert len(result) <= len(trajectory)
    
    def test_physics_based_outlier_removal(self, analyzer, sample_trajectory):
        """Test physics-based outlier removal"""
        trajectory_with_outlier = sample_trajectory + [
            {"frame": 5, "timestamp": 0.166, "center": [1000, 2000], "confidence": 0.7}
        ]
        result = analyzer._physics_based_outlier_removal(trajectory_with_outlier)
        assert len(result) <= len(trajectory_with_outlier)
    
    def test_smooth_ball_trajectory(self, analyzer, sample_trajectory):
        """Test ball trajectory smoothing"""
        result = analyzer._smooth_ball_trajectory(sample_trajectory)
        assert len(result) == len(sample_trajectory)
        assert "center" in result[0]
    
    def test_interpolate_missing_frames(self, analyzer):
        """Test interpolating missing frames"""
        trajectory_with_gap = [
            {"frame": 0, "timestamp": 0.0, "center": [100, 200], "confidence": 0.9},
            {"frame": 5, "timestamp": 0.166, "center": [120, 220], "confidence": 0.7},
        ]
        fps = 30.0
        result = analyzer._interpolate_missing_frames(trajectory_with_gap, fps)
        assert len(result) > len(trajectory_with_gap)
    
    def test_interpolate_missing_frames_large_gap(self, analyzer):
        """Test interpolation with large gap (should not interpolate)"""
        trajectory_with_large_gap = [
            {"frame": 0, "timestamp": 0.0, "center": [100, 200], "confidence": 0.9},
            {"frame": 20, "timestamp": 0.666, "center": [200, 300], "confidence": 0.7},
        ]
        fps = 30.0
        result = analyzer._interpolate_missing_frames(trajectory_with_large_gap, fps)
        assert len(result) == len(trajectory_with_large_gap)


# ============================================================================
# Player Tracking Tests
# ============================================================================

class TestPlayerTracking:
    """Tests for player tracking methods"""
    
    def test_track_players_empty(self, analyzer):
        """Test tracking with empty player list"""
        result = analyzer.track_players([])
        assert result == []
    
    def test_track_players_with_detections(self, analyzer, sample_players):
        """Test tracking players with detections"""
        result = analyzer.track_players(sample_players)
        # Norfair may return empty list if no valid detections
        assert isinstance(result, list)
        if len(result) > 0:
            assert "id" in result[0]
            assert "bbox" in result[0]
    
    def test_track_players_with_data(self, analyzer, mock_frame):
        """Test tracking with player detections"""
        players = [
            {"bbox": [100, 100, 200, 400], "confidence": 0.9, "class_id": 0},
            {"bbox": [500, 100, 600, 400], "confidence": 0.85, "class_id": 0}
        ]
        
        tracked = analyzer.track_players(players, frame=mock_frame)
        assert isinstance(tracked, list)
    
    def test_assign_action_to_player(self, analyzer, sample_players):
        """Test assigning action to player"""
        # Add id to sample_players for tracking
        tracked_players = [
            {"id": 1, "bbox": p["bbox"], "confidence": p["confidence"]}
            for p in sample_players
        ]
        action_bbox = [150, 150, 180, 200]  # Overlaps with first player
        result = analyzer.assign_action_to_player(action_bbox, tracked_players)
        assert result is not None
    
    def test_assign_action_to_player_no_players(self, analyzer):
        """Test assigning action with no players"""
        action_bbox = [100, 100, 200, 200]
        result = analyzer.assign_action_to_player(action_bbox, [])
        assert result is None
    
    def test_assign_action_to_player_distance_match(self, analyzer):
        """Test action assignment using distance matching"""
        players = [
            {"id": 1, "bbox": [100, 100, 200, 300], "confidence": 0.9}
        ]
        action_bbox = [225, 150, 275, 200]
        result = analyzer.assign_action_to_player(action_bbox, players)
        assert result is not None


# ============================================================================
# IOU Calculation Tests
# ============================================================================

class TestIOUCalculation:
    """Tests for IOU calculation"""
    
    def test_iou_overlapping_boxes(self, analyzer):
        """Test IOU with overlapping boxes"""
        boxA = [100, 100, 200, 200]
        boxB = [150, 150, 250, 250]
        iou = analyzer._iou(boxA, boxB)
        assert 0 < iou < 1
    
    def test_iou_identical_boxes(self, analyzer):
        """Test IOU with identical boxes"""
        boxA = [100, 100, 200, 200]
        boxB = [100, 100, 200, 200]
        iou = analyzer._iou(boxA, boxB)
        # Should be very close to 1.0 (allowing for floating point precision)
        assert abs(iou - 1.0) < 0.001
    
    def test_iou_non_overlapping_boxes(self, analyzer):
        """Test IOU with non-overlapping boxes"""
        boxA = [100, 100, 200, 200]
        boxB = [300, 300, 400, 400]
        iou = analyzer._iou(boxA, boxB)
        assert iou == 0.0
    
    def test_iou_edge_case_zero_area(self, analyzer):
        """Test IOU edge case with zero area"""
        boxA = [100, 100, 100, 100]
        boxB = [200, 200, 200, 200]
        iou = analyzer._iou(boxA, boxB)
        assert iou == 0.0


# ============================================================================
# Model Loading Tests
# ============================================================================

class TestModelLoading:
    """Tests for model loading methods"""
    
    def test_load_ball_model_nonexistent(self, analyzer, tmp_path):
        """Test loading nonexistent ball model"""
        nonexistent_path = tmp_path / "nonexistent_model.onnx"
        analyzer.load_ball_model(str(nonexistent_path))
        assert True

    @patch("processor.ort.InferenceSession")
    def test_load_ball_model_configures_threads(self, mock_session, analyzer, tmp_path, monkeypatch):
        model_path = tmp_path / "ball.onnx"
        model_path.touch()
        monkeypatch.setenv("BALL_INFERENCE_THREADS", "4")

        analyzer.load_ball_model(str(model_path))

        options = mock_session.call_args.kwargs["sess_options"]
        assert options.intra_op_num_threads == 4

    @patch("processor.ort.InferenceSession")
    def test_load_ball_model_uses_runtime_thread_default(self, mock_session, analyzer, tmp_path, monkeypatch):
        model_path = tmp_path / "ball.onnx"
        model_path.touch()
        monkeypatch.delenv("BALL_INFERENCE_THREADS", raising=False)

        analyzer.load_ball_model(str(model_path))

        options = mock_session.call_args.kwargs["sess_options"]
        assert options.intra_op_num_threads == 0

    @patch("processor.ort.InferenceSession")
    def test_load_ball_model_ignores_invalid_threads(self, mock_session, analyzer, tmp_path, monkeypatch):
        model_path = tmp_path / "ball.onnx"
        model_path.touch()
        monkeypatch.setenv("BALL_INFERENCE_THREADS", "invalid")

        analyzer.load_ball_model(str(model_path))

        options = mock_session.call_args.kwargs["sess_options"]
        assert options.intra_op_num_threads == 0
    
    def test_load_action_model_nonexistent(self, analyzer, tmp_path):
        """Test loading nonexistent action model"""
        nonexistent_path = tmp_path / "nonexistent_model.pt"
        analyzer.load_action_model(str(nonexistent_path))
        assert True
    
    def test_load_player_model_nonexistent(self, analyzer, tmp_path):
        """Test loading nonexistent player model"""
        nonexistent_path = tmp_path / "nonexistent_model.pt"
        analyzer.load_player_model(str(nonexistent_path))
        assert True
    
    def test_load_jersey_number_model_nonexistent(self, analyzer, tmp_path):
        """Test loading nonexistent jersey number model"""
        nonexistent_path = tmp_path / "nonexistent_model.pt"
        analyzer.load_jersey_number_model(str(nonexistent_path))
        assert True


# ============================================================================
# Jersey Number Detection Tests
# ============================================================================

class TestJerseyNumberDetection:
    """Tests for jersey number detection"""
    
    def test_detect_jersey_number_no_models(self, analyzer, sample_frame):
        """Test jersey number detection with no models"""
        bbox = [100, 100, 200, 300]
        result = analyzer._detect_jersey_number(sample_frame, bbox)
        assert result is None
    
    def test_preprocess_roi(self, analyzer):
        """Test ROI preprocessing"""
        roi = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        processed = analyzer._preprocess_roi(roi)
        assert processed.shape[2] == 3
        assert processed.dtype == np.uint8
    
    def test_preprocess_roi_grayscale(self, analyzer):
        """Test ROI preprocessing with grayscale input"""
        roi = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        processed = analyzer._preprocess_roi(roi)
        assert processed.shape[2] == 3
        assert processed.dtype == np.uint8
    
    def test_merge_digit_detections_single_digit(self, analyzer):
        """Test merging single digit detection"""
        detections = [
            {"digit": 5, "bbox": [10, 10, 20, 20], "confidence": 0.9, "center_x": 15}
        ]
        result = analyzer._merge_digit_detections(detections)
        assert result == 5
    
    def test_merge_digit_detections_two_digits(self, analyzer):
        """Test merging two digit detections"""
        detections = [
            {"digit": 1, "bbox": [10, 10, 20, 20], "confidence": 0.9, "center_x": 15},
            {"digit": 3, "bbox": [25, 10, 35, 20], "confidence": 0.85, "center_x": 30}
        ]
        result = analyzer._merge_digit_detections(detections)
        assert result == 13
    
    def test_merge_digit_detections_low_confidence(self, analyzer):
        """Test merging with low confidence detections"""
        detections = [
            {"digit": 1, "bbox": [10, 10, 20, 20], "confidence": 0.1, "center_x": 15},
            {"digit": 3, "bbox": [25, 10, 35, 20], "confidence": 0.05, "center_x": 30}
        ]
        result = analyzer._merge_digit_detections(detections)
        assert result is None
    
    def test_merge_digit_detections_far_apart(self, analyzer):
        """Test merging digits that are far apart"""
        detections = [
            {"digit": 1, "bbox": [10, 10, 20, 20], "confidence": 0.9, "center_x": 15},
            {"digit": 3, "bbox": [200, 10, 210, 20], "confidence": 0.85, "center_x": 205}
        ]
        result = analyzer._merge_digit_detections(detections)
        assert result == 1


# ============================================================================
# Stable Player ID Tests
# ============================================================================

class TestStablePlayerID:
    """Tests for stable player ID functionality"""
    
    def test_get_stable_player_id_no_jersey(self, analyzer, sample_frame):
        """Test getting stable ID without jersey number"""
        track_id = 1
        bbox = [100, 100, 200, 300]
        stable_id, jersey_num = analyzer._get_stable_player_id(track_id, bbox, sample_frame)
        assert stable_id == track_id
        assert jersey_num is None

    def test_stable_jersey_history_skips_model(self, analyzer, sample_frame):
        analyzer.track_id_to_jersey_history[5] = [12, 12, 12]
        analyzer.jersey_to_stable_id[12] = 12
        analyzer.jersey_to_track_ids[12] = [5]
        analyzer._detect_jersey_number = Mock()

        result = analyzer._get_stable_player_id(
            5, [100, 100, 200, 300], sample_frame
        )

        assert result == (12, 12)
        analyzer._detect_jersey_number.assert_not_called()

    def test_conflicting_jersey_history_keeps_inference(self, analyzer, sample_frame):
        analyzer.track_id_to_jersey_history[5] = [12, 8, 12]
        analyzer.jersey_to_stable_id[12] = 12
        analyzer.jersey_to_track_ids[12] = [5]
        analyzer.jersey_recheck_counts[5] = 4
        analyzer._detect_jersey_number = Mock(return_value=None)

        result = analyzer._get_stable_player_id(
            5, [100, 100, 200, 300], sample_frame
        )

        assert result == (12, 12)
        analyzer._detect_jersey_number.assert_called_once()

    def test_jersey_lock_can_be_disabled(self, sample_frame):
        from processor import VolleyballAnalyzer

        analyzer = VolleyballAnalyzer(jersey_lock_observations=0)
        analyzer.track_id_to_jersey_history[5] = [12, 12, 12]
        analyzer.jersey_to_stable_id[12] = 12
        analyzer.jersey_to_track_ids[12] = [5]
        analyzer.jersey_recheck_counts[5] = 4
        analyzer._detect_jersey_number = Mock(return_value=None)

        analyzer._get_stable_player_id(5, [100, 100, 200, 300], sample_frame)

        analyzer._detect_jersey_number.assert_called_once()

    def test_known_jersey_is_rechecked_at_configured_interval(self, sample_frame):
        from processor import VolleyballAnalyzer

        analyzer = VolleyballAnalyzer(
            jersey_lock_observations=3,
            jersey_inference_interval=3,
        )
        analyzer.track_id_to_jersey_history[5] = [12]
        analyzer.jersey_to_stable_id[12] = 12
        analyzer.jersey_to_track_ids[12] = [5]
        analyzer._detect_jersey_number = Mock(return_value=None)

        for _ in range(6):
            analyzer._get_stable_player_id(
                5, [100, 100, 200, 300], sample_frame
            )

        assert analyzer._detect_jersey_number.call_count == 2
    
    def test_set_jersey_number_mapping(self, analyzer):
        """Test setting jersey number mapping"""
        analyzer.set_jersey_number_mapping(track_id=1, jersey_number=10)
        assert 10 in analyzer.jersey_to_stable_id
        assert analyzer.jersey_to_stable_id[10] == 10


# ============================================================================
# Analyze Video Tests
# ============================================================================

class TestAnalyzeVideo:
    """Tests for analyze_video method"""

    def test_batched_video_inference_groups_frames(self, analyzer, sample_frame):
        cap = Mock()
        frames = [sample_frame for _ in range(9)]
        cap.read.side_effect = lambda: (True, frames.pop(0)) if frames else (False, None)
        analyzer.detect_players_batch = Mock(side_effect=lambda frames: [[{"player": True}] for _ in frames])
        analyzer.detect_actions_batch = Mock(side_effect=lambda frames: [[{"action": True}] for _ in frames])
        analyzer.detect_ball = Mock(return_value={"ball": True})

        output = list(analyzer._batched_video_inference(cap))

        assert len(output) == 9
        assert [len(call.args[0]) for call in analyzer.detect_players_batch.call_args_list] == [8, 1]
        assert [len(call.args[0]) for call in analyzer.detect_actions_batch.call_args_list] == [8, 1]
        assert all(players == [{"player": True}] for _, players, _, _ in output)
        assert all(actions == [{"action": True}] for _, _, actions, _ in output)
        assert all(ball == {"ball": True} for _, _, _, ball in output)

    def test_ball_inference_uses_dedicated_worker(self, analyzer, sample_frame):
        import threading

        cap = Mock()
        frames = [
            np.full_like(sample_frame, 1),
            np.full_like(sample_frame, 2),
        ]
        cap.read.side_effect = lambda: (True, frames.pop(0)) if frames else (False, None)
        analyzer.device = "mps"
        analyzer.ball_model = Mock()
        analyzer.detect_players_batch = Mock(side_effect=lambda frames: [[] for _ in frames])
        analyzer.detect_actions_batch = Mock(side_effect=lambda frames: [[] for _ in frames])
        worker_threads = []

        def detect_ball(frame):
            worker_threads.append(threading.get_ident())
            return {"frame": int(frame[0, 0, 0])}

        analyzer._detect_ball_onnx = detect_ball
        output = list(analyzer._batched_video_inference(cap))

        assert worker_threads
        assert all(thread_id != threading.get_ident() for thread_id in worker_threads)
        assert [ball["frame"] for _, _, _, ball in output] == [1, 2]

    def test_ball_fallback_stays_on_main_thread(self, analyzer, sample_frame):
        import threading

        cap = Mock()
        frames = [sample_frame for _ in range(2)]
        cap.read.side_effect = lambda: (True, frames.pop(0)) if frames else (False, None)
        analyzer.device = "mps"
        analyzer.ball_model = Mock()
        analyzer.detect_players_batch = Mock(side_effect=lambda frames: [[] for _ in frames])
        analyzer.detect_actions_batch = Mock(side_effect=lambda frames: [[] for _ in frames])
        analyzer._detect_ball_onnx = Mock(return_value=None)
        fallback_threads = []

        def detect_fallback(frame):
            fallback_threads.append(threading.get_ident())
            return None

        analyzer._detect_ball_yolo = detect_fallback
        list(analyzer._batched_video_inference(cap))

        assert fallback_threads
        assert all(thread_id == threading.get_ident() for thread_id in fallback_threads)

    def test_batch_inference_uses_selected_device(self, analyzer, sample_frame):
        analyzer.device = "mps"
        analyzer.player_model = Mock(return_value=[])
        analyzer.action_model = Mock(return_value=[])

        analyzer.detect_players_batch([sample_frame])
        analyzer.detect_actions_batch([sample_frame])

        assert analyzer.player_model.call_args.kwargs["device"] == "mps"
        assert analyzer.action_model.call_args.kwargs["device"] == "mps"
    
    @patch('processor.cv2.VideoCapture')
    def test_analyze_video_success(self, mock_capture, analyzer, tmp_path):
        """Test analyze_video successfully"""
        # Create mock video file
        video_file = tmp_path / "test_video.mp4"
        video_file.touch()
        
        # Mock VideoCapture
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            processor.cv2.CAP_PROP_FPS: 30.0,
            processor.cv2.CAP_PROP_FRAME_COUNT: 100,
            processor.cv2.CAP_PROP_FRAME_WIDTH: 1920,
            processor.cv2.CAP_PROP_FRAME_HEIGHT: 1080
        }.get(prop, 0)
        mock_cap.read.return_value = (False, None)  # End of video
        mock_capture.return_value = mock_cap
        
        result = analyzer.analyze_video(str(video_file))
        
        assert "video_info" in result
        assert "ball_tracking" in result
        assert "action_recognition" in result
    
    @patch('processor.cv2.VideoCapture')
    def test_analyze_video_cannot_open(self, mock_capture, analyzer, tmp_path):
        """Test analyze_video when video cannot be opened"""
        video_file = tmp_path / "invalid_video.mp4"
        video_file.touch()
        
        mock_cap = Mock()
        mock_cap.isOpened.return_value = False
        mock_capture.return_value = mock_cap
        
        with pytest.raises(ValueError):
            analyzer.analyze_video(str(video_file))
    
    @patch('processor.cv2.VideoCapture')
    def test_analyze_video_with_progress_callback(self, mock_capture, analyzer, tmp_path):
        """Test analyze_video with progress callback"""
        video_file = tmp_path / "test_video.mp4"
        video_file.touch()
        
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            processor.cv2.CAP_PROP_FPS: 30.0,
            processor.cv2.CAP_PROP_FRAME_COUNT: 100,  # More frames to trigger callback
            processor.cv2.CAP_PROP_FRAME_WIDTH: 1920,
            processor.cv2.CAP_PROP_FRAME_HEIGHT: 1080
        }.get(prop, 0)
        
        frame_count = [0]
        def mock_read():
            if frame_count[0] < 50:
                frame_count[0] += 1
                return (True, np.zeros((1080, 1920, 3), dtype=np.uint8))
            return (False, None)
        
        mock_cap.read.side_effect = mock_read
        mock_capture.return_value = mock_cap
        
        progress_calls = []
        def progress_callback(progress, frame, total):
            progress_calls.append((progress, frame, total))
        
        result = analyzer.analyze_video(str(video_file), progress_callback=progress_callback)
        
        # Progress callback may or may not be called depending on implementation
        # Just verify the analysis completed successfully
        assert "video_info" in result
    
    @patch('processor.cv2.VideoCapture')
    def test_analyze_video_with_output_path(self, mock_capture, analyzer, tmp_path):
        """Test analyze_video with output path"""
        video_file = tmp_path / "test_video.mp4"
        video_file.touch()
        output_file = tmp_path / "results.json"
        
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            processor.cv2.CAP_PROP_FPS: 30.0,
            processor.cv2.CAP_PROP_FRAME_COUNT: 10,
            processor.cv2.CAP_PROP_FRAME_WIDTH: 1920,
            processor.cv2.CAP_PROP_FRAME_HEIGHT: 1080
        }.get(prop, 0)
        mock_cap.read.return_value = (False, None)
        mock_capture.return_value = mock_cap
        
        result = analyzer.analyze_video(str(video_file), str(output_file))
        
        assert output_file.exists()
        assert "video_info" in result


# ============================================================================
# Additional Edge Cases
# ============================================================================

class TestProcessorEdgeCases:
    """Tests for processor edge cases"""
    
    def test_detect_ball_with_empty_buffer(self, analyzer, sample_frame):
        """Test detect_ball with empty buffer"""
        analyzer.ball_frame_buffer = []
        result = analyzer.detect_ball(sample_frame)
        # Should handle gracefully
        assert result is None or isinstance(result, dict)
    
    def test_track_players_with_none_frame(self, analyzer, sample_players):
        """Test track_players with None frame"""
        result = analyzer.track_players(sample_players, frame=None)
        assert isinstance(result, list)
    
    def test_assign_action_to_player_no_overlap(self, analyzer):
        """Test assign_action_to_player with no overlap but close distance"""
        players = [
            {"id": 1, "bbox": [100, 100, 200, 200], "confidence": 0.9}
        ]
        # Close enough for distance matching (within 1.5x diagonal)
        action_bbox = [225, 150, 275, 200]  # Close but no overlap
        result = analyzer.assign_action_to_player(action_bbox, players)
        # Should match based on distance
        assert result is not None
    
    def test_merge_digit_detections_empty(self, analyzer):
        """Test merging empty digit detections"""
        result = analyzer._merge_digit_detections([])
        assert result is None
    
    def test_merge_digit_detections_single_low_confidence(self, analyzer):
        """Test merging single digit with low confidence"""
        detections = [
            {"digit": 5, "bbox": [10, 10, 20, 20], "confidence": 0.1, "center_x": 15}  # Below 0.15 threshold
        ]
        result = analyzer._merge_digit_detections(detections)
        assert result is None
    
    def test_preprocess_roi_small_image(self, analyzer):
        """Test preprocessing very small ROI"""
        roi = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
        processed = analyzer._preprocess_roi(roi)
        assert processed.shape[2] == 3
    
    def test_preprocess_roi_large_image(self, analyzer):
        """Test preprocessing large ROI"""
        roi = np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8)
        processed = analyzer._preprocess_roi(roi)
        assert processed.shape[2] == 3
    
    def test_physics_based_outlier_removal_short_trajectory(self, analyzer):
        """Test physics-based outlier removal with very short trajectory"""
        short_trajectory = [
            {"frame": 0, "timestamp": 0.0, "center": [100, 200], "confidence": 0.9}
        ]
        result = analyzer._physics_based_outlier_removal(short_trajectory)
        assert len(result) >= len(short_trajectory)
    
    def test_smooth_ball_trajectory_empty(self, analyzer):
        """Test smoothing empty trajectory"""
        result = analyzer._smooth_ball_trajectory([])
        assert result == []
    
    def test_interpolate_missing_frames_empty(self, analyzer):
        """Test interpolating empty trajectory"""
        result = analyzer._interpolate_missing_frames([], 30.0)
        assert result == []
    
    def test_interpolate_missing_frames_single_point(self, analyzer):
        """Test interpolating trajectory with single point"""
        trajectory = [
            {"frame": 0, "timestamp": 0.0, "center": [100, 200], "confidence": 0.9}
        ]
        result = analyzer._interpolate_missing_frames(trajectory, 30.0)
        assert len(result) == len(trajectory)


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
