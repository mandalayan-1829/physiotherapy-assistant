"""
Tests for core/angle_calculator.py
"""

import sys
import os
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.angle_calculator import calculate_angle, get_landmark_coords


def test_right_angle():
    """90-degree angle: A=(0,1), B=(0,0), C=(1,0)"""
    angle = calculate_angle([0, 1], [0, 0], [1, 0])
    assert abs(angle - 90.0) < 0.1, f"Expected ~90°, got {angle}°"


def test_straight_line():
    """180-degree angle: straight line"""
    angle = calculate_angle([0, 1], [0, 0], [0, -1])
    assert abs(angle - 180.0) < 0.1, f"Expected ~180°, got {angle}°"


def test_45_degree_angle():
    """45-degree angle"""
    angle = calculate_angle([1, 1], [0, 0], [1, 0])
    assert abs(angle - 45.0) < 0.5, f"Expected ~45°, got {angle}°"


def test_zero_degree_angle():
    """0-degree angle: overlapping vectors"""
    angle = calculate_angle([0, 1], [0, 0], [0, 2])
    assert abs(angle - 0.0) < 0.1, f"Expected ~0°, got {angle}°"


def test_obtuse_angle():
    """Obtuse angle (> 90°)"""
    angle = calculate_angle([-1, 1], [0, 0], [1, 0])
    assert angle > 90, f"Expected obtuse angle, got {angle}°"
    assert angle < 136, f"Expected ~135°, got {angle}°"


def test_get_landmark_coords():
    """Test landmark coordinate extraction with mock landmarks."""

    class MockLandmark:
        def __init__(self, x, y):
            self.x = x
            self.y = y

    landmarks = [MockLandmark(0.5, 0.3), MockLandmark(0.8, 0.6)]
    coords = get_landmark_coords(landmarks, 0)
    assert coords == [0.5, 0.3], f"Expected [0.5, 0.3], got {coords}"


def test_negative_coordinates():
    """Test with negative coordinates (shouldn't happen in practice but should work)."""
    angle = calculate_angle([-1, -1], [0, 0], [1, 0])
    assert 0 <= angle <= 180, f"Angle out of range: {angle}°"
