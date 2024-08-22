"""
Test Emotion Detection
"""
import json
import unittest
from EmotionDetection.emotion_detection import emotion_detector


class TestEmotionDetection(unittest.TestCase):
    """Test case for the emotion_detection module."""

    def test_emotion_detector(self):
        """Test the emotion_detector function with a set of predefined statements."""
        with open('test.json', 'r', encoding='utf-8') as f:
            statements = json.load(f)['statements']

        for statement in statements:
            result = emotion_detector(statement['text'])
            self.assertEqual(result['dominant_emotion'], statement['expected_emotion'])


unittest.main()
