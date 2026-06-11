"""
Test suite for KINA speech analysis system.

This module contains unit tests and property-based tests for the core
text processing functions in kina.py:
- format_text(): Text formatting with newlines and wrapping
- lexical_diversity(): Vocabulary richness metrics
- sentence_complexity(): Sentence structure analysis
"""

import pytest
from hypothesis import given, strategies as st, settings
from kina import format_text, lexical_diversity, sentence_complexity


# ============================================================================
# Unit Tests for format_text()
# ============================================================================

class TestFormatText:
    """Unit tests for the format_text() function."""
    
    def test_empty_string(self):
        """Test that empty string input returns empty string."""
        result = format_text("")
        assert result == ""
    
    def test_single_sentence(self):
        """Test formatting of a single sentence."""
        text = "This is a test sentence."
        result = format_text(text)
        assert "This is a test sentence." in result
    
    def test_multiple_sentences(self):
        """Test formatting of multiple sentences with different punctuation."""
        text = "First sentence. Second sentence? Third sentence!"
        result = format_text(text)
        # Verify all sentences are present in the result
        assert "First sentence" in result
        assert "Second sentence" in result
        assert "Third sentence" in result


# ============================================================================
# Unit Tests for lexical_diversity()
# ============================================================================

class TestLexicalDiversity:
    """Unit tests for the lexical_diversity() function."""
    
    def test_empty_string(self):
        """Test that empty string returns (0, 0, 0) without errors."""
        score, total, unique = lexical_diversity("")
        assert score == 0
        assert total == 0
        assert unique == 0
    
    def test_all_unique_words(self):
        """Test text with all unique words returns diversity of 1.0."""
        text = "one two three four five"
        score, total, unique = lexical_diversity(text)
        assert score == 1.0
        assert total == 5
        assert unique == 5
    
    def test_all_repeated_words(self):
        """Test text with repeated words returns appropriate diversity."""
        text = "test test test"
        score, total, unique = lexical_diversity(text)
        assert total == 3
        assert unique == 1
        assert score == pytest.approx(1/3)


# ============================================================================
# Unit Tests for sentence_complexity()
# ============================================================================

class TestSentenceComplexity:
    """Unit tests for the sentence_complexity() function."""
    
    def test_empty_string(self):
        """Test that empty string returns (0, 0, feedback) without errors."""
        avg_len, conj_count, feedback = sentence_complexity("")
        assert avg_len == 0
        assert conj_count == 0
        assert isinstance(feedback, str)
    
    def test_single_sentence(self):
        """Test complexity analysis of a single sentence."""
        text = "This is a simple test sentence."
        avg_len, conj_count, feedback = sentence_complexity(text)
        assert avg_len > 0
        assert conj_count >= 0
        assert isinstance(feedback, str)
    
    def test_no_conjunctions(self):
        """Test text with no conjunctions."""
        text = "This is a test. Another test."
        avg_len, conj_count, feedback = sentence_complexity(text)
        assert conj_count == 0


# ============================================================================
# Property-Based Tests
# ============================================================================

class TestFormatTextProperties:
    """Property-based tests for format_text()."""
    pass


class TestLexicalDiversityProperties:
    """Property-based tests for lexical_diversity()."""
    pass


class TestSentenceComplexityProperties:
    """Property-based tests for sentence_complexity()."""
    pass


class TestEdgeCaseProperties:
    """Property-based tests for edge cases and special inputs."""
    pass
