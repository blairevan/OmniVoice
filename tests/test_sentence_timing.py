"""Tests for aligned and fallback sentence timing."""

import unittest

from omnivoice.utils.sentence_timing import (
    TimingResolution,
    build_weighted_sentence_timings,
    resolve_sentence_timings,
)
from omnivoice.utils.whisperx_alignment import AlignedCharacter


class SentenceTimingTests(unittest.TestCase):
    """Verify timing coverage, interpolation, and sample closure."""

    def test_repeated_characters_use_order_preserving_matches(self) -> None:
        """Match repeated characters by sequence position rather than membership."""
        characters = tuple(
            AlignedCharacter(char, index * 0.1, (index + 1) * 0.1)
            for index, char in enumerate("人人都说")
        )

        result = resolve_sentence_timings(
            ("人人", "都说"),
            characters,
            sample_rate=1000,
            generated_samples=400,
            left_overlap_samples=0,
            right_overlap_samples=0,
        )

        self.assertEqual("whisperx_alignment", result.timing_method)
        self.assertEqual(1.0, result.alignment_coverage)
        self.assertEqual((0, 200), (result.timings[0].start_sample, result.timings[0].end_sample))

    def test_95_percent_coverage_interpolates_missing_interior_unit(self) -> None:
        """Allow exactly one interior missing unit when both neighboring anchors exist."""
        expected = "甲乙丙丁戊己庚辛壬癸子丑寅卯辰巳午未申酉"
        actual = expected.replace("庚", "")
        characters = tuple(
            AlignedCharacter(char, index * 0.1, (index + 1) * 0.1)
            for index, char in enumerate(actual)
        )

        result = resolve_sentence_timings(
            (expected,),
            characters,
            sample_rate=1000,
            generated_samples=2000,
            left_overlap_samples=0,
            right_overlap_samples=0,
        )

        self.assertEqual("whisperx_alignment", result.timing_method)
        self.assertAlmostEqual(0.95, result.alignment_coverage)

    def test_95_percent_coverage_rejects_unbounded_missing_unit(self) -> None:
        """Fall back when a missing edge unit has no timestamp on both sides."""
        expected = "甲乙丙丁戊己庚辛壬癸子丑寅卯辰巳午未申酉"
        characters = tuple(
            AlignedCharacter(char, index * 0.1, (index + 1) * 0.1)
            for index, char in enumerate(expected[1:])
        )

        result = resolve_sentence_timings(
            (expected,),
            characters,
            sample_rate=1000,
            generated_samples=2000,
            left_overlap_samples=0,
            right_overlap_samples=0,
        )

        self.assertEqual("pronunciation_weight_fallback", result.timing_method)
        self.assertEqual("alignment_missing_unbounded_unit", result.fallback_reason)

    def test_missing_sentence_forces_whole_group_fallback(self) -> None:
        """Do not mix weighted timing into a group with an unaligned sentence."""
        result = resolve_sentence_timings(
            ("甲", "乙"),
            (AlignedCharacter("甲", 0.0, 0.2),),
            sample_rate=1000,
            generated_samples=400,
            left_overlap_samples=20,
            right_overlap_samples=20,
        )

        self.assertEqual("pronunciation_weight_fallback", result.timing_method)
        self.assertEqual(2, len(result.timings))
        self.assertEqual(20, result.timings[0].start_sample)
        self.assertEqual(380, result.timings[-1].end_sample)

    def test_weighted_timing_closes_integer_sample_window(self) -> None:
        """Assign the final rounded sample to the last sentence."""
        timings = build_weighted_sentence_timings(("甲", "乙", "丙"), 101, 10, 10)

        self.assertEqual(10, timings[0].start_sample)
        self.assertEqual(91, timings[-1].end_sample)
        self.assertTrue(all(item.start_sample < item.end_sample for item in timings))


if __name__ == "__main__":
    unittest.main()
