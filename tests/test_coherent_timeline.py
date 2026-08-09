"""Tests for normalized text layers and coherent synthesis groups."""

import unittest

from omnivoice.utils.coherent_timeline import (
    SubtitleUnit,
    _count_model_tokens,
    build_coherent_synthesis_groups,
    merge_coherent_units,
)


class _HuggingFaceStyleTokenizer:
    """Expose a token sequence separately from BatchEncoding-like metadata."""

    def tokenize(self, text: str) -> list[str]:
        """Return one token per source character for deterministic tests."""
        return list(text)

    def __call__(self, text: str) -> dict[str, list[int]]:
        """Return metadata whose mapping length must not be treated as token count."""
        return {"input_ids": list(range(len(text)))}


class CoherentTimelineTests(unittest.TestCase):
    """Verify the pure coherent-group contracts."""

    def test_merge_preserves_three_text_layers_and_offsets(self) -> None:
        """Merge adjacent units without losing display or alignment mappings."""
        first = SubtitleUnit(
            sentence_id=1,
            display_text="展示词。",
            synthesis_text="synthesis one。",
            alignment_text="synthesis one。",
            alignment_units=("s", "y"),
            synthesis_range=(0, 15),
            alignment_unit_range=(0, 2),
        )
        second = SubtitleUnit(
            sentence_id=2,
            display_text="课程。",
            synthesis_text="课程。",
            alignment_text="课程。",
            alignment_units=("课", "程"),
            synthesis_range=(0, 3),
            alignment_unit_range=(0, 2),
        )

        group = merge_coherent_units((first, second))

        self.assertEqual((1, 2), tuple(unit.sentence_id for unit in group.units))
        self.assertEqual("展示词。课程。", group.display_text)
        self.assertEqual("synthesis one。课程。", group.synthesis_text)
        self.assertEqual("synthesis one。课程。", group.alignment_text)
        self.assertEqual(("s", "y", "课", "程"), group.alignment_units)

    def test_groups_stop_at_token_limit_without_truncation(self) -> None:
        """Pack complete units and split only at sentence boundaries."""
        units = tuple(
            SubtitleUnit(
                sentence_id=index,
                display_text=text,
                synthesis_text=text,
                alignment_text=text,
                alignment_units=tuple(text),
                synthesis_range=(0, len(text)),
                alignment_unit_range=(0, len(text)),
            )
            for index, text in enumerate(("第一句。", "第二句。", "第三句。"), 1)
        )

        groups = build_coherent_synthesis_groups(
            units,
            tokenizer=_HuggingFaceStyleTokenizer(),
            max_tokens=6,
            max_synthesis_chars=80,
        )

        self.assertEqual(3, len(groups))
        self.assertEqual(["第一句。", "第二句。", "第三句。"], [group.display_text for group in groups])

    def test_merge_offsets_connect_alignment_on_synthesis_axis(self) -> None:
        """Keep connect character offsets valid after a pronunciation-length change."""
        first = SubtitleUnit(
            sentence_id=1,
            display_text="你",
            synthesis_text=" ni ",
            alignment_text="你",
            alignment_units=("你",),
            synthesis_range=(0, 4),
            alignment_unit_range=(0, 1),
            alignment_source_offsets=(1,),
        )
        second = SubtitleUnit(
            sentence_id=2,
            display_text="你好",
            synthesis_text="你好",
            alignment_text="你好",
            alignment_units=("你", "好"),
            synthesis_range=(0, 2),
            alignment_unit_range=(0, 2),
            alignment_source_offsets=(0, 1),
            connect_ranges=((0, 2),),
        )

        group = merge_coherent_units((first, second))

        self.assertEqual((1, 4, 5), group.alignment_source_offsets)
        self.assertEqual(((4, 6),), group.connect_ranges)

    def test_token_counter_rejects_ambiguous_tokenizer_output(self) -> None:
        """Do not silently treat tokenizer metadata keys as token IDs."""
        with self.assertRaisesRegex(TypeError, "input_ids"):
            _count_model_tokens(lambda _text: {"attention_mask": [1, 1]}, "测试")

    def test_token_counter_rejects_multiple_batches_for_one_string(self) -> None:
        """Require one input string to resolve to exactly one token-ID sequence."""
        with self.assertRaisesRegex(TypeError, "multiple batches"):
            _count_model_tokens(
                lambda _text: {"input_ids": [[1, 2], [3, 4]]},
                "测试",
            )

    def test_group_rejects_unit_larger_than_limit(self) -> None:
        """Never silently truncate a unit that exceeds the model limit."""
        unit = SubtitleUnit(
            sentence_id=1,
            display_text="超长句。",
            synthesis_text="超长句。",
            alignment_text="超长句。",
            alignment_units=("超", "长", "句"),
            synthesis_range=(0, 4),
            alignment_unit_range=(0, 3),
        )

        with self.assertRaisesRegex(ValueError, "exceeds"):
            build_coherent_synthesis_groups(
                (unit,),
                tokenizer=lambda text: list(text),
                max_tokens=2,
                max_synthesis_chars=80,
            )


if __name__ == "__main__":
    unittest.main()
