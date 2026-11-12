#!/usr/bin/env python3

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from math_benchmarks import score_benchmark_answer


class MathBenchmarkScoringTest(unittest.TestCase):
    def test_gsm8k_strict_format(self):
        result = score_benchmark_answer("GSM8K", "The answer is 18.", "18")
        self.assertTrue(result["score_correct"])
        self.assertTrue(result["format_compliant"])

    def test_gsm8k_currency_and_units_are_scoreable(self):
        for output in ("The answer is $18.", "The answer is 18 dollars."):
            with self.subTest(output=output):
                result = score_benchmark_answer("GSM8K", output, "18")
                self.assertTrue(result["score_correct"])
                self.assertFalse(result["format_compliant"])

    def test_gsm8k_last_number_fallback(self):
        result = score_benchmark_answer("GSM8K", "After calculating, the total is 18", "18")
        self.assertTrue(result["score_correct"])
        self.assertEqual(result["answer_parse_method"], "last_line_number")

    def test_math500_fraction_equivalence(self):
        result = score_benchmark_answer("MATH500", "The answer is 14/3.", r"\frac{14}{3}")
        self.assertTrue(result["score_correct"])
        self.assertEqual(result["verifier_method"], "prm800k_grade_answer")

    def test_math500_decimal_fraction_equivalence(self):
        result = score_benchmark_answer("MATH500", "The answer is 0.5.", r"\frac{1}{2}")
        self.assertTrue(result["score_correct"])

    def test_math500_radical_equivalence(self):
        result = score_benchmark_answer("MATH500", r"Final answer: $2\sqrt{2}$.", r"\sqrt{8}")
        self.assertTrue(result["score_correct"])

    def test_math500_tuple_equivalence(self):
        result = score_benchmark_answer(
            "MATH500",
            r"The answer is $(3, \pi/2)$.",
            r"\left( 3, \frac{\pi}{2} \right)",
        )
        self.assertTrue(result["score_correct"])


if __name__ == "__main__":
    unittest.main()
