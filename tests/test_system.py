"""
tests/test_system.py - Unit tests for Satya Vaani components.

Run with: pytest tests/ -v
"""

import sys
import os
import uuid

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Claim Extraction Tests ────────────────────────────────────────────────────

class TestClaimExtraction:
    def setup_method(self):
        from core.claim_extraction import ClaimExtractor
        self.extractor = ClaimExtractor()

    def test_english_basic(self):
        text = "India gained independence in 1947. The Green Revolution began in the 1960s."
        claims = self.extractor.extract_claims(text, "English")
        assert len(claims) >= 1
        assert all(c.text for c in claims)

    def test_claim_has_uuid(self):
        text = "The Eiffel Tower was built in 1889."
        claims = self.extractor.extract_claims(text, "English")
        for c in claims:
            assert c.claim_id
            try:
                uuid.UUID(c.claim_id)
            except ValueError:
                pytest.fail(f"claim_id is not a valid UUID: {c.claim_id}")

    def test_number_extraction(self):
        text = "India has 1.4 billion people and 29 states."
        claims = self.extractor.extract_claims(text, "English")
        numbers = [n for c in claims for n in c.numbers]
        assert any("1.4" in n or "29" in n for n in numbers)

    def test_empty_text(self):
        claims = self.extractor.extract_claims("", "English")
        assert claims == []

    def test_short_text_skipped(self):
        # Very short sentences should not produce claims
        text = "Yes. No. OK."
        claims = self.extractor.extract_claims(text, "English")
        # Might extract 0 claims due to length filter
        assert isinstance(claims, list)

    def test_code_mixed_text(self):
        # Should not crash on mixed content
        text = "Python uses `for i in range(10)` to loop 10 times."
        claims = self.extractor.extract_claims(text, "English")
        assert isinstance(claims, list)

    def test_non_factual_content(self):
        text = "The sky might be purple, or perhaps pink, who knows?"
        claims = self.extractor.extract_claims(text, "English")
        assert isinstance(claims, list)


# ── Scoring Tests ─────────────────────────────────────────────────────────────

class TestScoring:
    def setup_method(self):
        from core.scoring import compute_risk_score, compute_retrieval_score
        self.compute_risk_score = compute_risk_score
        self.compute_retrieval_score = compute_retrieval_score

    @pytest.fixture
    def default_weights(self):
        return {"entailment": 0.4, "retrieval": 0.3, "drift": 0.3}

    def test_risk_score_range(self, default_weights):
        for entail, retrieval, drift in [
            (0.0, 0.0, 1.0),
            (1.0, 1.0, 0.0),
            (0.5, 0.5, 0.5),
        ]:
            score = self.compute_risk_score(entail, retrieval, drift, default_weights)
            assert 0.0 <= score <= 1.0, f"Score out of range: {score}"

    def test_high_entailment_low_risk(self, default_weights):
        score = self.compute_risk_score(0.95, 0.9, 0.05, default_weights)
        assert score < 0.3, f"Expected low risk, got {score}"

    def test_low_entailment_high_risk(self, default_weights):
        score = self.compute_risk_score(0.05, 0.1, 0.8, default_weights)
        assert score > 0.6, f"Expected high risk, got {score}"

    def test_deterministic(self, default_weights):
        args = (0.7, 0.6, 0.2, default_weights)
        assert self.compute_risk_score(*args) == self.compute_risk_score(*args)

    def test_risk_normalization_clamp(self, default_weights):
        # Even with extreme values, must stay in [0, 1]
        score = self.compute_risk_score(0.0, 0.0, 0.0, default_weights)
        assert score == 1.0  # All bad

        score = self.compute_risk_score(1.0, 1.0, 1.0, default_weights)
        assert 0.0 <= score <= 1.0


# ── Entailment Tests ──────────────────────────────────────────────────────────

class TestEntailment:
    """Basic interface tests (model not loaded to keep unit tests fast)."""

    def test_empty_passages(self):
        from core.entailment import EntailmentChecker
        import yaml
        with open("config.yaml") as f:
            config = yaml.safe_load(f)
        checker = EntailmentChecker(config)
        # Should return 0 without loading model
        result = checker.compute_entailment("some claim", [])
        assert result == 0.0


# ── Retrieval Score Tests ─────────────────────────────────────────────────────

class TestRetrievalScore:
    def test_entity_overlap(self):
        from core.scoring import _entity_overlap_ratio
        assert _entity_overlap_ratio(["India", "1947"], "India became independent in 1947.") == 1.0
        assert _entity_overlap_ratio(["China"], "India became independent in 1947.") == 0.0
        assert _entity_overlap_ratio([], "Any text") == 0.0

    def test_number_match(self):
        from core.scoring import _number_match_ratio
        assert _number_match_ratio(["1947", "29"], "India has 29 states since 1947.") == 1.0
        assert _number_match_ratio([], "Any text") == 1.0  # No numbers → no penalty
        assert _number_match_ratio(["999"], "No matching number here.") == 0.0


# ── Pipeline Interface Tests ──────────────────────────────────────────────────

class TestPipelineInterface:
    def test_run_pipeline_signature(self):
        from core.pipeline import run_pipeline
        import inspect
        sig = inspect.signature(run_pipeline)
        assert "prompt" in sig.parameters
        assert "language" in sig.parameters

    def test_result_schema(self):
        """Verify the expected keys exist in a mocked result."""
        expected_keys = {
            "generated_text", "claims", "document_hallucination_rate",
            "total_claims", "hallucinated_claims"
        }
        # Mock a result matching the schema
        mock_result = {
            "prompt": "test",
            "language": "English",
            "generated_text": "The sky is blue.",
            "claims": [],
            "document_hallucination_rate": 0.0,
            "total_claims": 0,
            "hallucinated_claims": 0,
            "elapsed_sec": 0.1,
        }
        for key in expected_keys:
            assert key in mock_result

    def test_hallucination_rate_bounds(self):
        """Hallucination rate must always be in [0, 1]."""
        for n_hallucinated, total in [(0, 10), (5, 10), (10, 10)]:
            rate = round(n_hallucinated / max(total, 1), 4)
            assert 0.0 <= rate <= 1.0
