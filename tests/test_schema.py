"""Tests for state.json schema validation."""
import pytest
from think_more.schema import (
    Hypothesis,
    Probe,
    HistoryEntry,
    DataflowChain,
    State,
    HypothesisStatus,
)


class TestHypothesis:
    def test_create_valid_hypothesis(self):
        h = Hypothesis(
            id="h1",
            description="QuerySet.filter() returns None for empty list",
            status=HypothesisStatus.ACTIVE,
        )
        assert h.id == "h1"
        assert h.status == HypothesisStatus.ACTIVE
        assert h.evidence == []

    def test_hypothesis_with_evidence(self):
        h = Hypothesis(
            id="h1",
            description="Test",
            status=HypothesisStatus.ACTIVE,
            evidence=["MCP trace shows None return"],
        )
        assert len(h.evidence) == 1


class TestProbe:
    def test_create_valid_probe(self):
        probe = Probe(
            intent="Verify filter([]) return type",
            prediction={
                "if_pass": "h1 eliminated",
                "if_fail": "h1 confirmed",
            },
            test_command="pytest tests/test_filter.py -k empty",
        )
        assert probe.intent == "Verify filter([]) return type"
        assert "if_pass" in probe.prediction

    def test_probe_requires_intent(self):
        with pytest.raises(ValueError):
            Probe(
                intent="",  # Empty intent should fail
                prediction={"if_pass": "x", "if_fail": "y"},
            )


class TestState:
    def test_create_minimal_state(self):
        state = State(task_id="django__django-12345")
        assert state.version == 1
        assert state.hypotheses == []
        assert state.current_probe is None

    def test_create_full_state(self):
        state = State(
            task_id="django__django-12345",
            hypotheses=[
                Hypothesis(
                    id="h1",
                    description="Test hypothesis",
                    status=HypothesisStatus.ACTIVE,
                )
            ],
            current_probe=Probe(
                intent="Test intent",
                prediction={"if_pass": "a", "if_fail": "b"},
                test_command="pytest",
            ),
        )
        assert len(state.hypotheses) == 1
        assert state.current_probe is not None

    def test_state_to_json(self):
        state = State(task_id="test-123")
        json_str = state.model_dump_json(indent=2)
        assert "test-123" in json_str

    def test_state_from_json(self):
        json_data = {
            "version": 1,
            "task_id": "test-123",
            "hypotheses": [],
        }
        state = State.model_validate(json_data)
        assert state.task_id == "test-123"
