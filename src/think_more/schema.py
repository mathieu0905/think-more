"""Schema definitions for state.json."""
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class HypothesisStatus(str, Enum):
    """Status of a hypothesis."""
    ACTIVE = "active"
    ELIMINATED = "eliminated"
    CONFIRMED = "confirmed"


class Hypothesis(BaseModel):
    """A debugging hypothesis."""
    id: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    status: HypothesisStatus = HypothesisStatus.ACTIVE
    evidence: list[str] = Field(default_factory=list)


class Probe(BaseModel):
    """Current test probe with intent and prediction."""
    intent: str = Field(..., min_length=1)
    prediction: dict[str, str] = Field(...)
    test_command: str | None = None

    @field_validator("intent")
    @classmethod
    def intent_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("intent cannot be empty")
        return v

    @field_validator("prediction")
    @classmethod
    def prediction_has_required_keys(cls, v: dict) -> dict:
        required = {"if_pass", "if_fail"}
        if not required.issubset(v.keys()):
            raise ValueError("prediction must have 'if_pass' and 'if_fail' keys")
        return v


class HistoryEntry(BaseModel):
    """A single entry in the reasoning history."""
    round: int = Field(..., ge=1)
    probe: Probe | None = None
    result: str | None = None
    update: str | None = None
    timestamp: datetime = Field(default_factory=datetime.now)


class MCPCall(BaseModel):
    """Record of an MCP tool call."""
    tool: str
    input: dict[str, Any]
    output: dict[str, Any] | None = None


class DataflowChain(BaseModel):
    """Dataflow analysis results."""
    summary: str | None = None
    mcp_callgraph: MCPCall | None = None
    mcp_defuse: MCPCall | None = None


class State(BaseModel):
    """The main state.json schema."""
    version: int = 1
    task_id: str = Field(..., min_length=1)
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    dataflow_chain: DataflowChain | None = None
    current_probe: Probe | None = None
    history: list[HistoryEntry] = Field(default_factory=list)
