from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

from agents.v1 import AgentV1
from agents.v1_tuned import AgentV1Tuned
from agents.v2 import AgentV2
from agents.v2_embedding import AgentV2Embedding
from agents.v3 import AgentV3
from starter.agent import BaselineAgent


class RegisteredAgent(Protocol):
    def reset(self, session_id: str, user_profile: dict) -> None:
        ...

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        ...

    def close(self) -> None:
        ...


@dataclass(frozen=True)
class AgentSpec:
    name: str
    description: str
    factory: Callable[[str | Path], RegisteredAgent]


AGENTS = {
    "baseline": AgentSpec(
        name="baseline",
        description=(
            "Original non-LLM baseline. It performs stateless weighted BM25 retrieval "
            "using only the latest customer message."
        ),
        factory=BaselineAgent,
    ),
    "v1": AgentSpec(
        name="v1",
        description=(
            "First non-LLM agent. It combines stateful intent routing, category and FTS "
            "candidate generation, popularity-aware weighted reranking, and deterministic "
            "information-gain clarification questions. Products already shown are not "
            "recommended again unless the customer overrides the request."
        ),
        factory=AgentV1,
    ),
    "v1-tuned": AgentSpec(
        name="v1-tuned",
        description=(
            "Benchmark-specialized V1. It loads ranking weights selected on a "
            "product-disjoint generated training and validation corpus."
        ),
        factory=AgentV1Tuned,
    ),
    "v2": AgentSpec(
        name="v2",
        description=(
            "V1 retrieval with offline-LLM product attributes and family-weighted "
            "fine-grained clarification questions. Evaluator-facing questions use "
            "benchmark-compatible attribute names."
        ),
        factory=AgentV2,
    ),
    "v2-embedding": AgentSpec(
        name="v2-embedding",
        description=(
            "V2 clarification and reranking with hybrid lexical, category, and Qwen3 "
            "dense retrieval. Dense relevance is conservatively margin-calibrated once "
            "the customer supplies a constraint."
        ),
        factory=AgentV2Embedding,
    ),
    "v3": AgentSpec(
        name="v3",
        description=(
            "V2 retrieval with LLM-based conversational state parsing, continuous "
            "buying-versus-browsing weight blending, and LLM-written questions."
        ),
        factory=AgentV3,
    ),
}


def agent_names() -> tuple[str, ...]:
    return tuple(AGENTS)


def get_agent_spec(name: str) -> AgentSpec:
    return AGENTS[name]
