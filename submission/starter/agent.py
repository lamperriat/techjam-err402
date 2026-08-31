"""Compatibility entry point for evaluators importing starter.agent."""

if __package__ == "submission.starter":
    from ..agent import Agent, AgentV1, AgentV2, AgentV3
else:
    from agent import Agent, AgentV1, AgentV2, AgentV3

__all__ = ["Agent", "AgentV1", "AgentV2", "AgentV3"]
