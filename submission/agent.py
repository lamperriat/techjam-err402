"""Official err402 agent entry point."""

if __package__:
    from .src.err402.agents.v1 import AgentV1
    from .src.err402.agents.v2 import AgentV2
    from .src.err402.agents.v212 import AgentV212
    from .src.err402.agents.v3 import AgentV3
else:
    from src.err402.agents.v1 import AgentV1
    from src.err402.agents.v2 import AgentV2
    from src.err402.agents.v212 import AgentV212
    from src.err402.agents.v3 import AgentV3


Agent = AgentV1

__all__ = ["Agent", "AgentV1", "AgentV2", "AgentV212", "AgentV3"]
