"""Surgery agent: a frontier model with tools to inspect and edit weights, run after the rule-based
playbook could not fix a failure; the playbook's checks stay the judge."""
from .loop import AgentLimits, brief, run  # noqa: F401
from .providers import AnthropicProvider, OpenAIProvider, ScriptedProvider  # noqa: F401
from .workbench import Budget, Workbench  # noqa: F401
