"""FastAPI dependency providers for orchestrator services."""

from fastapi import Request


async def get_prompt_agent(request: Request):
	"""Dependency to get prompt response agent from app state."""
	if hasattr(request.app.state, 'prompt_agent') and request.app.state.prompt_agent:
		return request.app.state.prompt_agent
	else:
		# Fallback: create a new agent instance
		import os
		from ..agents import PromptResponseAgent
		return PromptResponseAgent(
			openai_api_key=os.getenv("OPENAI_API_KEY"),
			organization_id=os.getenv("ORGANIZATION_ID", "default-org")
		)