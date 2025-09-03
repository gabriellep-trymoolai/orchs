"""Orchestrator agents package."""

import os
import sys
from datetime import datetime
from typing import Optional

# Add path to main_response.py
current_dir = os.path.dirname(os.path.abspath(__file__))
main_response_path = os.path.join(current_dir, 'Prompt Response')
sys.path.insert(0, main_response_path)

try:
    from main_response import generate_llm_response, QueryRequest, QueryResponse
finally:
    if main_response_path in sys.path:
        sys.path.remove(main_response_path)

class PromptResponseAgent:
    def __init__(self, openai_api_key=None, organization_id="default"):
        self.openai_api_key = openai_api_key
        self.organization_id = organization_id

    async def process_prompt(self, request, db_session=None):
        """Process prompt using the main_response.py implementation"""
        try:
            # Get model from request or use default
            model = getattr(request, 'model', 'gpt-3.5-turbo')
            
            # Call the main_response function with model parameter
            result = await generate_llm_response(request.query, request.session_id, model=model)
            
            # Return a response object compatible with the API
            class AgentResponse:
                def __init__(self, result, model):
                    self.prompt_id = f"prompt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    self.response = result.get("answer", "")
                    self.model = model
                    self.total_tokens = result.get("tokens_used", 0)
                    self.cost = result.get("cost", 0.0)
                    self.latency_ms = result.get("latency_ms", 0)
                    self.timestamp = datetime.now()
                    # Add cache information
                    self.from_cache = result.get("from_cache", False)
                    self.cache_similarity = result.get("similarity", None)
            
            return AgentResponse(result, model)
            
        except Exception as e:
            raise Exception(f"Agent processing failed: {str(e)}")

class PromptResponseService:
    def __init__(self, organization_id="default"):
        self.organization_id = organization_id
        self.agent = PromptResponseAgent(organization_id=organization_id)

__all__ = ["PromptResponseAgent", "PromptResponseService", "generate_llm_response", "QueryRequest", "QueryResponse"]