"""
Answer Correctness Evaluation Module
Evaluates how correct and accurate the AI's answer is relative to the query
"""

import os
import json
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

async def evaluate_answer_correctness(query: str, answer: str) -> dict:
    """
    Evaluate the correctness of an answer given a query
    
    Args:
        query: The original question or prompt
        answer: The AI's response to evaluate
        
    Returns:
        dict: {"score": float, "reasoning": str}
    """
    
    prompt = f"""
    You are an expert evaluator assessing the correctness of AI responses. 
    
    Evaluate the following answer for correctness based on the given query:
    
    Query: {query}
    Answer: {answer}
    
    Consider the following criteria:
    1. Factual accuracy - Are the facts stated correct?
    2. Logical consistency - Does the answer follow logical reasoning?
    3. Completeness - Does it address all parts of the query?
    4. Precision - Is the information specific and accurate?
    
    Provide your evaluation as a JSON object with exactly this format:
    {{
        "score": 0.85,
        "reasoning": "Your detailed explanation here as a single string"
    }}
    
    IMPORTANT: The reasoning field must be a single string, not an object or array.
    Focus on objective correctness rather than style or presentation.
    """
    
    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert AI response evaluator. Always respond with valid JSON where 'reasoning' is a single string."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=500
        )
        
        result = json.loads(response.choices[0].message.content)
        
        # Ensure score is within valid range
        score = max(0.0, min(1.0, float(result.get("score", 0.0))))
        
        # Handle reasoning field - ensure it's always a string
        reasoning = result.get("reasoning", "No reasoning provided")
        
        # If reasoning is a dict, convert it to a readable string
        if isinstance(reasoning, dict):
            reasoning_parts = []
            for key, value in reasoning.items():
                reasoning_parts.append(f"{key.replace('_', ' ').title()}: {value}")
            reasoning = "; ".join(reasoning_parts)
        elif not isinstance(reasoning, str):
            reasoning = str(reasoning)
        
        return {"score": score, "reasoning": reasoning}
        
    except json.JSONDecodeError as e:
        return {"score": 0.0, "reasoning": f"JSON parsing error: {str(e)}"}
    except Exception as e:
        return {"score": 0.0, "reasoning": f"Error during evaluation: {str(e)}"}