"""
Goal Accuracy Evaluation Module
Evaluates how well the AI's answer achieves the intended goal of the query
"""

import os
import json
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

async def evaluate_goal_accuracy(query: str, answer: str) -> dict:
    """
    Evaluate how well the answer achieves the goal of the query
    
    Args:
        query: The original question or prompt
        answer: The AI's response to evaluate
        
    Returns:
        dict: {"score": float, "reasoning": str}
    """
    
    prompt = f"""
    You are an expert evaluator assessing how well AI responses achieve their intended goals.
    
    Evaluate how well the following answer achieves the goal implied by the query:
    
    Query: {query}
    Answer: {answer}
    
    Consider the following criteria:
    1. Goal identification - What was the user trying to accomplish?
    2. Solution effectiveness - Does the answer provide an effective solution?
    3. Actionability - Can the user take concrete steps based on this answer?
    4. Completeness - Does it fully address what the user needs?
    5. Practical value - Will this answer actually help the user achieve their goal?
    
    Provide your evaluation as a JSON object with exactly this format:
    {{
        "score": 0.85,
        "reasoning": "Your detailed explanation here as a single string"
    }}
    
    IMPORTANT: The reasoning field must be a single string, not an object or array.
    Focus on whether the answer actually helps the user accomplish what they set out to do.
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
