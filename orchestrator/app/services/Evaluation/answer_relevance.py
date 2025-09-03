"""
Answer Relevance Evaluation Module
Evaluates how relevant the AI's answer is to the given query
"""

import os
import json
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

async def evaluate_answer_relevance(query: str, answer: str) -> dict:
    """
    Evaluate the relevance of an answer given a query
    
    Args:
        query: The original question or prompt
        answer: The AI's response to evaluate
        
    Returns:
        dict: {"score": float, "reasoning": str}
    """
    
    prompt = f"""
    You are an expert evaluator assessing the relevance of AI responses.
    
    Evaluate how relevant the following answer is to the given query:
    
    Query: {query}
    Answer: {answer}
    
    Consider the following criteria:
    1. Direct relevance - Does the answer directly address the query?
    2. Contextual alignment - Does it stay within the scope of the question?
    3. Information utility - Is the provided information useful for the query?
    4. Focus - Does it avoid unnecessary tangents or irrelevant details?
    
    Provide your evaluation as a JSON object with:
    - "score": A float between 0.0 (completely irrelevant) and 1.0 (perfectly relevant)
    - "reasoning": A detailed explanation of your evaluation
    
    Focus on how well the answer addresses the specific query asked.
    """
    
    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert AI response evaluator. Always respond with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=500
        )
        
        result = json.loads(response.choices[0].message.content)
        
        # Ensure score is within valid range
        score = max(0.0, min(1.0, float(result.get("score", 0.0))))
        reasoning = result.get("reasoning", "No reasoning provided")
        
        return {"score": score, "reasoning": reasoning}
        
    except Exception as e:
        return {"score": 0.0, "reasoning": f"Error during evaluation: {str(e)}"}
