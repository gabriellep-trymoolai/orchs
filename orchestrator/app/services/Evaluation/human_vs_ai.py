"""
Human vs AI Evaluation Module
Evaluates how human-like the AI's answer is in terms of quality and naturalness
"""

import os
import json
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

async def evaluate_human_vs_ai(query: str, answer: str) -> dict:
    """
    Evaluate how human-like and high-quality the answer is compared to what a human expert might provide
    
    Args:
        query: The original question or prompt
        answer: The AI's response to evaluate
        
    Returns:
        dict: {"score": float, "reasoning": str}
    """
    
    prompt = f"""
    You are an expert evaluator comparing AI responses to human expert quality.
    
    Evaluate how the following answer compares to what a knowledgeable human expert might provide:
    
    Query: {query}
    Answer: {answer}
    
    Consider the following criteria:
    1. Natural language flow - Does it sound natural and conversational like human speech?
    2. Contextual understanding - Does it demonstrate deep understanding like a human expert?
    3. Nuanced reasoning - Does it show sophisticated thinking and consideration of complexities?
    4. Appropriate tone - Is the tone suitable and human-like for the context?
    5. Personal insight - Does it provide the kind of insights a human expert would offer?
    6. Emotional intelligence - Does it appropriately recognize and respond to emotional contexts?
    7. Practical wisdom - Does it demonstrate real-world understanding beyond just facts?
    8. Conversational quality - Would this feel like talking to a knowledgeable human?
    
    Provide your evaluation as a JSON object with:
    - "score": A float between 0.0 (clearly AI-generated, low quality) and 1.0 (indistinguishable from expert human)
    - "reasoning": A detailed explanation of your evaluation
    
    Focus on overall quality and human-likeness rather than just technical accuracy.
    """
    
    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert AI response evaluator specializing in human-AI quality comparison. Always respond with valid JSON."},
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
