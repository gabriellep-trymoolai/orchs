"""
Hallucination Evaluation Module
Evaluates whether the AI's answer contains hallucinations or fabricated information
"""

import os
import json
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

async def evaluate_hallucination(query: str, answer: str) -> dict:
    """
    Evaluate the presence of hallucinations in the answer
    
    Args:
        query: The original question or prompt
        answer: The AI's response to evaluate
        
    Returns:
        dict: {"score": float, "reasoning": str}
    """
    
    prompt = f"""
    You are an expert evaluator detecting hallucinations and fabricated information in AI responses.
    
    Evaluate the following answer for hallucinations or fabricated content:
    
    Query: {query}
    Answer: {answer}
    
    Consider the following indicators of hallucination:
    1. Fabricated facts - Claims that cannot be verified or are likely false
    2. Made-up sources - References to non-existent papers, websites, or authorities
    3. Impossible scenarios - Descriptions of events or situations that couldn't have occurred
    4. Contradictory information - Internal inconsistencies within the answer
    5. Overconfident claims - Stating uncertain information as definitive fact
    6. Anachronisms - Information that doesn't align with known timelines
    
    Provide your evaluation as a JSON object with:
    - "score": A float between 0.0 (severe hallucinations) and 1.0 (no hallucinations detected)
    - "reasoning": A detailed explanation of any hallucinations found or why the answer appears truthful
    
    Be thorough in your analysis and err on the side of caution when detecting potential fabrications.
    """
    
    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert AI response evaluator specializing in hallucination detection. Always respond with valid JSON."},
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
