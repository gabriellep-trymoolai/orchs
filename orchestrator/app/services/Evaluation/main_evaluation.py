"""
FastAPI Evaluation Framework
Main orchestration file for AI evaluation modules
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import uvicorn

# Import evaluation modules
from answer_correctness import evaluate_answer_correctness
from answer_relevance import evaluate_answer_relevance
from goal_accuracy import evaluate_goal_accuracy
from hallucination import evaluate_hallucination
from toxicity import evaluate_toxicity
from summarization import evaluate_summarization
from human_vs_ai import evaluate_human_vs_ai

# Load environment variables
load_dotenv()

app = FastAPI(
    title="AI Evaluation Framework",
    description="A comprehensive evaluation framework for AI responses",
    version="1.0.0"
)

class EvaluationInput(BaseModel):
    query: str
    answer: str

class EvaluationOutput(BaseModel):
    score: float
    reasoning: str

@app.get("/")
async def root():
    return {"message": "AI Evaluation Framework is running"}

@app.post("/evaluation/answer_correctness", response_model=EvaluationOutput)
async def evaluate_correctness(input_data: EvaluationInput):
    """Evaluate answer correctness"""
    try:
        result = await evaluate_answer_correctness(input_data.query, input_data.answer)
        return EvaluationOutput(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/evaluation/answer_relevance", response_model=EvaluationOutput)
async def evaluate_relevance(input_data: EvaluationInput):
    """Evaluate answer relevance"""
    try:
        result = await evaluate_answer_relevance(input_data.query, input_data.answer)
        return EvaluationOutput(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/evaluation/goal_accuracy", response_model=EvaluationOutput)
async def evaluate_goal(input_data: EvaluationInput):
    """Evaluate goal accuracy"""
    try:
        result = await evaluate_goal_accuracy(input_data.query, input_data.answer)
        return EvaluationOutput(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/evaluation/hallucination", response_model=EvaluationOutput)
async def evaluate_hallucination_endpoint(input_data: EvaluationInput):
    """Evaluate hallucination"""
    try:
        result = await evaluate_hallucination(input_data.query, input_data.answer)
        return EvaluationOutput(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/evaluation/toxicity", response_model=EvaluationOutput)
async def evaluate_toxicity_endpoint(input_data: EvaluationInput):
    """Evaluate toxicity"""
    try:
        result = await evaluate_toxicity(input_data.query, input_data.answer)
        return EvaluationOutput(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/evaluation/summarization", response_model=EvaluationOutput)
async def evaluate_summarization_endpoint(input_data: EvaluationInput):
    """Evaluate summarization quality"""
    try:
        result = await evaluate_summarization(input_data.query, input_data.answer)
        return EvaluationOutput(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/evaluation/human_vs_ai", response_model=EvaluationOutput)
async def evaluate_human_vs_ai_endpoint(input_data: EvaluationInput):
    """Evaluate human vs AI quality"""
    try:
        result = await evaluate_human_vs_ai(input_data.query, input_data.answer)
        return EvaluationOutput(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8003)
