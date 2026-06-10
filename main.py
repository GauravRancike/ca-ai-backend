from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List
import google.generativeai as genai
import json
import os

app = FastAPI(title="CA Inter AI Evaluation Backend")

# Allow Lovable to talk to this server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows any frontend to connect
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class StepWiseMarks(BaseModel):
    step_name: str
    allocated_marks: float
    awarded_marks: float
    justification: str

class ICAIEvaluationReport(BaseModel):
    total_score: float
    max_marks: float
    step_breakdown: List[StepWiseMarks]
    matched_keywords: List[str]
    missing_keywords: List[str]
    examiner_feedback: str

@app.post("/api/evaluate-answer", response_model=ICAIEvaluationReport)
async def evaluate_answer(file: UploadFile = File(...)):
    try:
        # Mocking the database retrieval for now
        suggested_answer = "As per SA 315, Inherent Risk is the susceptibility of an assertion to a misstatement..."
        max_question_marks = 5.0
        
        image_bytes = await file.read()
        
        # Pulls the API key you will set in Render
        genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
        model = genai.GenerativeModel('gemini-2.5-pro')
        
        system_instruction = f"""
        You are an expert ICAI Examiner evaluating CA Intermediate Group 2 exam papers.
        Evaluate the student's handwritten answer image strictly against the Official Suggested Answer.
        
        Official Suggested Answer: {suggested_answer}
        Maximum Marks: {max_question_marks}
        
        Apply step-wise marking and check for precise technical vocabulary.
        """
        
        response = model.generate_content(
            contents=[{"mime_type": file.content_type, "data": image_bytes}, system_instruction],
            generation_config={"response_mime_type": "application/json", "response_schema": ICAIEvaluationReport}
        )
        
        return json.loads(response.text)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
