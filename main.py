from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import google.generativeai as genai
import json
import os

app = FastAPI(title="CA Inter AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- MODELS FOR QUESTION GENERATOR ---
class QuestionRequest(BaseModel):
    subject: str
    chapter: str
    difficulty: str
    count: int
    includePYP: bool

class QuestionItem(BaseModel):
    question: str
    marks: float
    hint: str
    isPYP: bool
    suggested_answer: str

class GeneratedQuestionResponse(BaseModel):
    items: List[QuestionItem]

# --- MODELS FOR EXAMINER ---
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

# --- MODELS FOR GURU ---
class ChatMessage(BaseModel):
    message: str


# ==========================================
# FEATURE 3: QUESTION GENERATOR
# ==========================================
@app.post("/api/generate-questions", response_model=GeneratedQuestionResponse)
async def generate_questions(request: QuestionRequest):
    try:
        genai.configure(api_key=os.environ.get("EXAMINER_API_KEY"))
        
        system_instruction = f"""
        You are an expert ICAI Paper Setter. Generate exactly {request.count} exam questions.
        Subject: {request.subject}
        Chapter: {request.chapter}
        Difficulty: {request.difficulty}
        Include Past Year Questions (PYP): {request.includePYP}
        
        For each question, provide:
        - The question text
        - Marks (usually 4 to 6)
        - A brief hint for the student
        - A boolean flag if it mimics a PYP
        - The official detailed suggested answer (for backend grading later)
        """
        
        model = genai.GenerativeModel('gemini-2.5-flash', system_instruction=system_instruction)
        
        response = model.generate_content(
            contents="Generate the questions now.",
            generation_config={
                "response_mime_type": "application/json", 
                "response_schema": GeneratedQuestionResponse,
                "temperature": 0.4
            }
        )
        
        return json.loads(response.text)
        
    except Exception as e:
        print(f"Generator Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================
# FEATURE 1: THE AI EXAMINER (Image Grading)
# ==========================================
@app.post("/api/evaluate-answer", response_model=ICAIEvaluationReport)
async def evaluate_answer(
    file: UploadFile = File(...), 
    suggested_answer: str = Form(...),
    max_marks: float = Form(...)
):
    try:
        image_bytes = await file.read()
        genai.configure(api_key=os.environ.get("EXAMINER_API_KEY"))
        
        system_instruction = f"""
        You are an expert ICAI Examiner evaluating CA exam papers.
        Evaluate the student's handwritten answer strictly against this Official Suggested Answer:
        {suggested_answer}
        Maximum Marks: {max_marks}
        """
        
        model = genai.GenerativeModel('gemini-2.5-flash', system_instruction=system_instruction)
        
        response = model.generate_content(
            contents=[{"mime_type": file.content_type, "data": image_bytes}],
            generation_config={"response_mime_type": "application/json", "response_schema": ICAIEvaluationReport}
        )
        
        return json.loads(response.text)
        
    except Exception as e:
        print(f"Examiner Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================
# FEATURE 2: THE AI GURU (Doubt Solver)
# ==========================================
@app.post("/api/doubt-solver")
async def solve_doubt(chat: ChatMessage):
    try:
        genai.configure(api_key=os.environ.get("TUTOR_API_KEY"))
        
        system_instruction = """
        You are "Professor ICAI AI," a strict Chartered Accountant teaching CA courses. 
        Only answer queries related to the ICAI curriculum. Refuse all other topics.
        Base answers on SAs, AS, Ind AS, Income Tax Act, and Companies Act.
        """
        
        model = genai.GenerativeModel('gemini-2.5-flash', system_instruction=system_instruction)
        response = model.generate_content(contents=chat.message, generation_config={"temperature": 0.2})
        
        return {"response": response.text}
        
    except Exception as e:
        print(f"Guru Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
