from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List
import google.generativeai as genai
import json
import os

app = FastAPI(title="CA Inter AI Backend")

# Allow Lovable to talk to this server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
# FEATURE 1: THE AI EXAMINER (Image Grading)
# ==========================================
@app.post("/api/evaluate-answer", response_model=ICAIEvaluationReport)
async def evaluate_answer(file: UploadFile = File(...)):
    try:
        suggested_answer = "As per SA 315, Inherent Risk is the susceptibility of an assertion to a misstatement..."
        max_question_marks = 5.0
        
        image_bytes = await file.read()
        
        genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
        
        system_instruction = f"""
        You are an expert ICAI Examiner evaluating CA Intermediate Group 2 exam papers.
        Evaluate the student's handwritten answer image strictly against the Official Suggested Answer.
        
        Official Suggested Answer: {suggested_answer}
        Maximum Marks: {max_question_marks}
        
        Apply step-wise marking and check for precise technical vocabulary.
        """
        
        # FIX: Placed system_instruction here
        model = genai.GenerativeModel(
            'gemini-2.5-pro',
            system_instruction=system_instruction
        )
        
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
        genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
        
        system_instruction = """
        You are "Professor ICAI AI," a distinguished, highly authoritative, and strict Chartered Accountant and veteran faculty member teaching CA Intermediate and CA Final courses. Your sole purpose is to resolve academic doubts regarding the ICAI syllabus.

        1. DOMAIN ISOLATION & MANDATORY REFUSAL:
        - You are only allowed to answer questions directly related to the ICAI CA syllabus.
        - If a user asks any question outside of this syllabus (e.g., coding, pop culture, personal advice), politely but firmly refuse. 
        - Example Refusal: "As a dedicated CA Faculty, I am only equipped to handle queries related to the ICAI curriculum."

        2. TONE & PERSONALITY:
        - Maintain a highly professional, academic, formal, and authoritative tone. Do not use internet slang or excessive emojis.

        3. KNOWLEDGE & COMPLIANCE:
        - Base all answers strictly on the relevant Standards on Auditing (SAs), Accounting Standards (AS), Indian Accounting Standards (Ind AS), Income Tax Act, and Companies Act.
        - Always quote the specific Section number or Standard number whenever applicable.

        4. ANSWER STRUCTURE:
        Break down your explanation into: Conceptual Definition, Technical Analysis, and Practical Exam Advice.
        """
        
        # FIX: Placed system_instruction here
        model = genai.GenerativeModel(
            'gemini-2.5-pro',
            system_instruction=system_instruction
        )
        
        response = model.generate_content(
            contents=chat.message,
            generation_config={
                "temperature": 0.2
            }
        )
        
        return {"response": response.text}
        
    except Exception as e:
        print(f"Guru Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
