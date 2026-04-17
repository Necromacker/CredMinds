import os
import json
import base64
from typing import List
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
import google.generativeai as genai
import uvicorn

load_dotenv(override=True)

app = FastAPI()

# Configure Gemini
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    # Redacted print for debugging (only showing first 4 chars)
    print(f"DEBUG: Configuring Gemini with key starting with: {api_key[:4]}...")
    genai.configure(api_key=api_key)
else:
    print("WARNING: GEMINI_API_KEY not found in .env file!")

# ==================== BACKEND LOGIC ====================

class LLMWorker:
    @staticmethod
    async def parse_docs(files: List[UploadFile]):
        model = genai.GenerativeModel("gemini-flash-latest")
        prompt = """Analyze these documents for a loan application. 
        Extract data into this JSON format exactly:
        {
            "name": "", "age": 0, "gender": "Male", "marital": "Single", "dependents": 0,
            "city": "", "pan": "", "aadhaar": "",
            "emp_type": "Salaried", "employer": "", "monthly_salary": 0, "exp_total": 0,
            "cibil": 750, "existing_emi": 0, "bank_balance": 0,
            "loan_amount": 0, "loan_tenure": 36
        }
        Return ONLY the JSON."""
        
        contents = [prompt]
        for file in files:
            file_bytes = await file.read()
            contents.append({"mime_type": file.content_type, "data": file_bytes})
        
        try:
            response = model.generate_content(contents)
            text = response.text.strip()
            # Clean JSON
            json_str = text[text.find('{'):text.rfind('}')+1]
            return json.loads(json_str)
        except Exception as e:
            return {"error": str(e)}

    @staticmethod
    def analyze_profile(profile: dict):
        model = genai.GenerativeModel("gemini-flash-latest")
        prompt = f"""As a Senior Bank Credit Officer, perform a deep credit appraisal.
        Profile: {json.dumps(profile)}
        
        Analyze for:
        1. Financial stability and repayment capacity.
        2. Document consistency (Look for any naming or data disparities).
        3. Risk factors vs Strengths.

        Return ONLY a JSON object with these EXACT keys:
        {{
            "verdict": "Approved" or "Rejected",
            "reason": "Single concise sentence",
            "risk_score": 0-100,
            "strong_points": ["point 1", "point 2"],
            "weak_points": ["point 1", "point 2"],
            "disparities": ["mismatch found 1" or empty],
            "interest_rate": "percentage",
            "loan_amount": "formatted amount"
        }}"""
        
        try:
            response = model.generate_content(prompt)
            text = response.text.strip()
            json_str = text[text.find('{'):text.rfind('}')+1]
            data = json.loads(json_str)
            
            return {
                "verdict": data.get("verdict", "Rejected"),
                "reason": data.get("reason", "Analysis complete."),
                "risk_score": data.get("risk_score", 50),
                "strong_points": data.get("strong_points", ["Stable income"]),
                "weak_points": data.get("weak_points", ["High DTI"]),
                "disparities": data.get("disparities", []),
                "interest_rate": data.get("interest_rate", "10.5%"),
                "loan_amount": data.get("loan_amount", f"₹{profile.get('loan_amount', '0')}")
            }
        except Exception as e:
            return {"verdict": "Rejected", "reason": f"Analysis Error: {str(e)}", "risk_score": 0}

    @staticmethod
    def explain_reasoning(profile: dict, verdict: dict, question: str):
        model = genai.GenerativeModel("gemini-flash-latest")
        context = f"""
        Profile: {json.dumps(profile)}
        Verdict: {json.dumps(verdict)}
        Question: {question}

        Role: Senior CredMind Credit Officer.
        Constraint: Provide a professional banking assessment. Use NO markdown symbols (NO asterisks, NO bolding with symbols).
        Respond in 3-4 short, punchy lines.
        Tone: Formal, conservative, and data-driven.
        
        Format Example:
        Credit Strength: High monthly liquidity and stable employment history.
        Primary Risk: Debt-to-income ratio exceeds internal policy thresholds.
        Officer Recommendation: Increase down payment to 20% for approval.
        """
        try:
            response = model.generate_content(context)
            # Remove any lingering asterisks just in case
            clean_text = response.text.replace("*", "").strip()
            return {"response": clean_text}
        except Exception as e:
            return {"error": str(e)}

# ==================== ROUTES ====================

@app.get("/", response_class=HTMLResponse)
async def get_index():
    with open("index.html", "r") as f:
        return f.read()

@app.post("/parse")
async def parse(files: List[UploadFile] = File(...)):
    result = await LLMWorker.parse_docs(files)
    return result

@app.post("/analyze")
async def analyze(request: Request):
    data = await request.json()
    result = LLMWorker.analyze_profile(data)
    return result

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    # Expecting { "profile": {...}, "verdict": {...}, "question": "..." }
    result = LLMWorker.explain_reasoning(
        data.get("profile", {}), 
        data.get("verdict", {}), 
        data.get("question", "")
    )
    return result

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8503)