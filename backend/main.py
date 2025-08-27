from huggingface_hub import InferenceClient
from PyPDF2 import PdfReader
import os
from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import dotenv
import json
dotenv.load_dotenv()

app = FastAPI()

# Allow Next.js frontend to call API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# load the pdf
def load_pdf(uploaded_file: UploadFile):
    reader = PdfReader(uploaded_file.file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def send_to_huggingface(resume_text, job_description):
    client = InferenceClient(token=os.getenv("HUGGING_FACE_TOKEN"))
    completion = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-V3-0324",
        messages=[
            {
                "role": "user",
                "content": f"""
                You are an AI Resume Analyzer. 
                Input: 
                1. Resume text: {resume_text}
                2. Job description text {job_description}

                Task:
                - Extract main skills from the resume
                - Extract required skills from the job description
                - Compare and calculate a match score from 0 to 100
                - List matched skills, missing skills, and recommendations for the candidate to improve
                Return output in JSON format:
                {{ 
                "match_score": int,
                "resume_skills": [..],
                "job_skills": [..],
                "matched_skills": [..],
                "missing_skills": [..],
                "recommendations": [..]
                }}
                    
                
                """
            }
        ],
    )
    return completion.choices[0].message.content


@app.post("/match")
async def match_skills(resume: UploadFile, job_description: str = Form(...)):
    resume_text = load_pdf(resume)
    output = send_to_huggingface(resume_text, job_description)
     # response_text contains ```json { ... } ```
    clean = output.replace("```json", "").replace("```", "").strip()
    parsed = json.loads(clean)
    # print ("Parsed JSON:", parsed)  # debug print
    return parsed   # return real JSON


