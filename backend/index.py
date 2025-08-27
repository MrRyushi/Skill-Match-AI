from openai import OpenAI
from PyPDF2 import PdfReader
import os
from dotenv import load_dotenv

load_dotenv()

# load the pdf
def load_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def send_to_openai(text, api_key):
 
    client = OpenAI(
        api_key=api_key,
    )

    response = client.responses.create(
        model="gpt-5",
        reasoning={"effort": "medium"},
        instructions=f"""
            You are an AI Resume Analyzer. 
            Input: 
            1. Resume text
            2. Job description text

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
        """,
        input=text,
    )

    print(response.output_text)

# get the key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

resume_text = load_pdf("resume.pdf")
send_to_openai(resume_text, OPENAI_API_KEY)
