from fastapi import FastAPI, File, UploadFile, Form
from PyPDF2 import PdfReader
from langchain_google_genai import GoogleGenerativeAI
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

# Initialize Gemini model with API key from .env
google_api_key = os.getenv('GOOGLE_API_KEY')
llm = GoogleGenerativeAI(api_key=google_api_key, temperature=0, model="gemini-pro")

def generate_resume_text(text: str) -> str:
    prompt = """From the uploaded file, I need to determine its type. 
    If it's a resume, generate the file as text and format it well with 
    the corresponding titles, if it's something else, return None."""

    response = llm.invoke(prompt + text)
    return response

def tailor_resume(text: str, query: str) -> str:
    resume_text = generate_resume_text(text)

    prompt = f"Given the following resume: \n {resume_text} \n and job description: \n {query} \n, your task is to generate a tailored resume that aligns closely with the job requirements and is optimized for Applicant Tracking Systems (ATS). The goal is to ensure that the resume effectively showcases the candidate's relevant skills and experiences, increasing the likelihood of passing through automated screening processes. Please consider the key skills and qualifications mentioned in the job description, and modify the resume accordingly to enhance its ATS compatibility."
    response = llm.invoke(prompt)
    return response

def calculate_similarity(resume_text: str, description: str) -> str:
    prompt = f"Calculate the similarity between the following resume text and job description, and provide me the exact similarity score. Provide me the similarity score out of 100 and just return an integer value in response. \n\nResume: {resume_text}\n\nJob Description: {description}"

    response = llm.invoke(prompt)
    return response.strip()

@app.post("/calculate_similarity/")
async def calculate_similarity_endpoint(pdf: UploadFile = File(...), job_description: str = Form(...)):
    # Read PDF content
    pdf_reader = PdfReader(pdf.file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    # Calculate similarity between resume text and job description
    resume_text = text
    query = job_description
    similarity_percentage = calculate_similarity(resume_text, query)

    return {
        "similarity": similarity_percentage
    }

@app.post("/tailor_resume/")
async def tailor_resume_endpoint(pdf: UploadFile = File(...), job_description: str = Form(...)):
    # Read PDF content
    pdf_reader = PdfReader(pdf.file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    # Tailor resume based on job description
    resume_text = text
    query = job_description
    tailored_resume = tailor_resume(resume_text, query)

    return {
        "tailored_resume": tailored_resume
    }
@app.post("/test_audio/")
async def test_audio_endpoint(audio: UploadFile = File(...)):
    return {"filename": audio.filename}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    print("Server running on http://localhost")
