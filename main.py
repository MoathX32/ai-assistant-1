import os
import io
import json
import logging
from fastapi import FastAPI, HTTPException, UploadFile, File, Form ,BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Optional
import fitz  # PyMuPDF
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
import google.generativeai as genai
import re
import requests  # Added to handle the API request for lessons
import shutil
from pathlib import Path
import datetime
import time


# Initialize FastAPI app
app = FastAPI()
# Configure logging
logging.basicConfig(level=logging.INFO)


def clear_sessions_task():
    global sessions
    sessions.clear()
    logging.info("All sessions have been auto-cleared.")

def schedule_daily_clear(background_tasks: BackgroundTasks):
    current_time = datetime.datetime.now()
    # Calculate time until midnight
    time_until_midnight = (
        datetime.datetime.combine(current_time + datetime.timedelta(days=1), datetime.time.min)
        - current_time
    ).total_seconds()

    # Schedule the clear task to run at midnight
    background_tasks.add_task(clear_sessions_task)
    logging.info(f"Scheduled session clear and folder deletion in {time_until_midnight} seconds.")



# Load environment variables
load_dotenv()
genai_api_key = os.getenv("GENAI_API_KEY")

# Configure GenAI
genai.configure(api_key=genai_api_key)

vector_stores = {}
document_store = []
# Global flag to enable or disable the quiz endpoint
quiz_enabled = True
# Global variable to store the response text for quiz generation
last_response_text = None

sessions = {}
# Function Definitions

def get_single_pdf_chunks(pdf_bytes, filename, text_splitter):
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Empty PDF content.")
    
    pdf_stream = io.BytesIO(pdf_bytes)
    pdf_reader = fitz.open(stream=pdf_stream, filetype="pdf")
    pdf_chunks = []

    for page_num in range(pdf_reader.page_count):
        page = pdf_reader[page_num]
        page_text = page.get_text()  # Extract text from each page
        if page_text:
            page_chunks = text_splitter.split_text(page_text)
            for chunk in page_chunks:
                document = Document(page_content=chunk, metadata={"page": page_num, "filename": filename})
                logging.info(f"Adding document chunk with metadata: {document.metadata}")
                pdf_chunks.append(document)
    pdf_reader.close()
    return pdf_chunks

def get_all_pdfs_chunks(pdf_docs_with_names):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=1999
    )

    all_chunks = []
    for pdf_bytes, filename in pdf_docs_with_names:
        pdf_chunks = get_single_pdf_chunks(pdf_bytes, filename, text_splitter)
        all_chunks.extend(pdf_chunks)
    return all_chunks

def read_files_from_folder(folder_path):
    pdf_docs_with_names = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            with open(os.path.join(folder_path, filename), "rb") as file:
                pdf_docs_with_names.append((file.read(), filename))
    return pdf_docs_with_names

def get_vector_store(documents):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    try:
        vectorstore = FAISS.from_documents(documents=documents, embedding=embeddings)
        return vectorstore
    except Exception as e:
        logging.warning("Issue with creating the vector store.")
        raise HTTPException(status_code=500, detail="Issue with creating the vector store.")

def process_lessons(folder_path):
    pdf_docs_with_names = read_files_from_folder(folder_path)
    if not pdf_docs_with_names or any(len(pdf) == 0 for pdf, _ in pdf_docs_with_names):
        raise HTTPException(status_code=400, detail="One or more PDF files are empty.")

    documents = get_all_pdfs_chunks(pdf_docs_with_names)
    pdf_vectorstore = get_vector_store(documents)

    vector_stores["pdf_vectorstore"] = pdf_vectorstore
    document_store.extend(documents)  # Store original documents

    return {"message": "PDF files processed successfully"}

class QueryRequest(BaseModel):
    query: str
    optional_param: Optional[str] = None


import re

def clean_text(text):
    # Remove Markdown formatting like **, *, and unnecessary newlines
    cleaned_text = re.sub(r'\*+', '', text)
    cleaned_text = re.sub(r'\n+', ' ', cleaned_text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text

def get_response(context, question, model):
    chat_session = model.start_chat(history=[])

    prompt_template = """
    أنت مساعد ذكي متخصص في المواد الدراسيه باللغة العربيه حسب محتوي النص. أجب عن السؤال التالي بناءً على النص المرجعي المتاح.
    من فضلك، قدم الإجابة في نص عادي بدون تنسيق خاص مثل العلامات النجمية (*) أو أسلوب Markdown.
    يمكنك توليد وتقديم أمثلة تدعم الإجابة، بشرط أن تبقى ضمن إطار الموضوع دون الخروج عنه.
    لخص الاجابة لتكون مفيدة وكافية لإجابة السؤال.


    واذا كان السؤال بطريقه خاطئه او غير مفهوم لا تقدم اي اجابة. بدلاً من ذلك، أجب بالعبارة "incorrect question".
    واذا كان السؤال خارج اطار "النص المرجعي" لا تقدم اي اجابة. بدلاً من ذلك، أجب بالعبارة "out of topic".


    النص المرجعي: {context}\n
    السؤال: {question}\n
    """

    try:
        response = chat_session.send_message(prompt_template.format(context=context, question=question))
        response_text = response.text.strip()

        # Check for specific error responses and raise appropriate flags
        if "out of topic" in response_text.lower():
            logging.info("Flagged: Out of topic")
            return "", "OUT_OF_TOPIC", None
        elif "incorrect question" in response_text.lower():
            logging.info("Flagged: Incorrect question")
            return "", "INCORRECT_QUESTION", None

        # If the response doesn't match any flag, return it as a valid response
        logging.info(f"AI response: {response_text}")
        return response_text, None, None

    except Exception as e:
        logging.warning(e)
        return "", "ERROR", None

def generate_response(query_request: QueryRequest):
    global quiz_enabled, last_response_text  # Update the quiz flag and response text globally
    
    if "pdf_vectorstore" not in vector_stores:
        raise HTTPException(status_code=400, detail="PDF files must be processed first.")

    pdf_vectorstore = vector_stores['pdf_vectorstore']
    
    relevant_content = pdf_vectorstore.similarity_search(query_request.query, k=100)
    
    context = " ".join([doc.page_content for doc in relevant_content])

    generation_config = {
        "temperature": 0.2,
        "top_p": 1,
        "top_k": 1,
        "max_output_tokens": 8000,
    }

    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config,
        system_instruction="أنت مساعد ذكي متخصص في تقديم إجابات دقيقة باللغة العربية."
    )
    
    response, flag, _ = get_response(context, query_request.query, model)
    
    # Disable quiz and provide detailed error messages based on the response flag
    if flag == "OUT_OF_TOPIC":
        quiz_enabled = False
        last_response_text = None
        raise HTTPException(status_code=400, detail="The question is out of topic. Quiz generation is disabled.")
    elif flag == "INCORRECT_QUESTION":
        quiz_enabled = False
        last_response_text = None
        raise HTTPException(status_code=400, detail="The question is incorrect. Quiz generation is disabled.")
    
    quiz_enabled = True  # Enable quiz endpoint if a valid response is received
    last_response_text = response  # Store the response for generating quiz
    return response

def generate_questions_from_response(num_questions, question_type, response_text):
    if not response_text:
        raise HTTPException(status_code=400, detail="No response generated. Please generate a response first.")

    if question_type == "MCQ":
        prompt_template = f"""
        أنت مساعد ذكي متخصص في اللغة العربية. قم بتوليد {num_questions} من أسئلة الاختيار من متعدد (MCQs) بناءً على الإجابة التالية.
        تأكد أن الأسئلة ذكية وتعتمد على التحليل، ويمكنك تقديم أمثلة لتوضيح المفاهيم.
        يجب أن يحتوي كل سؤال على 4 خيارات وإجابة صحيحة، ويجب أن تكون الإجابة بصيغة JSON مع الحقول 'question' و 'choices' و 'correct answer'.

        الإجابة: {response_text}
        """
    else:
        prompt_template = f"""
        أنت مساعد ذكي متخصص في اللغة العربية. قم بتوليد {num_questions} من أسئلة صح/خطأ بناءً على الإجابة التالية.
        تأكد أن الأسئلة ذكية وتساهم في اختبار فهم الطالب، ويمكنك تقديم أمثلة إذا لزم الأمر.
        يجب أن تكون الإجابة بصيغة JSON مع الحقول 'question' و 'choices' و 'correct answer'.

        الإجابة: {response_text}
        """

    try:
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config={"temperature": 0.2, "top_p": 1, "top_k": 1, "max_output_tokens": 8000}
        )
        response = model.start_chat(history=[]).send_message(prompt_template)
        response_text = response.text.strip()

        logging.info(f"AI Model Response for questions: {response_text}")

        if response_text:
            response_json = clean_json_response(response_text)
            if response_json:
                return response_json
            else:
                logging.error("Response was not in JSON format.")
                raise HTTPException(status_code=500, detail="Response was not in JSON format.")
        else:
            raise HTTPException(status_code=500, detail="No response received from the model.")
    except Exception as e:
        logging.warning(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate questions.")

def clean_json_response(response_text):
    try:
        response_json = json.loads(response_text)
        return response_json
    except json.JSONDecodeError:
        try:
            cleaned_text = re.sub(r'```json', '', response_text).strip()
            cleaned_text = re.sub(r'```', '', cleaned_text).strip()
            cleaned_text = re.sub(r'\"([^\"]+?)\"', r'"\1"', cleaned_text)
            
            match = re.search(r'(\{.*\}|\[.*\])', cleaned_text, re.DOTALL)
            if match:
                cleaned_text = match.group(0)
                response_json = json.loads(cleaned_text)
                return response_json
            else:
                logging.error("No JSON object found in the response")
                return None
        except (ValueError, json.JSONDecodeError) as e:
            logging.error(f"Response is not valid JSON: {str(e)}")
            return None




def get_new_token():
    login_url = 'https://eduai.vitaparapharma.com/api/v1/auth/login'
    login_payload = {
        'username': 'Superadmin',
        'password': 'Password$404'
    }
    response = requests.post(login_url, json=login_payload)
    
    if response.status_code in [200, 202]:
        token = response.json().get('data')
        if token:
            logging.info(f"Token retrieved successfully: {token[:10]}...")
            return token
        else:
            logging.error("Token not found in the response.")
            return None
    else:
        logging.error(f"Login failed with status code: {response.status_code}")
        return None


# Base directory for storage
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@app.on_event("startup")
async def startup_event():
    schedule_daily_clear(BackgroundTasks())

# Endpoint to load lessons based on courseId
@app.post("/load_path/")
async def load_path(courseId: str = Form(...), studentId: str = Form(...)):
    # Generate a unique session ID
    session_id = f"{studentId}_{courseId}"

    # Define the folder path for storing the course lessons
    course_folder_path = os.path.join(BASE_DIR, "subject", courseId)

    # Check if the course folder already exists
    if not os.path.exists(course_folder_path):
        os.makedirs(course_folder_path, exist_ok=True)

        # Retrieve token
        token = get_new_token()
        if not token:
            raise HTTPException(status_code=500, detail="Failed to retrieve access token.")

        # Fetch lesson URLs using courseId
        api_url = f"https://eduai.vitaparapharma.com/api/v1/management/lesson/files?courseId={courseId}"
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.get(api_url, headers=headers)

        if response.status_code != 200:
            raise HTTPException(status_code=500, detail="Failed to retrieve lessons from the API.")

        lesson_data = response.json()
        if not lesson_data.get("success"):
            raise HTTPException(status_code=500, detail=lesson_data.get("message", "Error retrieving lessons."))

        # Download lesson files if the course folder does not exist
        lesson_urls = lesson_data.get("data", [])
        for i, url in enumerate(lesson_urls, start=1):
            lesson_filename = f"lesson_{i}.pdf"
            file_path = os.path.join(course_folder_path, lesson_filename)

            # Download the lesson file if it does not exist
            if not Path(file_path).exists():
                try:
                    lesson_response = requests.get(url)
                    with open(file_path, "wb") as file:
                        file.write(lesson_response.content)
                    logging.info(f"Downloaded: {lesson_filename}")
                except Exception as e:
                    raise HTTPException(status_code=500, detail="Error downloading lesson files.")
            else:
                logging.info(f"File already exists, skipping download: {lesson_filename}")

    # Check if the vector store for the course is already created
    if courseId not in vector_stores:
        # Process lessons and store the vector store for the course
        process_lessons(course_folder_path)
        vector_store = vector_stores["pdf_vectorstore"]
        vector_stores[courseId] = vector_store

    # Use the existing vector store for the session
    sessions[session_id] = {"vector_store": vector_stores[courseId], "last_response_text": None}

    return {"message": "PDF files processed successfully", "session_id": session_id}

@app.post("/query/")
async def query(query_request: str = Form(...), courseId: str = Form(...), studentId: str = Form(...)):
    session_id = f"{studentId}_{courseId}"

    if session_id not in sessions or "vector_store" not in sessions[session_id]:
        raise HTTPException(status_code=400, detail="PDF files must be processed first.")

    pdf_vectorstore = sessions[session_id]["vector_store"]

    # Parse the JSON string manually
    try:
        query_data = json.loads(query_request)
        query = query_data.get("query", None)
        optional_param = query_data.get("optional_param", None)

        if not query:
            raise HTTPException(status_code=400, detail="Query field is required.")
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format for query_request.")

    # Perform similarity search
    relevant_content = pdf_vectorstore.similarity_search(query, k=100)
    context = " ".join([doc.page_content for doc in relevant_content])

    generation_config = {
        "temperature": 0.2,
        "top_p": 1,
        "top_k": 1,
        "max_output_tokens": 8000,
    }

    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config,
        system_instruction="أنت مساعد ذكي متخصص في تقديم إجابات دقيقة باللغة العربية."
    )

    response, flag, _ = get_response(context, query, model)

    if flag in ["OUT_OF_TOPIC", "INCORRECT_QUESTION"]:
        sessions[session_id]["last_response_text"] = None
        raise HTTPException(status_code=400, detail="The question is invalid or out of topic.")

    sessions[session_id]["last_response_text"] = response
    return {"response": response}

@app.post("/quiz/")
async def quiz(courseId: str = Form(...), studentId: str = Form(...), question_type: str = Form(...), num_questions: int = Form(...)):
    session_id = f"{studentId}_{courseId}"

    if session_id not in sessions or not sessions[session_id]["last_response_text"]:
        raise HTTPException(status_code=400, detail="Quiz generation is disabled because no valid response was generated.")

    response_text = sessions[session_id]["last_response_text"]
    questions = generate_questions_from_response(num_questions, question_type, response_text)
    return {"questions": questions}

@app.post("/clear_sessions/")
async def clear_sessions():
    clear_sessions_task()
    return {"message": "All sessions and folders have been cleared successfully."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)