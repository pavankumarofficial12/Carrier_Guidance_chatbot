import logging
import os
import json
import tempfile
from datetime import datetime
from typing import Optional, Tuple

import boto3
import pytz
import whisper
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, DateTime, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker, declarative_base

# Load environment variables at startup
load_dotenv()

# ========================= CONFIGURATION =========================
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Database Configuration
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    logger.critical("DATABASE_URL is not set in environment variables!")
    raise Exception("DATABASE_URL environment variable is required")

# AWS Bedrock Configuration
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION_NAME = os.getenv("AWS_REGION_NAME")
MODEL_ID = os.getenv("MODEL_ID")

if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
    logger.warning("AWS Bedrock credentials are missing!")

# Initialize FastAPI
app = FastAPI(
    title="AI Career Guidance Chatbot",
    description="Personalized AI Career Counselor using Llama 3 via AWS Bedrock",
    version="1.0.0"
)

# Database Setup
engine = create_engine(DATABASE_URL, pool_recycle=3600, echo=False)
Session = sessionmaker(bind=engine)
Base = declarative_base()

# ========================= DATABASE MODELS =========================
class Education(Base):
    __tablename__ = 'education'
    
    user_id = Column(Integer, primary_key=True)
    school_education_board = Column(String(100))
    school_year_of_passout = Column(Integer)
    intermediate_college_specialization = Column(String(100))
    intermediate_diploma = Column(String(100))
    intermediate_year_of_passout = Column(Integer)
    graduation_college_specialization = Column(String(100))
    graduation_year_of_passout = Column(Integer)
    post_graduate_college_specialization = Column(String(100))
    post_graduate_year_of_passout = Column(Integer)


class PersonalDetails(Base):
    __tablename__ = 'personal_details'
    
    id = Column(Integer, primary_key=True)
    skills = Column(String(500))
    interests = Column(String(500))


class AiCareerChatBot(Base):
    __tablename__ = 'ai_career_chat_bot'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    bot_response = Column(String(1000))
    created_at = Column(DateTime, default=datetime.utcnow)
    user_question = Column(String(500))
    user_id = Column(Integer)


# Create tables
Base.metadata.create_all(engine)

# ========================= AWS BEDROCK CLIENT =========================
bedrock = None
try:
    bedrock = boto3.client(
        service_name='bedrock-runtime',
        region_name=AWS_REGION_NAME,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    )
    logger.info("AWS Bedrock client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize AWS Bedrock client: {str(e)}")


# ========================= HELPER FUNCTIONS =========================
def get_user_details(user_id: int) -> Tuple:
    session = Session()
    try:
        education_query = text("SELECT * FROM education WHERE user_id = :user_id")
        personal_query = text("SELECT * FROM personal_details WHERE id = :user_id")
        
        education = session.execute(education_query, {'user_id': user_id}).fetchone()
        personal = session.execute(personal_query, {'user_id': user_id}).fetchone()
        
        if not education or not personal:
            raise HTTPException(status_code=404, detail="User profile not found or incomplete")
        
        return education, personal
    except SQLAlchemyError as e:
        logger.error(f"Database error: {str(e)}")
        raise HTTPException(status_code=500, detail="Database error occurred")
    finally:
        session.close()


def get_career_guidance(user_details):
    if not bedrock:
        raise HTTPException(status_code=503, detail="AI service is currently unavailable")

    education, personal = user_details
    current_level = "unknown"
     
    if education.post_graduate_year_of_passout:
        current_level = "post_graduate"
    elif education.graduation_year_of_passout:
        current_level = "graduate"
    elif education.intermediate_year_of_passout:
        current_level = "intermediate"
    elif education.school_year_of_passout:
        current_level = "school"

    prompt = f"""As an AI career counselor, provide personalized career guidance for a student with the following details:

User ID: {education.user_id}
Current Education Level: {current_level}
School Education: Board - {education.school_education_board}, Year of Passout - {education.school_year_of_passout}
Intermediate: Specialization - {education.intermediate_college_specialization}, Diploma - {education.intermediate_diploma}, Year of Passout - {education.intermediate_year_of_passout}
Graduation: Specialization - {education.graduation_college_specialization}, Year of Passout - {education.graduation_year_of_passout}
Post-Graduation: Specialization - {education.post_graduate_college_specialization}, Year of Passout - {education.post_graduate_year_of_passout}
Skills: {personal.skills}
Interests: {personal.interests}

Based on the student's current education level ({current_level}), provide tailored recommendations as follows:

1. If the user has completed school:
   - Recommend specific options for intermediate education or diploma courses that align with their interests and skills.
   - Explain the benefits and career prospects for each recommended option.

2. If the user has completed intermediate:
   - Recommend suitable B.Tech programs and related courses that build upon their intermediate specialization.
   - Explain how these align with their intermediate specialization, skills, and interests.
   - Suggest potential career paths that these programs could lead to.

3. If the user has completed graduation:
   - Recommend both specific job opportunities and options for higher studies related to their degree.
   - Provide information on entry-level positions that match their skills and interests.
   - Suggest relevant post-graduate programs and explain their benefits in terms of career advancement.

4. If the user has completed post-graduation or has details for all education levels:
   - Focus on recommending job opportunities that specifically align with their highest level of education, skills, and interests.
   - Suggest specific roles, industries, and companies that match their qualifications and interests.
   - Provide advice on how to leverage their educational background and skills in the job market.
   - Recommend any additional certifications or skills that could enhance their career prospects.

For all recommendations:
- Be specific and provide detailed information about each option, tailored to the user's unique profile.
- Consider the user's skills and interests when making recommendations.
- Offer practical advice on how to pursue these options.
- If any information is missing, note that explicitly."""

    body = json.dumps({
        "prompt": prompt,
        "max_gen_len": 300,
        "temperature": 0.5,
        "top_p": 0.95
    })

    try:
        response = bedrock.invoke_model(
            body=body,
            modelId=MODEL_ID,
            accept='application/json',
            contentType='application/json'
        )
        response_body = json.loads(response['body'].read().decode('utf-8'))
        return response_body.get('generation', "Sorry, I couldn't generate a response.")
    except Exception as e:
        logger.error(f"Bedrock error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate career guidance")


def handle_followup_question(question: str, previous_guidance: str, user_details):
    if not bedrock:
        raise HTTPException(status_code=503, detail="AI service unavailable")

    education, personal = user_details

    prompt = f"""Previous career guidance for user (ID: {education.user_id}):
{previous_guidance}

User's Educational Background:
- School: {education.school_education_board}, Year: {education.school_year_of_passout}
- Intermediate: {education.intermediate_college_specialization}, Year: {education.intermediate_year_of_passout}
- Graduation: {education.graduation_college_specialization}, Year: {education.graduation_year_of_passout}
- Post-Graduation: {education.post_graduate_college_specialization}, Year: {education.post_graduate_year_of_passout}

Skills: {personal.skills}
Interests: {personal.interests}

The user has asked the following follow-up question:
{question}

Please provide a response based on the following guidelines:

1. If the question is related to the career guidance output, provide detailed clarification.
2. If the question is about education or career, give comprehensive answer.
3. If unrelated, respond with: "I apologize, but this question is not related to education or career guidance."

Keep response professional and helpful."""

    body = json.dumps({
        "prompt": prompt,
        "max_gen_len": 200,
        "temperature": 0.5,
        "top_p": 0.95
    })

    try:
        response = bedrock.invoke_model(
            body=body,
            modelId=MODEL_ID,
            accept='application/json',
            contentType='application/json'
        )
        response_body = json.loads(response['body'].read().decode('utf-8'))
        return response_body.get('generation', "I couldn't process your follow-up question.")
    except Exception as e:
        logger.error(f"Follow-up error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to process follow-up question")


def process_transcription(transcription: str):
    if not bedrock:
        raise HTTPException(status_code=503, detail="AI service unavailable")

    prompt = f"""Please analyze and respond to the following transcribed text:

{transcription}

Guidelines:
1. Provide a comprehensive and helpful response.
2. Maintain professional and engaging tone.
3. Ask for clarification if needed."""

    body = json.dumps({
        "prompt": prompt,
        "max_gen_len": 300,
        "temperature": 0.5,
        "top_p": 0.95
    })

    try:
        response = bedrock.invoke_model(
            body=body,
            modelId=MODEL_ID,
            accept='application/json',
            contentType='application/json'
        )
        response_body = json.loads(response['body'].read().decode('utf-8'))
        return response_body.get('generation', "I couldn't process the transcription.")
    except Exception as e:
        logger.error(f"Transcription processing error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to process transcription")


def store_chat_data(user_id: int, bot_response: str, user_question: Optional[str] = None):
    session = Session()
    try:
        chat = AiCareerChatBot(
            user_id=user_id,
            bot_response=bot_response,
            user_question=user_question,
            created_at=datetime.utcnow()
        )
        session.add(chat)
        session.commit()
        logger.info(f"Chat data stored successfully for user {user_id}")
    except Exception as e:
        logger.error(f"Failed to store chat data: {str(e)}")
        session.rollback()
    finally:
        session.close()


# ========================= PYDANTIC MODELS =========================
class GuidanceRequest(BaseModel):
    user_id: int
    followup_question: Optional[str] = None


# ========================= API ENDPOINTS =========================
@app.post("/guidance")
async def guidance(request: GuidanceRequest):
    try:
        user_details = get_user_details(request.user_id)
        initial_guidance = get_career_guidance(user_details)

        if request.followup_question:
            followup_response = handle_followup_question(
                request.followup_question, initial_guidance, user_details
            )
            store_chat_data(request.user_id, followup_response, request.followup_question)
            return {
                "status": "success",
                "initial_guidance": initial_guidance,
                "followup_response": followup_response
            }
        else:
            store_chat_data(request.user_id, initial_guidance)
            return {
                "status": "success",
                "career_guidance": initial_guidance
            }

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Unexpected error in /guidance: {str(e)}")
        raise HTTPException(status_code=500, detail="An internal error occurred")


@app.post("/audio-to-text")
async def audio_to_text(file: UploadFile = File(...)):
    if not file.filename.endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only .wav files are supported")

    temp_file_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file.write(await file.read())
            temp_file_path = temp_file.name

        model = whisper.load_model("base")
        result = model.transcribe(temp_file_path)
        transcription = result['text']

        model_response = process_transcription(transcription)

        store_chat_data(user_id=0, bot_response=model_response, user_question=transcription)

        return {
            "status": "success",
            "transcription": transcription,
            "ai_response": model_response
        }

    except Exception as e:
        logger.error(f"Audio processing error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to process audio file")
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception as cleanup_error:
                logger.warning(f"Failed to delete temp file: {cleanup_error}")


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "AI Career Guidance Bot"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
