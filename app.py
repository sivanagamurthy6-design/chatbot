import json
from google import genai
from dotenv import load_dotenv
import os
import urllib.parse
import base64
import re
import string
from twilio.rest import Client as TwilioClient
import logging
import pinecone
import PyPDF2
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone


logger = logging.getLogger()
logger.setLevel(logging.INFO)

<<<<<<< HEAD
=======


>>>>>>> 0de72ec (third commit)

load_dotenv()


pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(os.environ["PINECONE_INDEX_NAME"])


# Load once (important for Lambda performance)
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def _extract_body(event):
    """Extract user input, phone number, and raw body from event."""
    # "----------------------------------------------------------------------"
    raw_body = event.get("body", "")
  
       
    if event.get("isBase64Encoded", False):
        try:
            raw_body = base64.b64decode(raw_body).decode("utf-8")
        except Exception:
            pass  # fallback to raw string
    params = urllib.parse.parse_qs(raw_body, keep_blank_values=True)
    body_text = params.get("Body", [""])[0]
    from_number = params.get("From", [""])[0]
    if not from_number.startswith("whatsapp:+"):
        from_number = from_number.replace("whatsapp:", "")
        from_number = "whatsapp:+" + from_number.lstrip("+").strip()

    print("Final WhatsApp TO number:", from_number)
    to_number = params.get("To", [""])[0]
    message_sid = params.get("MessageSid", [""])[0]
    return body_text, from_number, to_number, message_sid

def get_query_embedding(text: str):
    return embedding_model.encode(text).tolist()

def retrieve_context_from_pinecone(
    query: str,
    top_k: int = 3,
    score_threshold: float = 0.5
):
    query_vector = get_query_embedding(query)

    result = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True
    )

    contexts = []
    for match in result.get("matches", []):
        if match["score"] >= score_threshold:
            contexts.append(match["metadata"].get("text", ""))

    return contexts



def lambda_handler(event, contexts=None):
    try:
        # Get input from event or default test string
       
        
        raw_body, from_number, to_number, message_sid = _extract_body(event)

        contexts=retrieve_context_from_pinecone(raw_body)
        if contexts:
            context_text = "\n\n".join(contexts)
            prompt = f"""act as a expert in rag retrival and retrive answer using ONLY the context below.
            Context:
            {context_text}
            Question:
            {raw_body}
            Keep the answer short and clear.
            """
            print("Using retrieved context for prompt.")
        else:
            prompt = f"""Act as a good Data Scientist.Answer the following question clearly and briefly:
            {raw_body}
            """
            print("No context retrieved; using default prompt.")
        gemini_api_key = os.environ["GEMINI_API_KEY"]
        if not gemini_api_key:
            raise Exception("GEMINI_API_KEY environment variable not set")
        
        client = genai.Client(api_key=gemini_api_key)

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt)
        print("Response from Gemini:", response.text)
        
        
        TWILIO_ACCOUNT_SID = os.environ['TWILIO_ACCOUNT_SID']
        TWILIO_AUTH_TOKEN = os.environ['TWILIO_AUTH_TOKEN']
        TWILIO_WHATSAPP_FROM = os.environ['TWILIO_WHATSAPP_FROM']

        twilio_client = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    
        # sent_msg = twilio_client.messages.create(
        #     #body=reply.text,
        #     body=response.text,
        #     #body=sms_body,
        #     from_=TWILIO_WHATSAPP_FROM,
        #     to=from_number
        # )
        try:
            sent_msg = twilio_client.messages.create(
                body=response.text,from_=os.environ["TWILIO_WHATSAPP_FROM"],
                to=from_number
            )
            print("Twilio SID:", sent_msg.sid)
        except Exception as twilio_error:
            print("TWILIO ERROR:", twilio_error)
    
    
        return {
            "statusCode": 200,
            # "body": json.dumps({
            # "input": raw_body,
            # "output": response.text
            #})
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
    

