
import os
import time
import duckdb
import gradio as gr
from google import genai
from google.genai import types

API_KEY = os.environ.get('secretName')

if not API_KEY:
    raise ValueError("GEMINI_API_KEY missing!")

client = genai.Client(api_key=API_KEY)

DB_PATH = "student_mental_health.duckdb"

if not os.path.exists(DB_PATH):
    raise FileNotFoundError("Database file missing!")

con = duckdb.connect(DB_PATH, read_only=True)

total_students = con.execute("""
    SELECT COUNT(*) FROM fact_depression
""").fetchone()[0]

risk_summary = con.execute("""
    SELECT risk_level, COUNT(*) AS total
    FROM fact_depression
    GROUP BY risk_level
""").fetchdf()

project_context = f"""
Project: Student Mental Health Data Warehouse

Total Students: {total_students}

Risk Distribution:
{risk_summary.to_string(index=False)}

Dataset is synthetic.
Risk levels are not medical diagnoses.
"""

SYSTEM_INSTRUCTION = f"""
You are a helpful AI assistant.

Support Bangla, English and Banglish.
Reply in the same language as the user.
Answer general knowledge questions too.

Project information:
{project_context}

Never invent project statistics.
Keep answers clear and concise.
"""

MODEL_CHAIN = [
    "gemini-3.8-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.7-flash"
]

def chatbot(message, history):

    if not message:
        yield "প্রশ্ন লিখে পাঠাও।"
        return

    prompt = f"User question: {message}"

    for model in MODEL_CHAIN:

        for attempt in range(2):

            response_text = ""

            try:
                stream = client.models.generate_content_stream(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_INSTRUCTION,
                        temperature=0.4,
                        max_output_tokens=350
                    )
                )

                for chunk in stream:
                    if chunk.text:
                        response_text += chunk.text
                        yield response_text

                if response_text:
                    return

            except Exception as e:

                if response_text:
                    yield response_text
                    return

                error = str(e).upper()

                if not any(code in error for code in [
                    "503", "429", "500", "502", "504",
                    "UNAVAILABLE", "RESOURCE_EXHAUSTED"
                ]):
                    yield f"API Error: {str(e)}"
                    return

                if attempt == 0:
                    time.sleep(2)

    yield "সবগুলো model ব্যস্ত। কিছুক্ষণ পর চেষ্টা করো।"


demo = gr.ChatInterface(
    fn=chatbot,
    type="messages",
    title="Student Mental Health AI Chatbot",
    description="Ask questions in Bangla, English or Banglish."
)

demo.launch(server_name="0.0.0.0", server_port=7860)
