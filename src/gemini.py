import os
import streamlit as st
from dotenv import load_dotenv
from google import genai


load_dotenv()


api_key = st.secrets.get(
    "GEMINI_API_KEY",
    os.getenv("GEMINI_API_KEY")
)

class GeminiClient:

    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY was not found in the .env file."
            )

        self.client = genai.Client(api_key=api_key)

    def generate_answer(self, question, context):

        prompt = f"""
You are a helpful document question-answering assistant.

Answer the user's question using ONLY the information
provided in the context below.

If the answer cannot be found in the context,
clearly say that the uploaded documents do not contain
enough information to answer the question.

Do not invent facts.

USER QUESTION:
{question}

CONTEXT:
{context}

Provide a clear and concise answer.
"""

        response = self.client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=prompt,
        )

        return response.text