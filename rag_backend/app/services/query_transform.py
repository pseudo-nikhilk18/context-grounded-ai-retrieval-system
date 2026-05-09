"""
Query rewrite for conversational RAG
"""

from typing import List, Dict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate

from app.config import settings


def transform_query(chat_history: List[Dict], user_query: str) -> str:

    # No history → no rewrite
    if not chat_history:
        return user_query

    history_text = "\n".join(
        f"{msg['role']}: {msg['content']}"
        for msg in chat_history
    )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Rewrite the user's question into a standalone question using conversation history."
        ),
        (
            "human",
            "Conversation:\n{history}\n\nQuestion:\n{question}"
        )
    ])

    llm = ChatGoogleGenerativeAI(
        model=settings.GEMINI_LLM_MODEL,
        temperature=0,
        google_api_key=settings.GEMINI_API_KEY,
    )

    chain = prompt | llm

    response = chain.invoke({
        "history": history_text,
        "question": user_query
    })

    return response.content.strip()