
from typing import List


def get_system_prompt(context: str) -> str:
    """
    Generate strict anti-hallucination system prompt.
    """

    prompt = (
        "You are a context-grounded AI assistant.\n\n"
        "CRITICAL RULES:\n"
        "1. Answer ONLY using the provided context.\n"
        "2. If the answer is not present, respond exactly:\n"
        '"I could not find the answer in the provided documents."\n'
        "3. Do not hallucinate or use outside knowledge.\n"
        "4. Provide concise factual answers.\n\n"
        "CONTEXT:\n"
        + context +
        "\n\nAnswer the user's question using ONLY the context above.\n"
        "Also provide a confidence level (Low/Medium/High)."
    )

    return prompt


def get_query_transform_prompt(chat_history: List[dict], current_query: str) -> str:
    """
    Generate prompt for query transformation.
    """

    history_text = ""
    if chat_history:
        history_text = "\n".join(
            f"{msg.get('role', 'user')}: {msg.get('content', '')}"
            for msg in chat_history[-5:]
        )

    return (
        "Rewrite the following question into a standalone question.\n\n"
        "Conversation History:\n"
        + history_text +
        "\n\nCurrent Question:\n"
        + current_query +
        "\n\nStandalone Question:"
    )