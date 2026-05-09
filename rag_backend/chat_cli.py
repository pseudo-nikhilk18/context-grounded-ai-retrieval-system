import requests

API_URL = "http://localhost:8000/query"
COLLECTION = "default"

chat_history = []

print("\n✅ RAG Chat Started (type 'exit' to quit)\n")

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        break

    payload = {
        "collection_name": COLLECTION,
        "query": user_input,
        "chat_history": chat_history,
    }

    response = requests.post(API_URL, json=payload)

    if response.status_code != 200:
        print("Error:", response.text)
        continue

    data = response.json()

    answer = data["answer"]

    print("\nAssistant:", answer, "\n")

    # store history
    chat_history.append({"role": "user", "content": user_input})
    chat_history.append({"role": "assistant", "content": answer})