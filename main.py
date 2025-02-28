import torch
import requests
from openai import OpenAI
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

import config
from classifier import tokenizer

client = OpenAI(
    api_key=config.openai_api_key,
)

conversation_history = [
    {"role": "system",
     "content": "You are a helpful AI assistant who can answer general questions and solve math problems."}
]

model_path = "math_classifier_distilbert"

try:
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    classifier = DistilBertForSequenceClassification.from_pretrained(model_path)
    classifier.eval()
    print("Loaded Model Config:", classifier.config)
except Exception as e:
    print("Error loading model or tokenizer: ", e)
    exit(1)


def classify_text(text):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier.to(device)
    """Classifies a text as math (1) or non-math (0) using DistilBERT."""
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)

        inputs = {key: val.to(device) for key, val in inputs.items()}

        with torch.no_grad():
            outputs = classifier(**inputs)

        prediction = torch.argmax(outputs.logits, dim=1).item()

        return prediction == 1

    except Exception as e:
        print("Error during classification: ", e)
        return False


def query_wolfram_alpha(query):
    url = "https://api.wolframalpha.com/v2/query"
    params = {
        "input": query,
        "format": "plaintext",
        "output": "JSON",
        "appid": config.wolframalpha_app_id
    }

    response = requests.get(url, params=params)
    data = response.json()

    if "queryresult" in data and data["queryresult"]["success"]:
        relevant_pods = ["Result", "Exact result", "Decimal approximation"]
        results = []

        for pod in data["queryresult"]["pods"]:
            if pod["title"] in relevant_pods:
                text = pod["subpods"][0]["plaintext"] if "subpods" in pod else "N/A"
                results.append(f"{text}")

        return "\n".join(results) if results else "No useful information found."
    else:
        return "Wolfram Alpha could not process this query."


def query_gpt(user_input, mode="general", wolfram_answer=None):
    if mode == "math":
        prompt = f"""
        The user asked: "{user_input}"
        Wolfram Alpha returned: "{wolfram_answer}"

        Rephrase the result naturally and concisely, as if you were answering the user directly.
        If the answer is just a number or simple result, return it clearly.
        """
    else:
        prompt = user_input

    conversation_history.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=conversation_history,
        stream=False
    )
    bot_response = response.choices[0].message.content.strip()

    conversation_history.append({"role": "assistant", "content": bot_response})

    return bot_response


def chatbot():
    print("Chatbot is running! Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        is_math = classify_text(user_input)

        print(f"Detected as math? {is_math}")

        if is_math:
            wolfram_response = query_wolfram_alpha(user_input)

            # If it's a simple numeric answer, return it directly
            if wolfram_response.isdigit():
                response = f"The answer is {wolfram_response}."
                conversation_history.append({"role": "assistant", "content": response})
            else:
                response = query_gpt(user_input, mode="math", wolfram_answer=wolfram_response)
        else:
            response = query_gpt(user_input, mode="general")

        print(f"Bot: {response}\n")


if __name__ == "__main__":
    chatbot()
