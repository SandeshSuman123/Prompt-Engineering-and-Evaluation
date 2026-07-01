# ===============================
# Prompt Evaluation Dataset Generator
# Using OpenRouter
# ===============================

import json
import os
from dotenv import load_dotenv
from openai import OpenAI

# -------------------------------
# Load API Key
# -------------------------------

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

MODEL = "meta-llama/llama-3.3-70b-instruct"


# -------------------------------
# Helper Functions
# -------------------------------

def add_user_message(messages, text):
    messages.append({
        "role": "user",
        "content": text
    })


def add_assistant_message(messages, text):
    messages.append({
        "role": "assistant",
        "content": text
    })


# -------------------------------
# Chat Function
# -------------------------------

def chat(messages, temperature=0.8, stop=None):

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=temperature,
        stop=stop
    )

    return response.choices[0].message.content


# -------------------------------
# Dataset Generation Prompt
# -------------------------------

PROMPT = """
Generate an evaluation dataset for testing prompts.

The prompts being tested generate:

1. Python code
2. JSON
3. Regular Expressions

Requirements:

- Return ONLY valid JSON.
- Return an array.
- Every object must contain exactly one key:

{
  "task": "description"
}

Generate 9 tasks.

Rules:

- 3 Python tasks
- 3 JSON tasks
- 3 Regex tasks

Focus on AWS-related tasks.

The tasks should be short.

Example:

[
  {
    "task": "Write a Python function that lists all S3 buckets."
  },
  {
    "task": "Generate JSON for an EventBridge rule that detects EC2 state changes."
  }
]
"""


# -------------------------------
# Generate Dataset
# -------------------------------

def generate_dataset():

    messages = []

    add_user_message(messages, PROMPT)

    # Assistant Prefilling
    add_assistant_message(
        messages,
        "```json\n"
    )

    response = chat(
        messages,
        temperature=1,
        stop=["```"]
    )

    return json.loads(response)


# -------------------------------
# Core Evaluation Pipeline
# -------------------------------

def run_prompt(test_case):
    """Merges the prompt template and test case input, then returns the model's output"""
    prompt = f"""
Please solve the following task:

{test_case["task"]}
"""

    messages = []
    add_user_message(messages, prompt)
    output = chat(messages)
    return output


def run_test_case(test_case):
    """Runs a single test case through run_prompt, then grades the result"""
    output = run_prompt(test_case)

    # TODO - Grading
    score = 10

    return {
        "output": output,
        "test_case": test_case,
        "score": score
    }


def run_eval(dataset):
    """Runs every test case in the dataset through run_test_case and collects results"""
    results = []

    for test_case in dataset:
        result = run_test_case(test_case)
        results.append(result)

    return results


# -------------------------------
# Main
# -------------------------------

def main():

    print("=" * 60)
    print("Generating Evaluation Dataset...")
    print("=" * 60)

    dataset = generate_dataset()

    print("\nGenerated Dataset:\n")
    print(json.dumps(dataset, indent=4))

    with open("dataset.json", "w") as f:
        json.dump(dataset, f, indent=4)

    print("\nDataset saved as dataset.json")

    print("\n" + "=" * 60)
    print("Running Evaluation...")
    print("=" * 60)

    with open("dataset.json", "r") as f:
        dataset = json.load(f)

    results = run_eval(dataset)

    print("\nEvaluation Results:\n")
    print(json.dumps(results, indent=2))

    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nResults saved as results.json")


if __name__ == "__main__":
    main()
