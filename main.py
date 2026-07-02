# ===============================
# Prompt Evaluation Pipeline
# Using OpenRouter
# ===============================

import ast
import json
import os
import re

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
- Every object must contain exactly two keys:

{
  "task": "description",
  "type": "python" | "json" | "regex"
}

Generate 9 tasks.

Rules:

- 3 Python tasks (type = "python")
- 3 JSON tasks (type = "json")
- 3 Regex tasks (type = "regex")

Focus on AWS-related tasks.

The tasks should be short.

Example:

[
  {
    "task": "Write a Python function that lists all S3 buckets.",
    "type": "python"
  },
  {
    "task": "Generate JSON for an EventBridge rule that detects EC2 state changes.",
    "type": "json"
  }
]
"""


def generate_dataset():
    messages = []
    add_user_message(messages, PROMPT)
    add_assistant_message(messages, "```json\n")

    response = chat(messages, temperature=1, stop=["```"])
    return json.loads(response)


# -------------------------------
# run_prompt: the "worker"
# -------------------------------

def run_prompt(test_case):
    """Merges the prompt template and test case input, then returns the model's raw output"""
    prompt = f"""
Please solve the following task. Return ONLY the raw code/output, with no
explanation and no markdown formatting.

{test_case["task"]}
"""

    messages = []
    add_user_message(messages, prompt)

    # Prefill so the model starts generating code directly,
    # without needing to know ahead of time if it's python/json/regex
    add_assistant_message(messages, "```code")

    output = chat(messages, stop=["```"])
    return output


# -------------------------------
# Code Grader
# -------------------------------

def extract_code(text):
    """Strips leftover markdown fences / language tags and whitespace"""
    text = text.strip()
    text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
    text = re.sub(r"```$", "", text)
    return text.strip()


def validate_json(text):
    try:
        json.loads(text.strip())
        return 10
    except json.JSONDecodeError:
        return 0


def validate_python(text):
    try:
        ast.parse(text.strip())
        return 10
    except SyntaxError:
        return 0


def validate_regex(text):
    try:
        re.compile(text.strip())
        return 10
    except re.error:
        return 0


def grade_syntax(output, test_case):
    """Routes to the right validator based on the test case's declared type"""
    code = extract_code(output)
    task_type = test_case.get("type", "").lower()

    if task_type == "json":
        return validate_json(code)
    elif task_type == "python":
        return validate_python(code)
    elif task_type == "regex":
        return validate_regex(code)
    else:
        # Unknown type - can't syntax-check it, so don't penalize
        return 10


# -------------------------------
# Model Grader (LLM-as-judge)
# -------------------------------

def grade_by_model(test_case, output):
    """Asks the model to act as a judge and score the output 0-10"""
    grading_prompt = f"""
You are an expert code reviewer grading an AI model's response to a task.

Task given to the model:
{test_case["task"]}

Model's response:
{output}

Grade the response from 0 to 10 based on:
- Correctness (does it actually solve the task?)
- Relevance (does it stay focused on AWS / the specific ask?)
- Completeness (is anything important missing?)

Respond with ONLY valid JSON in this exact format, nothing else:
{{
  "reasoning": "one or two sentence explanation",
  "score": <integer 0-10>
}}
"""

    messages = []
    add_user_message(messages, grading_prompt)
    add_assistant_message(messages, "```json\n")

    response = chat(messages, temperature=0, stop=["```"])

    try:
        return json.loads(response)
    except json.JSONDecodeError:
        # Fallback if the judge model misbehaves
        return {"reasoning": "Failed to parse grader output", "score": 0}


# -------------------------------
# run_test_case: combines both graders
# -------------------------------

def run_test_case(test_case):
    """Runs a single test case, then grades it with both graders and averages the scores"""
    output = run_prompt(test_case)

    model_grade = grade_by_model(test_case, output)
    model_score = model_grade["score"]

    syntax_score = grade_syntax(output, test_case)

    score = (model_score + syntax_score) / 2

    return {
        "output": output,
        "test_case": test_case,
        "model_score": model_score,
        "model_reasoning": model_grade.get("reasoning", ""),
        "syntax_score": syntax_score,
        "score": score
    }


# -------------------------------
# run_eval: the orchestrator
# -------------------------------

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

    avg_score = sum(r["score"] for r in results) / len(results)
    print(f"\nAverage Score: {avg_score:.2f}/10")


if __name__ == "__main__":
    main()
