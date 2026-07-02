# Prompt Evaluation Pipeline

A lightweight framework for systematically evaluating LLM prompt quality — instead of eyeballing a few outputs and calling it good, this pipeline generates a test dataset, runs prompts against a model, and scores every output using **two independent graders**: a fast deterministic code grader and an LLM-as-judge model grader.

Built with **Python**, **OpenRouter**, and **Llama 3.3 70B Instruct**.

---

## Why this exists

Prompt engineering without evaluation is guesswork. You tweak a prompt, glance at 2–3 outputs, and ship it — until an edge case silently breaks three weeks later. This pipeline treats prompts the way you'd treat code: with a repeatable test suite and a score you can track over time.

---

## How it works

```
generate_dataset()          →  creates N labeled test cases (task + type)
        │
        ▼
run_eval(dataset)            →  orchestrates the full run
        │
        ▼
run_test_case(test_case)     →  runs + grades ONE test case
        │
        ├──► run_prompt(test_case)     →  sends task to the model, gets raw output
        │
        ├──► grade_syntax(output)      →  deterministic code grader (0 or 10)
        │
        └──► grade_by_model(output)    →  LLM-as-judge grader (0–10 + reasoning)
        │
        ▼
   final score = average(syntax_score, model_score)
```

### The two-grader design

| Grader | Type | Checks | Cost |
|---|---|---|---|
| **Code grader** (`grade_syntax`) | Deterministic | Is the output *well-formed*? (valid Python/JSON/regex via `ast.parse`, `json.loads`, `re.compile`) | Free, instant |
| **Model grader** (`grade_by_model`) | LLM-as-judge | Is the output *actually correct and useful* for the task? | API call, slower |

Neither grader alone is trustworthy — syntactically valid code can still be useless, and an LLM judge can be fooled or inconsistent. Averaging both means an output has to be **objectively valid AND substantively good** to score well.

---

## Project structure

```
.
├── main.py              # Full pipeline: dataset generation, run, grade, save results
├── dataset.json          # Generated test cases (task + type), created on first run
├── results.json          # Per-test-case outputs, scores, and judge reasoning
├── .env.example           # Template for required environment variables
├── requirements.txt       # Python dependencies
└── README.md
```

---

## Setup

### 1. Clone and install dependencies

```bash
git clone <your-repo-url>
cd prompt-evaluation-pipeline
pip install -r requirements.txt
```

### 2. Configure your API key

Copy `.env.example` to `.env` and add your [OpenRouter](https://openrouter.ai/) API key:

```bash
cp .env.example .env
```

```
OPENROUTER_API_KEY=your_key_here
```

### 3. Run the pipeline

```bash
python main.py
```

This will:
1. Generate a 9-task evaluation dataset (3 Python / 3 JSON / 3 Regex, AWS-themed) → `dataset.json`
2. Run each task through the model → collect raw outputs
3. Grade every output with both graders
4. Save full results with scores and reasoning → `results.json`
5. Print the average score across the dataset

---

## Example output (`results.json` excerpt)

```json
{
  "output": "def list_s3_buckets():\n    import boto3\n    s3 = boto3.client('s3')\n    return [b['Name'] for b in s3.list_buckets()['Buckets']]",
  "test_case": {
    "task": "Write a Python function that lists all S3 buckets.",
    "type": "python"
  },
  "model_score": 9,
  "model_reasoning": "Correctly uses boto3 to list bucket names; minor: no error handling for missing credentials.",
  "syntax_score": 10,
  "score": 9.5
}
```

---

## Requirements

```
openai
python-dotenv
```

(`openai` SDK is used with `base_url` pointed at OpenRouter, not OpenAI directly.)

---

## Roadmap / possible extensions

- [ ] Parallelize `run_eval` with `concurrent.futures` to speed up multi-task runs
- [ ] Add a weighted scoring option (e.g. `0.3 * syntax + 0.7 * model`)
- [ ] Track scores across prompt versions to catch regressions over time
- [ ] Add a CLI flag to swap models/providers
- [ ] Add cost/token tracking per run
- [ ] Export results as a simple HTML/Markdown report

---

## License

MIT
