## Eval Protocol OpenAI RFT Quickstart

This repo is a **minimal example** of how to:

- **Write an Eval Protocol `@evaluation_test`** that operates on an `EvaluationRow`
- **Convert it into an OpenAI RFT Python grader** using `build_python_grader_from_evaluation_test`
- **Validate and run** that grader via the OpenAI `/fine_tuning/alpha/graders/*` HTTP APIs

The core idea is: you define your grading logic once as an Eval Protocol evaluation test, and then reuse that exact logic as a `{"type": "python", "source": ...}` grader in an OpenAI Reinforcement Fine-Tuning (RFT) job.

The main files are:

- `example_rapidfuzz.py`: defines an `@evaluation_test` called `rapidfuzz_eval` that uses `rapidfuzz.WRatio` to score similarity between a model’s answer and a reference answer.
- `test_openai_grader.py`: builds a Python grader from `rapidfuzz_eval`, validates it with `/graders/validate`, and runs it once with `/graders/run`.

## Installation

Create and activate a virtual environment (optional but recommended):

```bash
cd /Users/derekxu/Documents/code/openai-rft-quickstart
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install eval-protocol rapidfuzz
```

Set your OpenAI API key:

```bash
export OPENAI_API_KEY="sk-..."
```

## Running the Examples

From the project root (`/Users/derekxu/Documents/code/openai-rft-quickstart`) with your virtual environment activated:

1. **Run the Eval Protocol test with pytest**

This shows that `rapidfuzz_eval` works as a normal Eval Protocol `@evaluation_test`:

```bash
pytest example_rapidfuzz.py -vs
```

2. **Validate and run the Python grader against OpenAI’s `/graders/*` APIs**

This converts `rapidfuzz_eval` into a `type: "python"` grader spec and then validates/runs it via HTTP:

```bash
python test_openai_grader.py
```

You should see output similar to:

```json
validate response: {
  "grader": {
    "type": "python",
    "source": "def _ep_eval(row, **kwargs):\n    \"\"\"\n    Example @evaluation_test that scores a row using rapidfuzz.WRatio and\n    attaches an EvaluateResult.\n    \"\"\"\n    reference = row.ground_truth\n    assistant_msgs = [m for m in row.messages if m.role == 'assistant']\n    last_assistant_content = assistant_msgs[-1].content if assistant_msgs else ''\n    prediction = last_assistant_content if isinstance(last_assistant_content, str) else ''\n    from rapidfuzz import fuzz, utils\n    score = float(fuzz.WRatio(str(prediction), str(reference), processor=utils.default_process) / 100.0)\n    row.evaluation_result = EvaluateResult(score=score)\n    return row\n\n\nfrom typing import Any, Dict\nfrom types import SimpleNamespace\n\n\nclass EvaluationRow(SimpleNamespace):\n    \"\"\"Minimal duck-typed stand-in for an evaluation row.\n\n    Extend this with whatever attributes your eval logic uses.\n    \"\"\"\n    pass\n\n\nclass EvaluateResult(SimpleNamespace):\n    \"\"\"Simple stand-in for Eval Protocol's EvaluateResult.\n\n    This lets evaluation-style functions that construct EvaluateResult(score=...)\n    run inside the Python grader sandbox without importing eval_protocol.\n    \"\"\"\n\n    def __init__(self, score: float, **kwargs: Any) -> None:\n        super().__init__(score=score, **kwargs)\n\n\nclass Message(SimpleNamespace):\n    \"\"\"Duck-typed stand-in for eval_protocol.models.Message (role/content).\"\"\"\n    pass\n\n\ndef _build_row(sample: Dict[str, Any], item: Dict[str, Any]) -> EvaluationRow:\n    # Start from any item-provided messages (EP-style), defaulting to [].\n    raw_messages = item.get(\"messages\") or []\n    normalized_messages = []\n    for m in raw_messages:\n        if isinstance(m, dict):\n            normalized_messages.append(\n                Message(\n                    role=m.get(\"role\"),\n                    content=m.get(\"content\"),\n                )\n            )\n        else:\n            # Already Message-like; rely on duck typing (must have role/content)\n            normalized_messages.append(m)\n\n    reference = item.get(\"reference_answer\")\n    prediction = sample.get(\"output_text\")\n\n    # EP-style: ensure the model prediction is present as the last assistant message\n    if prediction is not None:\n        normalized_messages = list(normalized_messages)  # shallow copy\n        normalized_messages.append(Message(role=\"assistant\", content=prediction))\n\n    return EvaluationRow(\n        ground_truth=reference,\n        messages=normalized_messages,\n        item=item,\n        sample=sample,\n    )\n\n\ndef grade(sample: Dict[str, Any], item: Dict[str, Any]) -> float:\n    row = _build_row(sample, item)\n    result = _ep_eval(row=row)\n\n    # Try to normalize different result shapes into a float score\n    try:\n        from collections.abc import Mapping\n\n        if isinstance(result, (int, float)):\n            return float(result)\n\n        # EvaluateResult-like object with .score\n        if hasattr(result, \"score\"):\n            return float(result.score)\n\n        # EvaluationRow-like object with .evaluation_result.score\n        eval_res = getattr(result, \"evaluation_result\", None)\n        if eval_res is not None:\n            if isinstance(eval_res, Mapping):\n                if \"score\" in eval_res:\n                    return float(eval_res[\"score\"])\n            elif hasattr(eval_res, \"score\"):\n                return float(eval_res.score)\n\n        # Dict-like with score\n        if isinstance(result, Mapping) and \"score\" in result:\n            return float(result[\"score\"])\n    except Exception:\n        pass\n\n    return 0.0\n",
    "name": "grader-R5FhpA6BFQlo"
  }
}
run response: {
  "reward": 0.7555555555555555,
  "metadata": {
    "name": "grader-5XXSBZ9B1OJj",
    "type": "python",
    "errors": {
      "formula_parse_error": false,
      "sample_parse_error": false,
      "sample_parse_error_details": null,
      "truncated_observation_error": false,
      "unresponsive_reward_error": false,
      "invalid_variable_error": false,
      "invalid_variable_error_details": null,
      "other_error": false,
      "python_grader_server_error": false,
      "python_grader_server_error_type": null,
      "python_grader_runtime_error": false,
      "python_grader_runtime_error_details": null,
      "model_grader_server_error": false,
      "model_grader_refusal_error": false,
      "model_grader_refusal_error_details": null,
      "model_grader_parse_error": false,
      "model_grader_parse_error_details": null,
      "model_grader_exceeded_max_tokens_error": false,
      "model_grader_server_error_details": null,
      "endpoint_grader_internal_error": false,
      "endpoint_grader_internal_error_details": null,
      "endpoint_grader_server_error": false,
      "endpoint_grader_server_error_details": null,
      "endpoint_grader_safety_check_error": false
    },
    "execution_time": 6.831332206726074,
    "scores": {},
    "token_usage": null,
    "sampled_model_name": null
  },
  "sub_rewards": {},
  "model_grader_token_usage_per_model": {}
}
```

## How It Works

- **`example_rapidfuzz.py`**:
  - Defines a tiny inline demo dataset (`DEMO_ROWS`) with a user/assistant conversation and a `ground_truth`.
  - Implements `rapidfuzz_eval(row: EvaluationRow, **kwargs)` decorated with `@evaluation_test`, which:
    - Pulls the last assistant message from `row.messages`.
    - Compares it to `row.ground_truth` using `rapidfuzz.fuzz.WRatio`.
    - Writes the score into `row.evaluation_result = EvaluateResult(score=...)`.
  - Calls `build_python_grader_from_evaluation_test(rapidfuzz_eval)` to produce `RAPIDFUZZ_PYTHON_GRADER_SPEC` (a `{"type": "python", "source": "..."}` dict).

- **`test_openai_grader.py`**:
  - Imports `rapidfuzz_eval` and builds the grader with `build_python_grader_from_evaluation_test(rapidfuzz_eval)`.
  - **Validates** the grader via:
    - `POST https://api.openai.com/v1/fine_tuning/alpha/graders/validate`
  - **Runs** the grader once via:
    - `POST https://api.openai.com/v1/fine_tuning/alpha/graders/run`
  - Uses a simple payload with:
    - `item = {"reference_answer": "fuzzy wuzzy had no hair"}`
    - `model_sample = "fuzzy wuzzy was a bear"`

## Next Steps

Once you have a working Python grader from your Eval Protocol `@evaluation_test`, you can:

- Swap out the example logic in `rapidfuzz_eval` for your own evaluation logic (e.g., exact match, rubric-based scoring, or more complex metrics).
- Use the validated grader in an OpenAI **Reinforcement Fine-Tuning** job—see OpenAI’s docs on [preparing your dataset and creating a reinforcement fine-tuning job](https://platform.openai.com/docs/guides/reinforcement-fine-tuning#prepare-your-dataset) for full details.
