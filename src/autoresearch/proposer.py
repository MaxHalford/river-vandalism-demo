"""Build proposer context and call OpenAI to get candidate code."""

from __future__ import annotations

import json

from openai import OpenAI


SYSTEM_PROMPT = """You are an ML researcher iterating on a Wikipedia revert prediction model. The
system uses River for online learning. Propose ONE replacement that you believe will improve rolling
ROC-AUC over the next 48 hours of live edits.

Output ONLY Python code conforming to this contract. No prose, no markdown fences:

# hypothesis: <one-line explanation of what you are testing>
def make_model():
    \"\"\"Return a River pipeline exposing predict_proba_one(x) and learn_one(x, y).\"\"\"
    ...

# Optional. If omitted, the live featurize is used.
def featurize(event: dict, state: dict) -> dict:
    ...

Constraints:
- Allowed imports: river, numpy, math, collections, datetime, statistics, itertools, typing,
  dataclasses, re, functools, __future__.
- No filesystem, network, eval, exec, open, __import__.
- Must complete a 48h replay in under 90 seconds wall clock.
- Model output should be probability of class True (the edit will be reverted).
"""


def build_user_prompt(
    incumbent_code: str,
    feature_names: list[str],
    metrics: dict,
    hard_examples: list[dict],
    recent_proposals: list[dict],
) -> str:
    return (
        "## Incumbent model\n\n```python\n"
        + incumbent_code
        + "\n```\n\n## Available features\n\n"
        + ", ".join(feature_names)
        + "\n\n## Recent live metrics (rolling 2k labeled edits)\n\n"
        + json.dumps(metrics, indent=2)
        + "\n\n## Hard examples\n\n"
        + json.dumps(hard_examples, indent=2, default=str)
        + "\n\n## Your recent proposals and their outcomes (avoid repeating)\n\n"
        + json.dumps(recent_proposals, indent=2, default=str)
        + "\n\n## Task\n\nPropose ONE candidate. Include a `# hypothesis:` comment at the top.\n"
    )


def propose(client: OpenAI, model: str, user_prompt: str) -> tuple[str, str]:
    """Returns (hypothesis, code)."""
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.7,
    )
    code = (resp.choices[0].message.content or "").strip()
    if code.startswith("```"):
        # Strip a single code-fence wrapper if the model couldn't resist
        lines = code.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        code = "\n".join(lines)
    hypothesis = ""
    for line in code.splitlines():
        s = line.strip()
        if s.startswith("# hypothesis:"):
            hypothesis = s.split(":", 1)[1].strip()
            break
    return hypothesis, code
