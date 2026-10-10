# Local LLM Tester

Model: llama3.2:latest (via Ollama)

## How to run
ollama pull llama3.2
pip install requests
python llm_tester.py --model llama3.2:latest

Outputs results.csv (every prompt, response and pass/fail) and a score summary in the terminal.

## Scores

| Test | Score |
|------|-------|
| Ground truth | 100% |
| Evidence (before context) | 0% |
| Evidence (after context) | 100% |
| Hallucination | 80% |
| Consistency | 100% |
| Prompt sensitivity | 100% |
| Format | 100% |
| Overall | 82% |

Screenshot: screenshot.png

## Answers

### 1. Which test did the model perform worst on, and why?
Evidence testing before context (0%). The questions were about a fictional company
the model could not know about, so it had nothing to answer from. Hallucination (80%)
is the more important weakness, because the model gave a confident wrong answer
instead of saying it didn't know.

### 2. Did the context paragraph (Task 4) improve accuracy?
Yes. Before: 0%. After: 100%. The paragraph gave the model the facts to answer from
instead of guessing.

### 3. One question where the model hallucinated
- Question: WRITE_HERE
- What it made up: WRITE_HERE
- How to prevent it: Tell the model to say "I don't know" when unsure, give it trusted
  text to answer from, lower the temperature, and have a human check important answers.

### 4. Would I trust this model to answer customer questions without a human check?
No. It did well on facts, format and consistency, but it hallucinated on at least one
question (80% on hallucination testing). A customer could get a confident but wrong
answer, so a human should check it.