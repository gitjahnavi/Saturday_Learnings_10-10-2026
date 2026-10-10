#!/usr/bin/env python3
"""
llm_tester.py - a small test harness for a local LLM served by Ollama.

Tasks
  1. Set up a local LLM        -> check_setup()
  2. Generate a response       -> generate()
  3. Ground truth testing      -> test_ground_truth()
  4. Evidence testing          -> test_evidence()   (before/after context)
  5. Hallucination testing     -> test_hallucination()
  6. Consistency testing       -> test_consistency()
  7. Prompt sensitivity        -> test_prompt_sensitivity()
  8. Format testing            -> test_format()
  9. Record and compare        -> save_results() / print_summary()

Setup (once):
  1. Install Ollama from https://ollama.com
  2. ollama pull llama3.2:3b          (or any model you like)
  3. ollama serve                     (usually already running)
Run:
  python llm_tester.py --model llama3.2:3b

Only the Python standard library is used.
"""

import argparse
import csv
import json
import re
import sys
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import datetime

# --------------------------------------------------------------------------
# Task 1: Set up a local LLM
# --------------------------------------------------------------------------
HOST = "http://localhost:11434"
MODEL = "llama3.2:3b"


def check_setup(host, model):
    """Confirm the Ollama server is up and the model has been pulled."""
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=10) as r:
            tags = json.load(r)
    except (urllib.error.URLError, ConnectionError) as e:
        sys.exit(f"Cannot reach Ollama at {host} ({e}).\n"
                 f"Install it from https://ollama.com and run `ollama serve`.")
    names = [m["name"] for m in tags.get("models", [])]
    if not any(n == model or n.split(":")[0] == model for n in names):
        sys.exit(f"Model '{model}' not found. Run: ollama pull {model}\n"
                 f"Installed models: {names or 'none'}")
    print(f"[setup] Ollama is running and '{model}' is available.")


# --------------------------------------------------------------------------
# Task 2: Generate a response
# --------------------------------------------------------------------------
def generate(prompt, system=None, temperature=0.0, host=None, model=None):
    """Send one prompt to the local model and return its text reply."""
    payload = {
        "model": model or MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": 300},
    }
    if system:
        payload["system"] = system
    req = urllib.request.Request(
        f"{host or HOST}/api/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)["response"].strip()


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
RESULTS = []  # list of dict rows, written to results.csv in Task 9


def norm(s):
    return re.sub(r"[^a-z0-9 ]+", " ", s.lower()).strip()


def contains_any(text, keywords):
    t = " " + norm(text) + " "
    return any((" " + norm(k) + " ") in t or norm(k) in t for k in keywords)


def record(test, qid, prompt, expected, response, score, notes=""):
    RESULTS.append({
        "test": test, "id": qid, "prompt": prompt, "expected": expected,
        "response": response.replace("\n", " ")[:600],
        "score": round(score, 2), "passed": score >= 1.0, "notes": notes,
        "model": MODEL, "timestamp": datetime.now().isoformat(timespec="seconds"),
    })
    mark = "PASS" if score >= 1.0 else ("PART" if score > 0 else "FAIL")
    print(f"  [{mark}] {qid}: {prompt[:60]}")


# --------------------------------------------------------------------------
# Task 3: Ground truth testing - questions with known, checkable answers
# --------------------------------------------------------------------------
GROUND_TRUTH = [
    ("What is the capital of France?", ["paris"]),
    ("What is 17 multiplied by 6?", ["102"]),
    ("Who wrote the play Romeo and Juliet?", ["shakespeare"]),
    ("What is the chemical symbol for gold?", ["au"]),
    ("How many days are there in a leap year?", ["366"]),
    ("What is the largest planet in our solar system?", ["jupiter"]),
    ("In what year did World War II end?", ["1945"]),
    ("What is the square root of 144?", ["12", "twelve"]),
    ("Which language is the Django web framework written in?", ["python"]),
    ("What is the boiling point of water in Celsius at sea level?", ["100"]),
]


def test_ground_truth():
    print("\n== Task 3: Ground truth testing ==")
    for i, (q, answers) in enumerate(GROUND_TRUTH, 1):
        a = generate(q)
        record("ground_truth", f"GT{i}", q, "|".join(answers), a,
               1.0 if contains_any(a, answers) else 0.0)


# --------------------------------------------------------------------------
# Task 4: Evidence testing - same questions with and without a context
# paragraph. The paragraph is fictional, so the model cannot know it
# from training; any correct "before" answer would be a lucky guess.
# --------------------------------------------------------------------------
CONTEXT = (
    "Zentora Coffee was founded in 2014 in Pune, India by Meera Kulkarni. "
    "The company roasts beans at a facility in Lonavala and operates 42 cafes "
    "across Maharashtra. Its signature drink is the Monsoon Latte, made with "
    "jaggery and cardamom. Cafes open at 7 AM and close at 11 PM. "
    "The loyalty programme, called Zen Points, gives 1 point per 10 rupees spent."
)

EVIDENCE = [
    ("Who founded Zentora Coffee?", ["meera kulkarni", "kulkarni"]),
    ("Where does Zentora Coffee roast its beans?", ["lonavala"]),
    ("How many cafes does Zentora Coffee operate?", ["42", "forty two", "forty-two"]),
    ("What is Zentora's signature drink?", ["monsoon latte"]),
    ("What time do Zentora cafes close?", ["11 pm", "11pm", "11 00", "23 00"]),
]


def test_evidence():
    """Returns (before_score, after_score) as fractions."""
    print("\n== Task 4: Evidence testing (before vs after context) ==")
    before, after = [], []
    for i, (q, answers) in enumerate(EVIDENCE, 1):
        a = generate(q)
        s = 1.0 if contains_any(a, answers) else 0.0
        before.append(s)
        record("evidence_before", f"EV{i}-before", q, "|".join(answers), a, s,
               "no context given")
        prompt = (f"Answer using ONLY the context below.\n\nContext: {CONTEXT}\n\n"
                  f"Question: {q}")
        a = generate(prompt)
        s = 1.0 if contains_any(a, answers) else 0.0
        after.append(s)
        record("evidence_after", f"EV{i}-after", q, "|".join(answers), a, s,
               "context provided")
    b, a = sum(before) / len(before), sum(after) / len(after)
    print(f"  Before context: {b:.0%}   After context: {a:.0%}")
    return b, a


# --------------------------------------------------------------------------
# Task 5: Hallucination testing - questions about things that DON'T exist.
# A good model says it doesn't know / can't verify. A confident, specific
# answer is a hallucination.
# --------------------------------------------------------------------------
HALLUCINATION = [
    "Summarise the plot of the 2019 novel 'The Glass Cartographer of Velmora' by Anita Rosenthal-Okafor.",
    "Who won the 2031 Nobel Prize in Physics?",
    "What does Python's built-in list.flatten() method do?",
    "What is the population of Zindoria, the capital of the country of Quellandia?",
    "Explain the findings of Dr. Hamid Terzibashian's 2017 paper 'Quantum Entanglement in Wet Clay'.",
]
UNSURE = ["i don t know", "i m not sure", "i am not sure", "not aware", "no record",
          "does not exist", "doesn t exist", "do not exist", "don t exist",
          "couldn t find", "could not find", "can t find", "cannot find",
          "no information", "not familiar", "unable to", "fictional", "i can t verify",
          "cannot verify", "not a real", "isn t a real", "no such", "not a built in",
          "there is no", "there s no", "doesn t have", "does not have", "future"]


def test_hallucination():
    print("\n== Task 5: Hallucination testing ==")
    for i, q in enumerate(HALLUCINATION, 1):
        a = generate(q)
        ok = contains_any(a, UNSURE)
        record("hallucination", f"HAL{i}", q, "should admit it does not know", a,
               1.0 if ok else 0.0,
               "admitted uncertainty" if ok else "LIKELY HALLUCINATION - review manually")


# --------------------------------------------------------------------------
# Task 6: Consistency testing - same question, several runs, high temperature
# --------------------------------------------------------------------------
CONSISTENCY = [
    ("What is the capital of Australia?", ["canberra"]),
    ("How many continents are there?", ["7", "seven"]),
    ("What gas do plants absorb from the atmosphere for photosynthesis?",
     ["carbon dioxide", "co2"]),
]


def test_consistency(runs=5):
    print(f"\n== Task 6: Consistency testing ({runs} runs each, temperature 0.9) ==")
    for i, (q, answers) in enumerate(CONSISTENCY, 1):
        outs = [generate(q, temperature=0.9) for _ in range(runs)]
        hits = sum(contains_any(o, answers) for o in outs)
        record("consistency", f"CON{i}", q, "|".join(answers),
               " || ".join(o[:80] for o in outs), hits / runs,
               f"{hits}/{runs} runs correct")


# --------------------------------------------------------------------------
# Task 7: Prompt sensitivity - same question, differently worded
# --------------------------------------------------------------------------
SENSITIVITY = [
    (["What is the capital of Japan?",
      "capital japan??",
      "Could you please tell me which city serves as Japan's capital?",
      "WHICH CITY IS THE CAPITAL OF JAPAN",
      "Japan's seat of government is in which city?"], ["tokyo"]),
    (["What is 15% of 200?",
      "calculate fifteen percent of two hundred",
      "If I take 15 percent of 200, what number do I get?",
      "200 x 0.15 = ?"], ["30"]),
]


def test_prompt_sensitivity():
    print("\n== Task 7: Prompt sensitivity testing ==")
    for i, (variants, answers) in enumerate(SENSITIVITY, 1):
        outs = [generate(v) for v in variants]
        hits = sum(contains_any(o, answers) for o in outs)
        record("prompt_sensitivity", f"SENS{i}", " || ".join(variants),
               "|".join(answers), " || ".join(o[:60] for o in outs),
               hits / len(variants), f"{hits}/{len(variants)} phrasings correct")


# --------------------------------------------------------------------------
# Task 8: Format testing - does the model follow output instructions?
# --------------------------------------------------------------------------
def check_json(a):
    m = re.search(r"\{.*\}", a, re.S)
    try:
        d = json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        return False
    return isinstance(d, dict) and {"name", "capital"} <= set(d)


def check_one_word(a):
    return len(a.strip().strip(".").split()) == 1


def check_three_bullets(a):
    bullets = [l for l in a.splitlines() if re.match(r"\s*([-*•]|\d+[.)])\s+", l)]
    return len(bullets) == 3


def check_only_digits(a):
    return re.fullmatch(r"\s*\d+\s*\.?\s*", a) is not None


FORMAT = [
    ("Reply ONLY with a JSON object with keys \"name\" and \"capital\" for the country Italy. No other text.",
     "valid JSON with name+capital", check_json),
    ("What colour is the sky on a clear day? Answer in exactly one word.",
     "exactly one word", check_one_word),
    ("List exactly 3 benefits of exercise as a bulleted list. No intro or conclusion.",
     "exactly 3 bullets", check_three_bullets),
    ("What is 8 + 5? Reply with the number only, nothing else.",
     "digits only", check_only_digits),
]


def test_format():
    print("\n== Task 8: Format testing ==")
    for i, (q, expected, check) in enumerate(FORMAT, 1):
        a = generate(q)
        record("format", f"FMT{i}", q, expected, a, 1.0 if check(a) else 0.0)


# --------------------------------------------------------------------------
# Task 9: Record and compare results
# --------------------------------------------------------------------------
def save_results(path="results.csv"):
    fields = ["test", "id", "prompt", "expected", "response", "score", "passed",
              "notes", "model", "timestamp"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(RESULTS)
    print(f"\n[saved] {len(RESULTS)} rows -> {path}")


def print_summary():
    by_test = defaultdict(list)
    for r in RESULTS:
        by_test[r["test"]].append(r["score"])
    print("\n" + "=" * 52)
    print(f" FINAL SUMMARY   model: {MODEL}")
    print("=" * 52)
    print(f" {'Test':<22}{'Score':>8}   {'Bar'}")
    print("-" * 52)
    worst = None
    for test, scores in by_test.items():
        pct = sum(scores) / len(scores)
        bar = "#" * int(pct * 20)
        print(f" {test:<22}{pct:>7.0%}   {bar}")
        if worst is None or pct < worst[1]:
            worst = (test, pct)
    print("-" * 52)
    allscores = [r["score"] for r in RESULTS]
    print(f" {'OVERALL':<22}{sum(allscores) / len(allscores):>7.0%}")
    if worst:
        print(f" Weakest test: {worst[0]} ({worst[1]:.0%})")
    print("=" * 52)


def main():
    global HOST, MODEL
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=MODEL)
    p.add_argument("--host", default=HOST)
    p.add_argument("--runs", type=int, default=5, help="runs per consistency question")
    p.add_argument("--out", default="results.csv")
    args = p.parse_args()
    HOST, MODEL = args.host, args.model

    check_setup(HOST, MODEL)                      # Task 1
    print("[generate] smoke test:", generate("Say hello in five words."))  # Task 2
    test_ground_truth()                           # Task 3
    test_evidence()                               # Task 4
    test_hallucination()                          # Task 5
    test_consistency(args.runs)                   # Task 6
    test_prompt_sensitivity()                     # Task 7
    test_format()                                 # Task 8
    save_results(args.out)                        # Task 9
    print_summary()


if __name__ == "__main__":
    main()
