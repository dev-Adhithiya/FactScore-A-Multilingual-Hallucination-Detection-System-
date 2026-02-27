"""
tests/test_poly_fever.py - Evaluate hallucination detection using local Poly-FEVER TSV.

Columns: ID, Label, Topic Distribution, en, zh-CN, hi, ar, bn, ja, ko, ta, th, ka, am
Label values: "true" (supported) or "false" (refuted)

Usage:
    python tests/test_poly_fever.py --language en --max_samples 50
    python tests/test_poly_fever.py --language hi --max_samples 50
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv
import json
import argparse
import requests

MODEL_SERVER  = "http://127.0.0.1:8001"
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Auto-detect TSV path
TSV_PATH = os.path.join(_PROJECT_ROOT, "Poly-FEVER", "Poly-FEVER.tsv")
if not os.path.exists(TSV_PATH):
    TSV_PATH = "F:/Project/Poly-FEVER/Poly-FEVER.tsv"

LANGUAGE_MAP = {
    "en":  ("en",    "English"),
    "hi":  ("hi",    "Hindi"),
    "ta":  ("ta",    "Tamil"),
    "bn":  ("bn",    "Bengali"),
    "zh":  ("zh-CN", "English"),
    "ar":  ("ar",    "English"),
    "ja":  ("ja",    "English"),
    "ko":  ("ko",    "English"),
    "th":  ("th",    "English"),
    "ka":  ("ka",    "English"),
    "am":  ("am",    "English"),
}


def load_poly_fever(language="en", max_samples=50):
    if not os.path.exists(TSV_PATH):
        raise FileNotFoundError(f"Poly-FEVER TSV not found at: {TSV_PATH}")

    col_name, _ = LANGUAGE_MAP.get(language, ("en", "English"))
    samples = []

    with open(TSV_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if len(samples) >= max_samples:
                break

            claim = row.get(col_name, "").strip()
            raw_label = row.get("Label", "").strip().lower()

            # Label is "true" = SUPPORTS, "false" = REFUTES
            if raw_label == "true":
                label = "SUPPORTS"
            elif raw_label == "false":
                label = "REFUTES"
            else:
                continue  # skip unknown labels

            if not claim:
                continue

            samples.append({
                "id":    row.get("ID", ""),
                "claim": claim,
                "label": label,
            })

    print(f"Loaded {len(samples)} samples | language='{language}' | column='{col_name}'")
    print(f"TSV path: {TSV_PATH}")
    return samples


def score_claim(claim, language_name):
    resp = requests.post(
        f"{MODEL_SERVER}/analyze",
        json={"prompt": claim, "language": language_name},
        timeout=180,
    )
    resp.raise_for_status()
    data   = resp.json()
    claims = data.get("claims", [])
    return claims[0] if claims else {"risk_score": 0.5, "hallucinated": False}


def evaluate(language="en", max_samples=50):
    # Check model server
    try:
        health = requests.get(f"{MODEL_SERVER}/health", timeout=3).json()
        if not health.get("models_loaded"):
            print("❌ Model server not ready. Run: python core/model_server.py")
            return
        print("✅ Model server is ready.\n")
    except Exception:
        print("❌ Model server not running. Run: python core/model_server.py")
        return

    _, language_name = LANGUAGE_MAP.get(language, ("en", "English"))

    try:
        samples = load_poly_fever(language, max_samples)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return

    if not samples:
        print(f"No samples found for language='{language}'")
        return

    tp = fp = tn = fn = 0
    results = []

    print(f"Evaluating {len(samples)} claims | language={language} | model_language={language_name}\n")
    print(f"{'#':<5} {'Ground':<14} {'Predicted':<14} {'Risk':<8} {'Result'}")
    print("-" * 58)

    for i, sample in enumerate(samples):
        claim  = sample["claim"]
        ground = sample["label"]   # SUPPORTS or REFUTES

        try:
            result    = score_claim(claim, language_name)
            risk      = result.get("risk_score", 0.5)
            predicted = "REFUTES" if result.get("hallucinated", False) else "SUPPORTS"
        except Exception as e:
            print(f"  ⚠️  Error on claim {i+1}: {e}")
            predicted, risk = "SUPPORTS", 0.5

        match = predicted == ground

        if ground == "REFUTES"  and predicted == "REFUTES":  tp += 1
        if ground == "SUPPORTS" and predicted == "REFUTES":  fp += 1
        if ground == "SUPPORTS" and predicted == "SUPPORTS": tn += 1
        if ground == "REFUTES"  and predicted == "SUPPORTS": fn += 1

        print(f"{i+1:<5} {ground:<14} {predicted:<14} {risk:<8.3f} {'✅' if match else '❌'}")
        print(f"      \"{claim[:90]}\"")

        results.append({
            "id":        sample["id"],
            "claim":     claim[:120],
            "ground":    ground,
            "predicted": predicted,
            "risk":      round(risk, 4),
            "correct":   match,
        })

    # ── Metrics ───────────────────────────────────────────────────────────
    total     = len(results)
    correct   = tp + tn
    accuracy  = correct / total                         if total              else 0
    precision = tp / (tp + fp)                         if (tp + fp) > 0      else 0
    recall    = tp / (tp + fn)                         if (tp + fn) > 0      else 0
    f1        = 2*precision*recall/(precision+recall)   if (precision+recall) > 0 else 0

    print(f"\n{'='*58}")
    print(f"  Poly-FEVER Results  |  language={language}  |  n={total}")
    print(f"{'='*58}")
    print(f"  Accuracy   : {accuracy:.1%}  ({correct}/{total} correct)")
    print(f"  Precision  : {precision:.1%}  (flagged hallucinations that were real)")
    print(f"  Recall     : {recall:.1%}  (real hallucinations that were caught)")
    print(f"  F1 Score   : {f1:.1%}")
    print(f"\n  Confusion Matrix:")
    print(f"    TP (correctly flagged)     : {tp}")
    print(f"    TN (correctly passed)      : {tn}")
    print(f"    FP (wrongly flagged)       : {fp}")
    print(f"    FN (missed hallucination)  : {fn}")
    print(f"{'='*58}")

    out_path = os.path.join(_PROJECT_ROOT, "tests", f"poly_fever_results_{language}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "language":  language,
            "total":     total,
            "accuracy":  round(accuracy,  4),
            "precision": round(precision, 4),
            "recall":    round(recall,    4),
            "f1":        round(f1,        4),
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "samples":   results,
        }, f, indent=2, ensure_ascii=False)

    print(f"\n  Results saved to tests/poly_fever_results_{language}.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--language",    default="en")
    parser.add_argument("--max_samples", default=50, type=int)
    args = parser.parse_args()
    evaluate(args.language, args.max_samples)
