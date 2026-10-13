import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = "mlx-community/Qwen2.5-3B-Instruct-4bit"
OUTPUT = ROOT / "outputs" / "column_predictions.json"

BOOL_KEYS = [
    "direct_identifier",
    "quasi_identifier",
    "sensitive_attribute",
    "needs_review",
]

DESCRIPTIONS = {
    "age": "The person's age in years.",
    "workclass": "The person's employment sector or employment type.",
    "fnlwgt": (
        "Census sampling weight used for population statistics. "
        "Not a person ID. Under this project's threat model, "
        "it is unavailable for external matching."
    ),
    "education": "The person's highest education level.",
    "education_num": "Numeric encoding of the person's education level.",
    "marital_status": "The person's marital status.",
    "occupation": "The person's occupation category.",
    "relationship": "The person's relationship role within their household.",
    "race": "The person's recorded racial category.",
    "sex": "The person's recorded sex.",
    "capital_gain": "The person's financial capital gains.",
    "capital_loss": "The person's financial capital losses.",
    "hours_per_week": "The number of hours the person works per week.",
    "native_country": "The person's country of origin.",
    "income": (
        "The person's annual income bracket: "
        "greater than $50,000 or at most $50,000."
    ),
}

INSTRUCTIONS = """
Classify ONE column in a dataset about people.

THREAT MODEL:
An attacker may know a person's demographic, education,
employment, or household facts and combine them to match records.
Private financial details are not assumed externally available.

CLASSIFICATION POLICY:
1. direct_identifier:
   True for names, personal IDs, or other direct identifiers.

2. quasi_identifier:
   True when the attribute can help narrow down candidate records
   in combination with externally known facts.
   It does NOT need to identify someone alone.
   Common values and few categories do NOT rule out this role.
   Age is a typical example of a quasi-identifier.
   Consider demographic, education, employment, and household
   facts under the threat model above.

3. sensitive_attribute:
   True for private financial information, including income
   brackets, capital gains, and capital losses.
   An income bracket remains sensitive even with only two categories.
   This project also treats race, sex, marital status, and household
   relationship information as sensitive.

4. needs_review:
   A BOOLEAN: true if the meaning or role is uncertain,
   otherwise false. Never put an explanation in this field.

Roles are independent. A column may be both QI and Sensitive.
Follow the supplied description and do not invent statistics.
Put the explanation ONLY in the reason field.

OUTPUT FORMAT:
Return exactly one JSON object with these five keys:
{
  "direct_identifier": false,
  "quasi_identifier": true,
  "sensitive_attribute": false,
  "needs_review": false,
  "reason": "A short explanation consistent with the selected roles."
}

The object above demonstrates the format, not the answer.
All four flags must be JSON booleans, never strings.
Include reason as a nonempty string.
No Markdown and no text outside the JSON.
"""


def parse_prediction(text):
    decoder = json.JSONDecoder()
    last_problem = "No JSON object found."

    for index, character in enumerate(text):
        if character != "{":
            continue

        try:
            prediction, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue

        if not isinstance(prediction, dict):
            continue

        invalid = [
            key
            for key in BOOL_KEYS
            if type(prediction.get(key)) is not bool
        ]

        if invalid:
            last_problem = (
                "Missing or non-boolean fields: " + ", ".join(invalid)
            )
            continue

        reason = prediction.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            last_problem = "Missing or empty reason string."
            continue

        return {
            key: prediction[key]
            for key in BOOL_KEYS + ["reason"]
        }

    raise ValueError(last_problem)


def generate(prompt):
    command = [
        sys.executable,
        "-m",
        "mlx_lm.generate",
        "--model",
        MODEL,
        "--prompt",
        prompt,
        "--max-tokens",
        "400",
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            result.stderr.strip()
            or result.stdout.strip()
            or "Model generation failed."
        )

    return result.stdout


def classify(model_input):
    base_prompt = (
        INSTRUCTIONS
        + "\nCOLUMN TO CLASSIFY:\n"
        + json.dumps(model_input, ensure_ascii=False)
    )

    attempts = []
    prompt = base_prompt

    for attempt_number in range(1, 3):
        try:
            raw_output = generate(prompt)
        except RuntimeError as error:
            attempts.append({
                "attempt": attempt_number,
                "error": str(error),
            })
            return None, attempts, str(error)

        attempt = {
            "attempt": attempt_number,
            "raw_output": raw_output,
        }

        try:
            prediction = parse_prediction(raw_output)
            attempt["status"] = "ok"
            attempts.append(attempt)
            return prediction, attempts, None

        except ValueError as error:
            attempt["status"] = "error"
            attempt["error"] = str(error)
            attempts.append(attempt)

            prompt = (
                base_prompt
                + "\nFORMAT REMINDER:\n"
                + "Your previous response failed validation: "
                + str(error)
                + "\nClassify the column again. "
                + "needs_review must be true or false. "
                + "Put all explanations in reason. "
                + "Return all five required keys."
            )

    return None, attempts, attempts[-1]["error"]


def main():
    metadata_path = ROOT / "data" / "column_metadata.csv"

    with metadata_path.open(
        encoding="utf-8-sig",
        newline="",
    ) as file:
        columns = list(csv.DictReader(file))

    if not columns:
        raise SystemExit("The metadata file is empty.")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    if OUTPUT.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        backup = OUTPUT.with_name(
            f"column_predictions_{timestamp}.json"
        )
        backup.write_bytes(OUTPUT.read_bytes())

    results = []

    for number, metadata in enumerate(columns, start=1):
        column = metadata["column"]

        model_input = {
            "dataset": metadata.get("dataset", "adult"),
            "column": column,
            "description": DESCRIPTIONS.get(
                column,
                "Meaning unknown; needs human review.",
            ),
        }

        print(
            f"[{number}/{len(columns)}] Classifying {column}...",
            flush=True,
        )

        prediction, attempts, error = classify(model_input)

        entry = {
            "column": column,
            "model_input": model_input,
            "metadata": metadata,
            "attempts": attempts,
        }

        if prediction is None:
            entry["status"] = "error"
            entry["error"] = error
            print(f"  Error: {error}\n")

        else:
            entry.update(prediction)
            entry["other"] = not any(
                prediction[key] for key in BOOL_KEYS[:3]
            )
            entry["status"] = "ok"

            print(f"  QI: {prediction['quasi_identifier']}")
            print(f"  Sensitive: {prediction['sensitive_attribute']}")
            print(f"  Needs review: {prediction['needs_review']}")
            print(f"  Reason: {prediction['reason']}\n")

        results.append(entry)

        OUTPUT.write_text(
            json.dumps(results, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    successful = [
        result for result in results
        if result["status"] == "ok"
    ]

    failed = [
        result["column"] for result in results
        if result["status"] == "error"
    ]

    print("\nPredicted QI columns:")
    print([
        result["column"]
        for result in successful
        if result["quasi_identifier"]
    ])

    print("\nPredicted Sensitive columns:")
    print([
        result["column"]
        for result in successful
        if result["sensitive_attribute"]
    ])

    print("\nFailed columns:", failed)

    if failed:
        print(
            "Classification is incomplete. "
            "Review failed columns before calculating privacy metrics."
        )

    print("\nSaved:", OUTPUT)
    print("Model predictions require human review before approval.")


if __name__ == "__main__":
    main()