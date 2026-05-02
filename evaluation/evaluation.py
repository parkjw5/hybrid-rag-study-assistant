import json
import csv
import time
from pathlib import Path

from core.generator import RAGGenerator


# -----------------------------------
# Config
# -----------------------------------
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
OUTPUT_PATH = "data/results.csv"
QUERY_PATH = "evaluation/queries.json"


# -----------------------------------
# Helper: Detect Abstention
# -----------------------------------
def is_abstained(answer: str):
    return answer.strip() == "I cannot find the answer in the provided materials."


# -----------------------------------
# Main Evaluation
# -----------------------------------
def run_evaluation():

    print("Loading questions...")
    with open(QUERY_PATH, "r", encoding="utf-8") as f:
        queries = json.load(f)

    generator = RAGGenerator()

    Path("data").mkdir(exist_ok=True)

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)

        # ✅ Updated CSV Header
        writer.writerow([
            "question_id",
            "category",
            "alpha",
            "question",
            "answer",
            "confidence",
            "abstained",
            "num_sources",
            "retrieved_context"
        ])

        total_runs = len(queries) * len(ALPHAS)
        run_count = 0

        for alpha in ALPHAS:
            print(f"\nRunning evaluation for alpha = {alpha}")

            for q in queries:
                run_count += 1
                print(f"[{run_count}/{total_runs}] Q{q['id']}")

                answer, sources, confidence = generator.answer(
                    question=q["question"],
                    alpha=alpha
                )

                abstained = is_abstained(answer)
                num_sources = len(sources)

                # ✅ Combine retrieved sources into one string
                # Assuming `sources` is a list of strings
                combined_context = "\n\n---\n\n".join(
                    [str(s) for s in sources]
                )

                writer.writerow([
                    q["id"],
                    q["category"],
                    alpha,
                    q["question"],
                    answer.replace("\n", " "),
                    confidence,
                    abstained,
                    num_sources,
                    combined_context.replace("\n", " ")  # keep CSV safe
                ])

                csvfile.flush()
                time.sleep(0.5)

    print("\nEvaluation complete.")
    print(f"Results saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    run_evaluation()