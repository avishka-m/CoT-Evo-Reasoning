"""
main.py
-------
Entry point for the CoT-Evo GA Medical Reasoning System.

Batch Mode (default):
    Iterates over every JSON file in the `data/` folder.
    Files already having a matching output are SKIPPED.
    Up to MAX_PARALLEL_FILES files are processed concurrently.

Single-file override (optional):
    python main.py --data data/medical_qa.json
    python main.py --data data/medical_qa.json --generations 10
    python main.py --data data/sample.json --generations 1 --population 5
"""

import argparse
import os
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import GA_GENERATIONS, GA_POPULATION_SIZE, OUTPUT_DIR, DATA_DIR, MAX_PARALLEL_FILES
import config  # allow overriding via CLI args

from data_loader import load_dataset
from evolution_loop import run_evolution


def parse_args():
    parser = argparse.ArgumentParser(
        description="CoT-Evo: Genetic Algorithm for Medical Reasoning Optimization"
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help=(
            "Path to a single medical QA JSON file. "
            "If omitted, ALL files in the data/ folder are processed."
        ),
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=GA_GENERATIONS,
        help=f"Number of GA generations (default: {GA_GENERATIONS})",
    )
    parser.add_argument(
        "--population",
        type=int,
        default=GA_POPULATION_SIZE,
        help=f"Population size per question (default: {GA_POPULATION_SIZE})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Output JSON file path. Only used when --data targets a single file. "
            "In batch mode the output name mirrors the input filename."
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose per-question output",
    )
    return parser.parse_args()


def collect_pending_files(data_dir: str, output_dir: str):
    """
    Discover all .json files in data_dir.
    Return (input_path, output_path) tuples for files that do NOT yet have output.
    """
    os.makedirs(output_dir, exist_ok=True)

    all_input_files = sorted(glob.glob(os.path.join(data_dir, "*.json")))

    if not all_input_files:
        print(f"[BatchRunner] No JSON files found in '{data_dir}'.")
        return []

    pending = []
    for input_path in all_input_files:
        filename    = os.path.basename(input_path)
        output_path = os.path.join(output_dir, filename)

        if os.path.exists(output_path):
            print(f"[BatchRunner] SKIP  '{filename}'  →  output already exists.")
        else:
            print(f"[BatchRunner] QUEUE '{filename}'  →  will be processed.")
            pending.append((input_path, output_path))

    return pending


def process_single_file(
    input_path: str,
    output_path: str,
    generations: int,
    verbose: bool,
):
    """Load dataset from input_path, evolve it, save to output_path."""
    filename = os.path.basename(input_path)
    print("\n" + "=" * 60)
    print(f"  Processing : {filename}")
    print(f"  Output     : {output_path}")
    print("=" * 60)

    dataset = load_dataset(input_path)
    if not dataset:
        print(f"[ERROR] No valid records in '{input_path}'. Skipping.")
        return None

    results = run_evolution(
        dataset=dataset,
        generations=generations,
        output_path=output_path,
        verbose=verbose,
    )

    # Print best result for each question
    print(f"\n\n  -- FINAL EVOLVED REASONING ({filename}) --\n")
    for r in results:
        best = r["best_reasoning"]
        print(f"  ID      : {r['id']}")
        print(f"  Question: {r['question'][:100]}...")
        print(f"  Fitness : {best['fitness_score']:.4f}")
        print(f"  Model   : {best['model']}")
        print(
            f"  Scores  : Clinical Mean:{best.get('clinical_criteria_mean', 0):.2f} "
            f"| Evidence Score:{best.get('evidence_score', 0):.2f}"
        )
        print(f"  Grade   : {best.get('evidence_grade', 'D')}")
        print(f"  Step Breakdown:")
        for step in best.get("per_step", []):
            idx  = step.get("step_index", "?")
            scr  = step.get("score", 0)
            lvl  = step.get("evidence_level", "D")
            clin = step.get("clinical_scores", {})
            clin_str = (
                f"S:{clin.get('safety_utility', 0):.1f} "
                f"V:{clin.get('clinical_validity', 0):.1f} "
                f"R:{clin.get('clinical_relevance', 0):.1f}"
            )
            print(f"    Step {idx:2} [Lvl {lvl} | Score {scr:.2f}] -> {clin_str}")
        print()

    return results


def main():
    args = parse_args()

    # Allow CLI to override config constants
    config.GA_GENERATIONS     = args.generations
    config.GA_POPULATION_SIZE = args.population

    verbose = not args.quiet

    print("=" * 60)
    print("  CoT-Evo: Medical Reasoning GA System")
    print("=" * 60)
    print(f"  Generations      : {args.generations}")
    print(f"  Population       : {args.population}")
    print(f"  Parallel evals   : {config.MAX_PARALLEL_EVALS}")
    print(f"  Parallel questions: {config.MAX_PARALLEL_QUESTIONS}")
    print(f"  Parallel files   : {MAX_PARALLEL_FILES}")
    print("=" * 60)

    # ── Mode: single file explicitly provided ──────────────────────────────────
    if args.data is not None:
        output_path = args.output or os.path.join(
            OUTPUT_DIR, os.path.basename(args.data)
        )
        print(f"  Mode   : single file")
        print(f"  Dataset: {args.data}")
        print(f"  Output : {output_path}\n")
        process_single_file(
            input_path=args.data,
            output_path=output_path,
            generations=args.generations,
            verbose=verbose,
        )
        return

    # ── Mode: batch — process all files in data/ ───────────────────────────────
    print(f"  Mode   : batch (all files in '{DATA_DIR}/')\n")

    pending = collect_pending_files(DATA_DIR, OUTPUT_DIR)

    if not pending:
        print("\n[BatchRunner] Nothing to process. All files already have outputs.")
        return

    total = len(pending)
    print(f"\n[BatchRunner] {total} file(s) to process  "
          f"(up to {MAX_PARALLEL_FILES} concurrently).\n")

    success = 0
    failed  = 0

    # ── Parallel file processing ───────────────────────────────────────────────
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_FILES) as executor:
        futures = {
            executor.submit(
                process_single_file,
                input_path,
                output_path,
                args.generations,
                verbose,
            ): os.path.basename(input_path)
            for input_path, output_path in pending
        }

        for future in as_completed(futures):
            filename = futures[future]
            try:
                result = future.result()
                if result is not None:
                    print(f"[BatchRunner] ✓ Finished: {filename}")
                    success += 1
                else:
                    print(f"[BatchRunner] ✗ Skipped (no data): {filename}")
                    failed += 1
            except Exception as e:
                print(f"[BatchRunner] ERROR processing '{filename}': {e}")
                failed += 1

    print("\n" + "=" * 60)
    print("  BATCH COMPLETE")
    print(f"  Processed : {success}/{total} succeeded")
    if failed:
        print(f"  Failed    : {failed}")
    print("=" * 60)


if __name__ == "__main__":
    main()
