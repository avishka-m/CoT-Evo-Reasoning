"""
main.py
-------
Entry point for the CoT-Evo GA Medical Reasoning System.

Usage:
    python main.py --data data/medical_qa.json
    python main.py --data data/medical_qa.json --generations 10
    python main.py --data data/sample.json --generations 1 --population 5
"""

import argparse
import os
from config import GA_GENERATIONS, GA_POPULATION_SIZE, OUTPUT_DIR
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
        default=os.path.join("data", "medical_qa.json"),
        help="Path to the medical QA dataset JSON file",
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
        default=os.path.join(OUTPUT_DIR, "evolved_reasoning.json"),
        help="Output JSON file path",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose per-question output",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Allow CLI to override config constants
    config.GA_GENERATIONS    = args.generations
    config.GA_POPULATION_SIZE = args.population

    print("=" * 60)
    print("  CoT-Evo: Medical Reasoning GA System")
    print("=" * 60)
    print(f"  Dataset     : {args.data}")
    print(f"  Generations : {args.generations}")
    print(f"  Population  : {args.population}")
    print(f"  Output      : {args.output}")
    print("=" * 60)

    # Load dataset
    dataset = load_dataset(args.data)
    if not dataset:
        print("[ERROR] No valid records found in dataset. Exiting.")
        return

    # Run evolution
    results = run_evolution(
        dataset=dataset,
        generations=args.generations,
        output_path=args.output,
        verbose=not args.quiet,
    )

    # Print best result for each question
    print("\n\n  -- FINAL EVOLVED REASONING --\n")
    for r in results:
        best = r['best_reasoning']
        print(f"  ID      : {r['id']}")
        print(f"  Question: {r['question'][:100]}...")
        print(f"  Fitness : {best['fitness_score']:.4f}")
        print(f"  Model   : {best['model']}")
        
        # Display Clinical and Evidence Scores
        print(f"  Scores  : Clinical Mean:{best.get('clinical_criteria_mean',0):.2f} | Evidence Score:{best.get('evidence_score',0):.2f}")
        print(f"  Grade   : {best.get('evidence_grade', 'D')}")
        
        print(f"  Step Breakdown:")
        for step in best.get('per_step', []):
            idx = step.get('step_index', '?')
            scr = step.get('score', 0)
            lvl = step.get('evidence_level', 'D')
            
            # Clinical scores summary
            clin = step.get('clinical_scores', {})
            clin_str = f"S:{clin.get('safety_utility',0):.1f} V:{clin.get('clinical_validity',0):.1f} R:{clin.get('clinical_relevance',0):.1f}"
            
            print(f"    Step {idx:2} [Lvl {lvl} | Score {scr:.2f}] -> {clin_str}")
        print()


if __name__ == "__main__":
    main()
