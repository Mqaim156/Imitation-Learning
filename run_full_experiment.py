"""
run_full_experiment.py
----------------------
Run the complete domain randomization experiment.

This script:
1. Trains BC on no-DR demos
2. Trains BC on with-DR demos
3. Evaluates both on training seeds
4. Evaluates both on test seeds
5. Generates comparison table

Assumes demos have already been collected in data/no_dr and data/with_dr
"""

import os
import json
import pandas as pd

import config
from train_bc import train
from evaluate_policy import evaluate_policy


def main():
    print("\n" + "="*70)
    print("DOMAIN RANDOMIZATION EXPERIMENT")
    print("="*70 + "\n")
    
    # Paths
    no_dr_data = os.path.join(config.DATA_DIR, "no_dr")
    with_dr_data = os.path.join(config.DATA_DIR, "with_dr")
    no_dr_model = os.path.join(config.MODELS_DIR, "bc_no_dr.pt")
    with_dr_model = os.path.join(config.MODELS_DIR, "bc_with_dr.pt")
    
    # Check data exists
    for path in [no_dr_data, with_dr_data]:
        if not os.path.exists(os.path.join(path, "metadata.jsonl")):
            print(f"ERROR: No demos found at {path}")
            print("Please collect demos first with collect_demos.py")
            return
    
    # =================================================================
    # STEP 1: Train policies
    # =================================================================
    print("\n" + "-"*50)
    print("STEP 1: Training policies")
    print("-"*50)
    
    print("\n[1a] Training BC on demos WITHOUT domain randomization...")
    train(no_dr_data, no_dr_model)
    
    print("\n[1b] Training BC on demos WITH domain randomization...")
    train(with_dr_data, with_dr_model)
    
    # =================================================================
    # STEP 2: Evaluate policies
    # =================================================================
    print("\n" + "-"*50)
    print("STEP 2: Evaluating policies")
    print("-"*50)
    
    results = {}
    
    # Evaluate no-DR policy
    print("\n[2a] Evaluating BC (no DR) on TRAINING seeds...")
    results["no_dr_train"] = evaluate_policy(
        no_dr_model, config.TRAINING_SEEDS, num_episodes=50
    )
    
    print("\n[2b] Evaluating BC (no DR) on TEST seeds...")
    results["no_dr_test"] = evaluate_policy(
        no_dr_model, config.TEST_SEEDS, num_episodes=50
    )
    
    # Evaluate with-DR policy
    print("\n[2c] Evaluating BC (with DR) on TRAINING seeds...")
    results["with_dr_train"] = evaluate_policy(
        with_dr_model, config.TRAINING_SEEDS, num_episodes=50
    )
    
    print("\n[2d] Evaluating BC (with DR) on TEST seeds...")
    results["with_dr_test"] = evaluate_policy(
        with_dr_model, config.TEST_SEEDS, num_episodes=50
    )
    
    # =================================================================
    # STEP 3: Generate comparison table
    # =================================================================
    print("\n" + "-"*50)
    print("STEP 3: Results Summary")
    print("-"*50)
    
    # Create summary table
    table_data = {
        "Policy": ["BC (no DR)", "BC (no DR)", "BC (with DR)", "BC (with DR)"],
        "Eval Seeds": ["Training", "Test", "Training", "Test"],
        "Success Rate": [
            f"{results['no_dr_train']['success_rate']*100:.1f}%",
            f"{results['no_dr_test']['success_rate']*100:.1f}%",
            f"{results['with_dr_train']['success_rate']*100:.1f}%",
            f"{results['with_dr_test']['success_rate']*100:.1f}%",
        ],
        "Avg Time": [
            f"{results['no_dr_train']['avg_time_to_success']:.1f}" if results['no_dr_train']['avg_time_to_success'] else "N/A",
            f"{results['no_dr_test']['avg_time_to_success']:.1f}" if results['no_dr_test']['avg_time_to_success'] else "N/A",
            f"{results['with_dr_train']['avg_time_to_success']:.1f}" if results['with_dr_train']['avg_time_to_success'] else "N/A",
            f"{results['with_dr_test']['avg_time_to_success']:.1f}" if results['with_dr_test']['avg_time_to_success'] else "N/A",
        ],
    }
    
    df = pd.DataFrame(table_data)
    print("\n" + df.to_string(index=False))
    
    # Compute generalization drop
    no_dr_drop = results['no_dr_train']['success_rate'] - results['no_dr_test']['success_rate']
    with_dr_drop = results['with_dr_train']['success_rate'] - results['with_dr_test']['success_rate']
    
    print(f"\n\nGENERALIZATION ANALYSIS:")
    print(f"  BC (no DR): {no_dr_drop*100:.1f}% drop on new seeds")
    print(f"  BC (with DR): {with_dr_drop*100:.1f}% drop on new seeds")
    
    if with_dr_drop < no_dr_drop:
        improvement = no_dr_drop - with_dr_drop
        print(f"\n  ✓ Domain randomization IMPROVES generalization by {improvement*100:.1f}%")
    else:
        print(f"\n  ✗ Domain randomization did not improve generalization")
    
    # =================================================================
    # STEP 4: Failure analysis
    # =================================================================
    print("\n" + "-"*50)
    print("STEP 4: Failure Analysis")
    print("-"*50)
    
    for name, result in results.items():
        print(f"\n{name}:")
        for ftype, count in result['failure_counts'].items():
            if ftype != "none" and count > 0:
                print(f"  {ftype}: {count}")
    
    # Save all results
    results_path = os.path.join(config.PROJECT_ROOT, "experiment_results.json")
    
    # Convert for JSON
    results_json = {}
    for key, val in results.items():
        results_json[key] = {
            k: v for k, v in val.items() 
            if k != "episode_results"
        }
    
    with open(results_path, "w") as f:
        json.dump(results_json, f, indent=2)
    
    print(f"\n\nFull results saved to: {results_path}")
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()