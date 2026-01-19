import argparse
import subprocess
import sys
import os

def run_command(command, description):
    print(f"\n🚀 {description}...")
    print(f"   Command: {command}")
    try:
        # Stream output to console in real-time
        process = subprocess.Popen(
            command, 
            shell=True, 
            stdout=sys.stdout,
            stderr=sys.stderr,
            cwd=os.getcwd()
        )
        process.wait()
        
        if process.returncode != 0:
            print(f"❌ {description} failed with exit code {process.returncode}")
            sys.exit(process.returncode)
        else:
            print(f"✅ {description} completed.")
            
    except Exception as e:
        print(f"❌ Error executing {description}: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Run full RAG evaluation suite")
    parser.add_argument("--llm_provider", type=str, required=True, choices=["gemini", "groq", "openai", "grok"], help="LLM Provider to use")
    parser.add_argument("--run_name", type=str, default=None, help="Name for the final report (e.g. 'groq_v1')")
    parser.add_argument("--use_synthetic", action="store_true", default=True, help="Use synthetic data files (default: True)")
    
    args = parser.parse_args()
    
    # Define file paths based on flag
    suffix = "_synthetic.json" if args.use_synthetic else ".json"
    
    files = {
        "retrieval": f"data/test_retrieval{suffix}",
        "generation": f"data/test_generation{suffix}",
        "ragas": f"data/test_ragas{suffix}"
    }
    
    # 1. Eval Retrieval
    run_command(
        f"python src/eval_retrieval.py --test_file {files['retrieval']}",
        "Evaluating Retrieval (Recall)"
    )
    
    import time
    print("⏳ Sleeping 10s before Generation eval...")
    time.sleep(10)

    # 2. Eval Generation
    run_command(
        f"python src/eval_generation.py --test_file {files['generation']} --llm_provider {args.llm_provider}",
        "Evaluating Generation (Factuality)"
    )
    
    print("⏳ Sleeping 30s before RAGAS eval (Heavy API usage)...")
    time.sleep(30)
    
    # 3. Eval RAGAS
    run_command(
        f"python src/eval_ragas.py --test_file {files['ragas']} --llm_provider {args.llm_provider}",
        "Evaluating RAGAS (Reasoning)"
    )
    
    # 4. Aggregate
    save_flag = f"--save_name {args.run_name}" if args.run_name else ""
    run_command(
        f"python src/aggregate_scores.py {save_flag}",
        "Aggregating Final RQI Score"
    )
    
    print("\n🎉 Full Evaluation Suite Finished Successfully!")

if __name__ == "__main__":
    main()
