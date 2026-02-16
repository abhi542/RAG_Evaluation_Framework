import argparse
import subprocess
import sys
import os
import json
import time

# Ensure project root is in path
sys.path.append(os.getcwd())

# Define Prompt Versions
PROMPT_VERSIONS = {
    "prompt_v0": "You are a helpful assistant. Answer the user’s question clearly and concisely.",
    "prompt_v1": "You are a helpful assistant.\nAnswer the user’s question clearly, using well-structured sentences.\nPrefer concise explanations, but include key details when helpful.",
    "prompt_v2": "You are a factual assistant.\nAnswer the question using only the information provided in the retrieved context.\nIf the context is insufficient, say that explicitly instead of guessing.",
    "prompt_v3": "You are an expert assistant.\nFirst, reason internally about what information from the retrieved context is relevant.\nThen provide a complete, well-explained answer grounded in that context.\nAvoid speculation and unsupported claims.",
    "prompt_v4": "You are an expert, evaluation-aware assistant.\n\nInstructions:\n- Ground every claim strictly in the retrieved context\n- Be complete but avoid unnecessary verbosity\n- Do not introduce external knowledge\n- If multiple interpretations exist, state them clearly\n- If the context is insufficient, say that explicitly\n\nProduce a clear, faithful, and well-structured answer."
}

def run_command(command, description, env=None):
    print(f"\n🚀 {description}...")
    print(f"   Command: {command}")
    try:
        # Stream output to console in real-time
        process = subprocess.Popen(
            command, 
            shell=True, 
            stdout=sys.stdout,
            stderr=sys.stderr,
            cwd=os.getcwd(),
            env=env if env else os.environ.copy()
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

def generate_change_summary(baseline_response, new_response, llm_provider):
    """
    Generates a brief summary of how the new response differs from the baseline.
    Uses the same LLM provider as the main run.
    """
    try:
        # Import internally to avoid circular dependencies if simple import
        from src.rag_pipeline import get_llm
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser

        llm = get_llm(llm_provider)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert editor comparing two AI responses."),
            ("human", f"""Compare the following two responses to the same question.
            
            Baseline Response:
            {baseline_response}
            
            New Response:
            {new_response}
            
            Briefly describe how the second answer differs from the first in meaning, detail, or completeness.
            Focus on the *quality* of the change (e.g. "more grounded", "less hallucination", "better structure").
            Keep it under 2 sentences.
            """)
        ])
        
        chain = prompt | llm | StrOutputParser()
        return chain.invoke({})
    except Exception as e:
        print(f"[WARNING] Failed to generate change summary: {e}")
        return "Comparison failed."

def run_prompt_version_eval(llm_provider, prompt_versions_to_run, use_synthetic):
    """
    Runs the RAG pipeline for specified prompt versions and saves structured output.
    """
    from src.rag_pipeline import rag
    
    # 1. Load Test Data
    suffix = "_synthetic.json" if use_synthetic else ".json"
    test_file = f"data/test_generation{suffix}"
    
    if not os.path.exists(test_file):
        print(f"❌ Test file not found: {test_file}")
        return

    with open(test_file, 'r') as f:
        test_data = json.load(f)

    results_dir = "data/prompt_versions"
    os.makedirs(results_dir, exist_ok=True)
    output_file = f"{results_dir}/prompt_impact_analysis.json"
    
    # Load existing results if any to preserve baseline
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            full_results = json.load(f)
    else:
        full_results = {}

    print(f"\n🔬 Starting Prompt Version Analysis for: {prompt_versions_to_run}")
    
    for p_ver in prompt_versions_to_run:
        # Load fresh or existing results
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                full_results = json.load(f)
        else:
             full_results = {}
             
        # Step 1: Generate Responses & Baseline Comparison (Detailed View)
        # We need to ensure we have the response text first
        
        # Check if we need to run RAG generation for this version?
        # Yes, to populate the 'response' and 'change_summary' fields in the detailed JSON
        print(f"\n--- Processing {p_ver} ---")
        
        # 1.1 Generation Loop (Per Query)
        # We only need to do this if we don't have the response yet (or forced)
        # But to be safe and consistent, we should probably run it unless we want to rely on the eval scripts?
        # The user wants "Response" and "Metadata" in the JSON. Eval scripts don't easily give metadata per query in a structured way.
        # So we keep the generation loop.
        
        for item in test_data:
            question = item.get("query")
            if not question: continue
            
            if question not in full_results:
                full_results[question] = {"question": question, "prompt_results": {}}
            
            if p_ver not in full_results[question]["prompt_results"]:
                print(f"   Generating {p_ver} response for: '{question[:30]}...'")
                sys_prompt = PROMPT_VERSIONS.get(p_ver)
                res = rag(question, llm_provider=llm_provider, system_prompt=sys_prompt, prompt_version=p_ver)
                
                entry = {
                    "response": res["answer"],
                    "metadata": {k:v for k,v in res.items() if k != "answer"}
                }
                
                # If not baseline, generate summary
                if p_ver != "prompt_v0":
                    # Ensure baseline exists (it should, if we processed v0 first)
                    # Implementation detail: User constraints say v0 is baseline.
                    # If v0 is not in results, we should probably error or generate it first.
                    # For now assuming v0 is generated first or exists.
                    if "prompt_v0" in full_results[question]["prompt_results"]:
                        baseline_resp = full_results[question]["prompt_results"]["prompt_v0"]["response"]
                        entry["change_summary"] = generate_change_summary(baseline_resp, res["answer"], llm_provider)
                
                full_results[question]["prompt_results"][p_ver] = entry
                
                # Save intermediate
                with open(output_file, 'w') as f:
                    json.dump(full_results, f, indent=2)
                time.sleep(2)

        # Step 2: Run Full Evaluation Suite for this Version (Aggregate Scores)
        # We use Env Vars to force the pipeline to use this prompt version
        print(f"   Running Evaluation Scripts for {p_ver}...")
        
        # Set Env Vars for subprocesses
        env = os.environ.copy()
        env["RAG_SYSTEM_PROMPT"] = PROMPT_VERSIONS[p_ver]
        env["RAG_PROMPT_VERSION"] = p_ver
        
        # 2.1 Eval Retrieval (Optional? But user asked for retrieval score)
        # Retrieval shouldn't change with prompt, but we run it to get the score in the final report
        run_command(f"python src/eval_retrieval.py --test_file data/test_retrieval{suffix}", f"Eval Retrieval ({p_ver})", env=env)
        
        # 2.2 Eval Generation
        run_command(f"python src/eval_generation.py --test_file data/test_generation{suffix} --llm_provider {llm_provider}", f"Eval Generation ({p_ver})", env=env)
        
        # 2.3 Eval RAGAS
        run_command(f"python src/eval_ragas.py --test_file data/test_ragas{suffix} --llm_provider {llm_provider}", f"Eval RAGAS ({p_ver})", env=env)
        
        # 2.4 Aggregate
        run_command(f"python src/aggregate_scores.py --save_name {p_ver}", f"Aggregate Scores ({p_ver})", env=env)
        
        # Step 3: Harvest Scores and Attach to JSON
        report_path = f"data/results/report_{p_ver}.json"
        if os.path.exists(report_path):
            with open(report_path, 'r') as f:
                scores = json.load(f)
            
            # Attach these scores to EVERY question entry for this prompt version
            # (As per requirement "Attach evaluation scores... to the output JSON")
            print(f"   Attaching scores from {report_path}...")
            
            for q in full_results:
                if p_ver in full_results[q]["prompt_results"]:
                    # Create a copy of scores to add prompt_version metadata if missing
                    score_entry = scores.copy()
                    score_entry["prompt_version"] = p_ver
                    full_results[q]["prompt_results"][p_ver]["scores"] = score_entry
            
            with open(output_file, 'w') as f:
                json.dump(full_results, f, indent=2)
        else:
            print(f"⚠️ Warning: Report file {report_path} not found. Scores not attached.")

    print(f"\n✅ Prompt Analysis Complete. Results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Run full RAG evaluation suite")
    parser.add_argument("--llm_provider", type=str, required=True, choices=["gemini", "groq", "openai", "grok"], help="LLM Provider to use")
    parser.add_argument("--run_name", type=str, default=None, help="Name for the final report (e.g. 'groq_v1')")
    parser.add_argument("--use_synthetic", action="store_true", default=True, help="Use synthetic data files (default: True)")
    
    # New Argument for Prompt Versions
    parser.add_argument("--prompt_versions", type=str, default=None, help="Comma-separated list of prompt versions to run (e.g. 'prompt_v0,prompt_v1'). If None, runs standard eval.")

    args = parser.parse_args()
    
    # Define file paths based on flag
    suffix = "_synthetic.json" if args.use_synthetic else ".json"
    
    files = {
        "retrieval": f"data/test_retrieval{suffix}",
        "generation": f"data/test_generation{suffix}",
        "ragas": f"data/test_ragas{suffix}"
    }

    # MODE SWITCH: If --prompt_versions is provided, ONLY run the version analysis
    if args.prompt_versions:
        versions_to_run = [v.strip() for v in args.prompt_versions.split(",")]
        # Validate versions
        for v in versions_to_run:
            if v not in PROMPT_VERSIONS:
                print(f"❌ Error: Unknown prompt version '{v}'. Available: {list(PROMPT_VERSIONS.keys())}")
                sys.exit(1)
        
        run_prompt_version_eval(args.llm_provider, versions_to_run, args.use_synthetic)
        return # Exit main after prompt analysis

    # --- STANDARD EVALUATION FLOW (Original) ---
    
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
