import json
import os
import argparse

def get_justification(retrieval, generation, ragas):
    justifications = []
    
    # Retrieval Analysis
    if retrieval < 0.6:
        justifications.append("❌ Retrieval is the bottleneck. The system struggled to find relevant documents. This implies that your embedding model (all-MiniLM-L6-v2) may not understand the domain-specific terms in the queries.")
    else:
        justifications.append("✅ Retrieval is healthy. The system is consistently finding the right context.")

    # Generation Analysis
    if generation < 0.6:
        justifications.append("❌ Factuality is low. The LLM missed mandated keywords. It might be hallucinating or summarizing too aggressively, missing key details.")
    elif generation > 0.8:
        justifications.append("✅ Factuality is high. The LLM is accurately including the required keywords and numbers.")

    # RAGAS Analysis
    if ragas < 0.6:
        justifications.append("❌ Reasoning (RAGAS) is poor. While the system might find keywords, the AI Judge found the answers unfaithful or irrelevant to the complex questions.")
    elif ragas > 0.8:
        justifications.append("✅ Reasoning is strong. The system handles complex queries well and produces faithful answers.")
    
    # Combined Paradox
    if retrieval > 0.8 and ragas < 0.5:
        justifications.append("⚠️ The 'RAG Paradox' is detected: Good Retrieval but Bad Reasoning. The LLM has the data but fails to synthesize it correctly. Try a smarter LLM (e.g., Upgrade from 8b to 70b).")
        
    return "\n".join(justifications)

def aggregate_scores(save_name=None):
    print("\n--- RAG Quality Index (RQI) Report ---\n")
    
    # 1. Load Scores
    try:
        with open("data/results/retrieval_score.json", "r") as f:
            retrieval = json.load(f).get("retrieval_score", 0)
    except FileNotFoundError:
        print("Warning: Retrieval score not found. Run src/eval_retrieval.py first.")
        retrieval = 0

    try:
        with open("data/results/generation_score.json", "r") as f:
            generation = json.load(f).get("generation_score", 0)
    except FileNotFoundError:
        print("Warning: Generation score not found. Run src/eval_generation.py first.")
        generation = 0

    try:
        with open("data/results/ragas_score.json", "r") as f:
            ragas = json.load(f).get("ragas_score", 0)
    except FileNotFoundError:
        print("Warning: RAGAS score not found. Run src/eval_ragas.py first.")
        ragas = 0

    # 2. Weights (Senior Engineer logic from README)
    W_RETRIEVAL = 0.4
    W_GENERATION = 0.3
    W_RAGAS = 0.3

    # 3. Calculate RQI
    rqi = (retrieval * W_RETRIEVAL) + (generation * W_GENERATION) + (ragas * W_RAGAS)

    # 4. Print Report
    print(f"Retrieval Score (Recall):   {retrieval:.2f}  (Weight: {W_RETRIEVAL})")
    print(f"Generation Score (Facts):   {generation:.2f}  (Weight: {W_GENERATION})")
    print(f"RAGAS Score (Reasoning):    {ragas:.2f}  (Weight: {W_RAGAS})")
    print("-" * 40)
    
    # Color code output if terminal supports it (simple logic)
    grade = "F"
    if rqi >= 0.9: grade = "A+"
    elif rqi >= 0.8: grade = "A"
    elif rqi >= 0.7: grade = "B"
    elif rqi >= 0.6: grade = "C"
    elif rqi >= 0.5: grade = "D"

    print(f"Final RQI Score:            {rqi:.2f} / 1.00")
    print(f"System Grade:               {grade}")
    print("-" * 40)
    
    # 5. Detailed Justification
    print("\n📝 SYSTEM JUSTIFICATION:")
    print(get_justification(retrieval, generation, ragas))
    print("-" * 40)

    # 6. Save Report
    report_data = {
        "model_name": save_name if save_name else "default",
        "retrieval": retrieval,
        "generation": generation,
        "ragas": ragas,
        "rqi": rqi,
        "grade": grade
    }
    
    if save_name:
        filename = f"data/results/report_{save_name}.json"
        with open(filename, "w") as f:
            json.dump(report_data, f, indent=4)
        print(f"\n[INFO] Full report saved to: {filename}")
        print("You can compare this with other models later.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_name", type=str, help="Name of the model/run to save results (e.g. 'groq_8b', 'gemini_flash')")
    args = parser.parse_args()
    
    aggregate_scores(args.save_name)
