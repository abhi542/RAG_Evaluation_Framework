import json
import os
import argparse
import pandas as pd
from datasets import Dataset 
from ragas import evaluate
from ragas.metrics import (
    ContextPrecision,
    ContextRecall,
    ResponseRelevancy,
    Faithfulness
)

# Initialize classes
context_precision = ContextPrecision()
context_recall = ContextRecall()
answer_relevance = ResponseRelevancy()
faithfulness = Faithfulness()
from rag_pipeline import rag

def evaluate_ragas(test_file, index_dir, provider="offline", llm_provider="openai"):
    """
    Runs comprehensive RAG evaluation using RAGAS.
    """
    if not os.path.exists(test_file):
        print(f"Test file not found: {test_file}")
        return

    with open(test_file, 'r') as f:
        test_data = json.load(f)

    # Prepare data for RAGAS
    # RAGAS expects: question, answer, contexts, ground_truths
    
    questions = []
    answers = []
    contexts = []
    ground_truths = []

    print(f"Running RAG pipeline for {len(test_data)} queries...")
    
    for item in test_data:
        query = item.get("query")
        ground_truth = item.get("ground_truth") # RAGAS typically needs ground truth for recall
        
        if not query:
            continue
            
        result = rag(query, index_dir=index_dir, embedding_provider=provider, llm_provider=llm_provider)
        
        questions.append(query)
        answers.append(result["answer"])
        contexts.append(result["retrieved_chunks"])
        ground_truths.append(ground_truth if ground_truth else "")
        
        print(f"Questions: {query}")
        print(f"Generated: {result['answer']}")
        print(f"Ground Truth: {ground_truth}\n")
        import time
        time.sleep(10) # Avoid RPM limit

    # Create HF Dataset
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }
    dataset = Dataset.from_dict(data)

    print("Running RAGAS evaluation...")
    
    # Configure RAGAS with the same LLM/Embeddings as the pipeline
    from rag_pipeline import get_llm
    from embed_store import get_embeddings
    
    # Ragas expects LangChain objects
    eval_llm = get_llm(llm_provider)
    eval_embeddings = get_embeddings(provider)

    results = evaluate(
        dataset=dataset,
        metrics=[
            context_precision,
            context_recall,
            faithfulness,
            answer_relevance,
        ],
        llm=eval_llm, 
        embeddings=eval_embeddings
    )
    
    print("\nRAGAS Results:")
    print(results)
    
    # Save results
    results_df = results.to_pandas()
    results_df.to_csv("data/ragas_results.csv", index=False)
    results_df.to_csv("data/ragas_results.csv", index=False)
    print("Results saved to data/ragas_results.csv")

    # Calculate average across all metrics for single scalar
    # RAGAS returns a dict like {'context_precision': 0.8, 'faithfulness': 0.9...}
    # We take the mean of means.
    
    # Calculate average across all metrics for single scalar
    if not results_df.empty:
         # Explicitly select numeric metric columns
         target_metrics = ['context_precision', 'context_recall', 'faithfulness', 'answer_relevancy']
         # Only keep columns that actually exist in the dataframe
         metric_cols = [col for col in target_metrics if col in results_df.columns]
         
         if metric_cols:
             print(f"Calculating average from metrics: {metric_cols}")
             avg_ragas_score = results_df[metric_cols].mean().mean()
         else:
             print("Warning: No standard metric columns found in results.")
             avg_ragas_score = 0
    else:
         avg_ragas_score = 0

    # Save score for aggregation
    os.makedirs("data/results", exist_ok=True)
    with open("data/results/ragas_score.json", "w") as f:
        json.dump({"ragas_score": avg_score if 'avg_score' in locals() else avg_ragas_score}, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file", default="data/test_ragas.json")
    parser.add_argument("--index_dir", default="data/faiss_index")
    parser.add_argument("--provider", default="offline")
    parser.add_argument("--llm_provider", default="openai") # openai, vertex, grok, gemini
    args = parser.parse_args()

    # Check .env
    from dotenv import load_dotenv
    load_dotenv()
    
    evaluate_ragas(args.test_file, args.index_dir, provider=args.provider, llm_provider=args.llm_provider)
