import json
import argparse
import random
import os
from tqdm import tqdm
from dotenv import load_dotenv

# Import LLM setup from pipeline
from rag_pipeline import get_llm

# Fix imports for optional dependencies in LCEL
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List

load_dotenv()

# Define Output Structures using Pydantic
class RetrievalTestCase(BaseModel):
    query: str = Field(description="The question to ask")
    expected_keywords: List[str] = Field(description="List of 3-5 unique keywords that must appear in the source document chunk")

class GenerationTestCase(BaseModel):
    query: str = Field(description="The question to ask")
    expected_keywords: List[str] = Field(description="List of 3-5 keywords that must appear in the answer")

class RagasTestCase(BaseModel):
    query: str = Field(description="A complex question requiring reasoning")
    ground_truth: str = Field(description="A detailed, comprehensive answer to the question based ONLY on the text")

class SyntheticDataBatch(BaseModel):
    retrieval_cases: List[RetrievalTestCase]
    generation_cases: List[GenerationTestCase]
    ragas_cases: List[RagasTestCase]

def generate_data(docs_file, num_samples=5, llm_provider="openai"):
    print(f"Loading documents from {docs_file}...")
    chunks = []
    with open(docs_file, 'r', encoding='utf-8') as f:
        for line in f:
            chunks.append(json.loads(line))
    
    if not chunks:
        print("No chunks found.")
        return

    # Sample random chunks to generate questions from
    selected_chunks = random.sample(chunks, min(num_samples, len(chunks)))
    
    llm = get_llm(llm_provider)
    
    parser = JsonOutputParser(pydantic_object=SyntheticDataBatch)
    
    prompt = PromptTemplate(
        template="""You are an expert at creating evaluation datasets for RAG systems.
        Given the following text chunk from a document, generate:
        1. One 'Retrieval' test case: A specific, detail-oriented question where the answer is explicitly found in the text. Avoid generic questions. Keyphrases must be exact terms from the text.
        2. One 'Generation' test case: A question asking for a conceptual explanation or summary of the chunk. Keywords expected should cover the main ideas.
        3. One 'Ragas' test case: A complex question requiring reasoning, cause-and-effect analysis, or synthesis of the information. Provide a detailed, comprehensive ground truth answer.
        
        Text Chunk:
        {text}
        
        {format_instructions}
        """,
        input_variables=["text"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    
    chain = prompt | llm | parser
    
    all_retrieval = []
    all_generation = []
    all_ragas = []
    
    print(f"Generating synthetic data using {llm_provider}...")
    for chunk in tqdm(selected_chunks):
        try:
            # We treat the single chunk result as a batch of 1 for simplicity of pydantic model, 
            # effectively asking for 1 of each type per chunk.
            # We modify the prompt or logic if we want more per chunk, but 1:1 is good for now.
            
            # Note: The parser expects the structure of SyntheticDataBatch, so the LLM should return lists.
            # We ask for lists of length 1 in the prompt logic implicitly or we can be explicit.
            # Let's rely on the schema.
            
            output = chain.invoke({"text": chunk["text"]})
            
            # Append to master lists
            all_retrieval.extend(output["retrieval_cases"])
            all_generation.extend(output["generation_cases"])
            all_ragas.extend(output["ragas_cases"])
            
        except Exception as e:
            print(f"Failed to generate for chunk {chunk['chunk_id']}: {e}")
            # Try to debug by running without parser if possible, or just print typical error
            # For now just let the user know.

    # Save to files
    print("\nSaving generated datasets...")
    
    with open("data/test_retrieval_synthetic.json", "w") as f:
        json.dump(all_retrieval, f, indent=4)
    print(f"- data/test_retrieval_synthetic.json ({len(all_retrieval)} items)")

    with open("data/test_generation_synthetic.json", "w") as f:
        json.dump(all_generation, f, indent=4)
    print(f"- data/test_generation_synthetic.json ({len(all_generation)} items)")

    with open("data/test_ragas_synthetic.json", "w") as f:
        json.dump(all_ragas, f, indent=4)
    print(f"- data/test_ragas_synthetic.json ({len(all_ragas)} items)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs_file", default="data/docs_metadata.jsonl")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of chunks to process")
    parser.add_argument("--llm_provider", default="openai")
    args = parser.parse_args()
    
    generate_data(args.docs_file, args.num_samples, args.llm_provider)
