import json
import pandas as pd
import os

def export_impact_analysis(input_file, output_file):
    """
    Reads the nested impact analysis JSON and exports it to a flat Excel file.
    
    Args:
        input_file (str): Path to the input JSON file.
        output_file (str): Path to the output Excel file.
    """
    print(f"Reading data from {input_file}...")
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file}")
        return
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from {input_file}")
        return

    rows = []

    # Iterate through questions
    for question_text, qdata in data.items():
        # questions are keys, but we also want to be robust if structure varies slightly
        # based on user example:
        # Key is "Question Text"
        # Value contains "question": "Question Text", "prompt_results": {...}
        
        # We can use the key as the stable identifier for now if needed, 
        # or just use the text. The user req asked for 'question_id' (stable identifier from top-level key)
        question_id = question_text 
        
        # Just in case the inner "question" field is different or missing, defaulting to key
        actual_question_text = qdata.get("question", question_text)
        
        prompt_results = qdata.get("prompt_results", {})
        
        for prompt_version, pdata in prompt_results.items():
            scores = pdata.get("scores", {})
            
            # Extract scores with safe defaults (None for missing)
            # Ensure we look for the specific keys requested
            grade = scores.get("grade")
            rqi = scores.get("rqi")
            ragas_score = scores.get("ragas")
            retrieval_score = scores.get("retrieval")
            generation_score = scores.get("generation")
            model_name = scores.get("model_name")
            
            row = {
                "question_id": question_id,
                "question_text": actual_question_text,
                "prompt_version": prompt_version,
                "grade": grade,
                "rqi": rqi,
                "ragas": ragas_score,
                "retrieval": retrieval_score,
                "generation": generation_score,
                "model_name": model_name
            }
            rows.append(row)

    if not rows:
        print("No data found to export.")
        return

    df = pd.DataFrame(rows)

    # Explicitly ensure numeric types for score columns
    numeric_cols = ["rqi", "ragas", "retrieval", "generation"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    print(f"Data flattened. Found {len(df)} rows.")
    print(f"Exporting to {output_file}...")

    # Ensure output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        df.to_excel(output_file, index=False)
        print("Export successful! ✅")
    except Exception as e:
        print(f"Error writing to Excel file: {e}")

if __name__ == "__main__":
    # Define paths relative to the project root or use absolute paths
    # Assuming script is run from project root, but let's be robust
    
    # Base directory is the parent of src (i.e., project root)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    input_json_path = os.path.join(base_dir, "data", "prompt_versions", "prompt_impact_analysis.json")
    output_excel_path = os.path.join(base_dir, "data", "prompt_versions", "impact_analysis.xlsx")
    
    export_impact_analysis(input_json_path, output_excel_path)
