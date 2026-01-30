import pandas as pd
import os

def create_dashboard(input_file, output_file):
    """
    Reads the flat valid Excel file and creates a multi-sheet dashboard.
    """
    print(f"Reading data from {input_file}...")
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found.")
        return

    df = pd.read_excel(input_file)
    
    print(f"Creating dashboard at {output_file}...")
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        # Check if xlsxwriter is installed, though pandas often defaults to 'openpyxl' if xlsxwriter isn't there.
        # The user requested 'xlsxwriter' explicitly in the snippet, so we try to use it.
        # If it fails, we can fallback or let it error (user likely has env capable if they provided snippet).
        # We will wrap in try-except for import error if needed, but standard pandas usage is safer.
        
        with pd.ExcelWriter(output_file) as writer:

            # =========================
            # Sheet 1: Raw Data
            # =========================
            df.to_excel(writer, sheet_name="Raw_Data", index=False)

            # =========================
            # Sheet 2: Prompt-Level Summary (main dashboard table)
            # =========================
            # Ensure numeric columns are actually numeric before groupby
            numeric_cols = ["rqi", "ragas", "retrieval", "generation"]
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            prompt_summary = (
                df.groupby("prompt_version")[numeric_cols]
                .mean()
                .reset_index()
                .round(4)
            )

            prompt_summary.to_excel(
                writer,
                sheet_name="Prompt_Summary",
                index=False
            )

            # =========================
            # Sheet 3: Question-Level RQI (heatmap / robustness)
            # =========================
            question_rqi = (
                df.pivot_table(
                    index="question_text",
                    columns="prompt_version",
                    values="rqi",
                    aggfunc="mean"
                )
                .reset_index()
                .round(4)
            )

            question_rqi.to_excel(
                writer,
                sheet_name="Question_RQI",
                index=False
            )

            # =========================
            # Sheet 4: Delta Analysis
            # =========================
            # Calculate deltas against ALL unique prompt versions found
            unique_versions = sorted(df["prompt_version"].unique())
            
            # Start with the base dataframe
            deltas_df = df.copy()
            
            # List of metrics to calculate deltas for
            metrics = ["rqi", "ragas"]
            
            # Keep track of delta columns to add them to output key columns
            delta_cols = []

            for base_ver in unique_versions:
                # Prepare the baseline data for this version
                # We filter for the specific version and select relevant columns
                baseline_subset = df[df["prompt_version"] == base_ver][["question_text"] + metrics].copy()
                
                # Rename columns to differentiate (e.g., rqi -> rqi_v0)
                suffix = f"_{base_ver}"
                rename_map = {m: f"{m}{suffix}" for m in metrics}
                baseline_subset = baseline_subset.rename(columns=rename_map)
                
                # Merge this specific baseline back to the main df on question_text
                # Note: This repeats the baseline values for every row with the same question
                deltas_df = deltas_df.merge(baseline_subset, on="question_text", how="left")
                
                # Calculate deltas for each metric
                for m in metrics:
                    base_col = f"{m}{suffix}"
                    delta_col_name = f"{m}_delta_vs_{base_ver}"
                    
                    if base_col in deltas_df.columns:
                        deltas_df[delta_col_name] = (deltas_df[m] - deltas_df[base_col]).round(4)
                        delta_cols.append(delta_col_name)
            
            # Select final columns: Identities + Metrics + All Deltas
            # We want to keep the original structure but append all the verified delta columns
            
            basic_cols = [
                "question_id", "question_text", "prompt_version", 
                "grade", "rqi", "ragas", "retrieval", "generation", "model_name"
            ]
            
            # Filter basic_cols to what actually exists
            final_cols = [c for c in basic_cols if c in deltas_df.columns]
            
            # Add the dynamically generated delta columns
            final_cols.extend(delta_cols)
            
            deltas_df = deltas_df[final_cols]

            deltas_df.to_excel(
                writer,
                sheet_name="Delta_Analysis",
                index=False
            )

        print("✅ impact_analysis_dashboard.xlsx created")
        
    except Exception as e:
        print(f"Error creating dashboard: {e}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_xlsx_path = os.path.join(base_dir, "data", "prompt_versions", "impact_analysis.xlsx")
    output_dashboard_path = os.path.join(base_dir, "data", "prompt_versions", "impact_analysis_dashboard.xlsx")
    
    create_dashboard(input_xlsx_path, output_dashboard_path)
