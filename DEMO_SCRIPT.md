# 🎥 RAG Evaluation Framework - Team Presentation Script

**Context**: You are presenting to the entire Engineering Team / Stakeholders.
**Goal**: Demonstrate how you turned a "black box" AI prototype into a "measured, reliable system".
**Tone**: Professional, confident, engineer-to-engineer.

---

## 🎬 1. The "Why" (Introduction)
**Visual Cue**: 
> Open the `README.md` file (or a slide if you have one) showing the project title.
> Then switch to `src/rag_pipeline.py` and scroll slowly.

**Voiceover**:
"Hello everyone. Today I want to show you the **RAG Evaluation Framework** I've built.

We're all excited about Retrieval Augmented Generation (RAG). It allows us to chat with our own data. But as I was building this, I hit a major problem: **Trust.**

When the bot gives an answer, how do we *know* it's right?
- Did it just hallucinate?
- Did it miss the relevant document?
- Or did it just fail to understand the question?

Without metrics, we're just guessing. So, I built this framework to move us from 'It feels right' to 'The data says it's accurate'."

---

## 🏗️ 2. The Architecture (How it Works)
**Visual Cue**: 
> Open `implementation_plan.md` (or the architecture diagram if you have one).
> Briefly hover over the file names in the VS Code sidebar: `load_docs.py`, `embed_store.py`, `rag_pipeline.py`.

**Voiceover**:
"Let me briefly walk you through the architecture.
1.  **Ingestion**: I use `load_docs.py` to chunk our PDFs and store them in a **FAISS** vector database.
2.  **Retrieval**: I'm using `SentenceTransformers` for embeddings to keep it fast and cost-effective.
3.  **Generation**: The pipeline is modular. I can hot-swap **OpenAI**, **Gemini**, or **Groq (Llama-3)** with a single flag.

But the real magic isn't the bot—it's the **Evaluation Engine**."

---

## ⚖️ 3. The Unified Evaluation Suite
**Visual Cue**: 
> Open `src/run_eval.py`. Highlight the `main()` function showing the 3 steps.

**Voiceover**:
"I solved the 'Evaluation' problem by building a unified test suite called `run_eval.py`.
It automates three specific tests:

1.  **Retrieval Recall**: It checks if the vector DB actually finds the document containing the answer.
2.  **Generation Factuality**: It checks if the LLM followed specific constraints (like including mandatory keywords).
3.  **Reasoning (RAGAS)**: This is the advanced part. I use an **AI Judge** (another LLM) to grade the 'Faithfulness' and 'Relevancy' of the answer."

---

## 🚀 4. The Live Demo (One-Click Execution)
**Visual Cue**: 
> Open the Terminal.
> Clear the screen (`clear` or `cls`).
> Type: `python src/run_eval.py --llm_provider groq --run_name groq_demo_final`
> **Action**: Press Enter. Let it run.

**Voiceover**:
"In the past, testing a new model took me hours of manual checking.
Now, it takes one command.

*(Point to terminal as logs scroll)*
You can see it finding the documents... generating the answers using **Groq's Llama-3**... and now the AI Judge is scoring it in real-time.

I also implemented robust **Rate Limit Handling**. If the API throttles us (which happens often with free tiers), my system automatically detects it, backs off, and retries. This ensures our nightly builds never crash due to API fluctuation."

---

## 📈 5. Ensuring Reliability (The "Rule of 3")
**Visual Cue**: 
> Briefly show a slide or just explain.
> Or show three different JSON reports if you have them prepared (e.g., `run1.json`, `run2.json`, `run3.json`).

**Voiceover**:
"One critical lesson I learned: **LLMs are non-deterministic.**
Even with `temperature=0`, you can get slightly different answers for the same question.
This is why my framework encourages the **'Rule of 3'**. 
For a production deployment, I run this evaluation suite 3 times and average the scores.
- This smooths out any random noise.
- It ensures that if a model passes, it's not a fluke.
(For this demo, I'm showing just one run for speed, but the underlying principle is statistical significance.)"

---

## 📊 5. Analyzing the Results (The "RQI" Metric)
**Visual Cue**: 
> Open `data/results/report_groq_demo_final.json`.
> Highlight the `rqi` score and the `system_grade`.

**Voiceover**:
"The run is done. Let's look at the report.
I developed a composite metric called the **RQI (RAG Quality Index)**. It condenses all those complex logs into a single health score, from 0 to 1.

Here, Groq scored a **[READ SCORE, e.g., 0.64]**.
But *why*?

*(Highlight the "justification" section in the output or JSON)*
My system generates a **Justification**. It tells us:
- 'Retrieval is the bottleneck'. (This means I need a better Embedding Model).
- 'Factuality is High'. (This means Llama-3 is good at following instructions).

This transforms raw numbers into **Actionable Engineering Insights**."

---

## ⚔️ 6. Head-to-Head Comparison (Gemini vs Groq)
**Visual Cue**: 
> Open `data/results/report_gemini_demo_result.json` on the right side (Split Screen).
> Keep groq report on left, gemini report on right.

**Voiceover**:
"Finally, this framework allows me to do objective A/B testing.
Here is **Llama-3 (Left)** vs **Gemini (Right)**.

We can see that Gemini has a higher **Reasoning Score**, but Llama-3 was faster.
Based on this data, I can confidently recommend which model we should use for production, rather than just guessing."

---

---

## 📉 7. Reporting & Visualization Value
**Visual Cue**: 
> Briefly show the `data/results/` folder with multiple JSONs.
> Explain how this JSON can be ingested by a dashboard (like Grafana or Streamlit).

**Voiceover**:
"The output isn't just text—it's **Structured Data**.
My framework saves every run as a JSON object.
This is crucial because it allows us to:
1.  **Track Quality Over Time**: We can plot these scores on a graph to see if a new PR degrades the model.
2.  **Automated Gatekeeping**: We can set a CI/CD rule: 'If RQI < 0.7, block the deployment.'
This turns AI evaluation from a 'feeling' into a **Business Process**."

---

## 🔮 8. Limitations & Future Roadmap
**Visual Cue**: 
> Show a slide titled "Roadmap" or just speak candidly to the camera.

**Voiceover**:
"Finally, I want to be honest about the current limitations.
1.  **Latency**: The detailed RAGAS evaluation takes about 60 seconds per query. For production, we would run this asynchronously, not in the hot path.
2.  **Synthetic Bias**: I'm currently using synthetic questions. The next step is to curate a 'Golden Dataset' of *real* user queries.

**Future Roadmap**:
- **Hybrid Search**: I plan to combine Vector Search (FAISS) with Keyword Search (BM25) to improve retrieval of acronyms and IDs.
- **Cross-Encoder Re-Ranking**: Adding a re-ranker step to boost precision.
- **Feedback Loop**: Integrating user thumbs-up/down actions back into the evaluation dataset."

---

## 🏁 9. Conclusion
**Visual Cue**: 
> Switch to your Webcam or a "Thank You" slide.

**Voiceover**:
"To summarize: I've built a system that doesn't just 'work', but **monitors itself**.
It's modular, automated, and ready to be integrated into our CI/CD pipeline.
This ensures that as we scale, our AI remains accurate and trustworthy.

Thank you. I'm happy to take any questions."
