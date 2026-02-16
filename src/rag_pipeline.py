import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
# LCEL Imports
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
# LLMs
# Moved inside function
try:
    from src.embed_store import get_embeddings
except ImportError:
    from embed_store import get_embeddings

load_dotenv()

def get_llm(provider):
    if provider == "openai":
        try:
             from langchain_openai import ChatOpenAI
        except ImportError:
             raise ImportError("langchain-openai is not installed.")
             
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found")
        return ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=api_key, temperature=0)
    elif provider == "vertex":
        try:
            from langchain_google_vertexai import VertexAI
        except ImportError:
            raise ImportError("langchain-google-vertexai is not installed.")
            
        project = os.getenv("GCP_PROJECT")
        location = os.getenv("GCP_REGION")
        return VertexAI(model_name="gemini-pro", project=project, location=location, temperature=0)
    elif provider == "grok":
        try:
             from langchain_openai import ChatOpenAI
        except ImportError:
             raise ImportError("langchain-openai is not installed.")
             
        api_key = os.getenv("XAI_API_KEY")
        if not api_key:
            raise ValueError("XAI_API_KEY not found in .env")
        
        # Grok uses OpenAI-compatible API
        return ChatOpenAI(
            model="grok-beta", 
            openai_api_key=api_key, 
            openai_api_base="https://api.x.ai/v1",
            temperature=0
        )
    elif provider == "gemini":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError:
            raise ImportError("langchain-google-genai is not installed.")

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in .env")
            
        return ChatGoogleGenerativeAI(model="gemini-flash-latest", google_api_key=api_key, temperature=0)

    elif provider == "groq":
        try:
             from langchain_groq import ChatGroq
        except ImportError:
             raise ImportError("langchain-groq is not installed.")
             
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in .env")
            
        print(f"[DEBUG] Using GROQ_API_KEY ending in: ...{api_key[-4:]}")
        
        return ChatGroq(model_name="llama-3.3-70b-versatile", groq_api_key=api_key, temperature=0)

    else:
        raise ValueError(f"Unknown LLM provider: {provider}")

def rag(query, index_dir="data/faiss_index", top_k=3, embedding_provider="offline", llm_provider="openai", system_prompt=None, prompt_version="prompt_v0"):
    """
    End-to-end RAG function using LCEL. 
    Returns a dictionary with query, answer, and retrieved documents.
    """
    
    # 1. Load Embeddings & Index
    try:
        embedding_model = get_embeddings(embedding_provider)
        vectorstore = FAISS.load_local(index_dir, embedding_model, allow_dangerous_deserialization=True)
    except Exception as e:
        return {"error": f"Failed to load index/embeddings: {e}"}

    # 2. Setup Retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

    # 3. Setup LLM
    try:
        llm = get_llm(llm_provider)
    except Exception as e:
        return {"error": f"Failed to initialize LLM: {e}"}

    # 4. Define Prompt
    template = """Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer. 
    Use three sentences maximum and keep the answer concise.

    Context: {context}

    Question: {question}

    Helpful Answer:"""
    custom_rag_prompt = PromptTemplate.from_template(template)

    # Load environment variables with override to pick up new keys
    load_dotenv(override=True)
    
    # 3. Setup RAG Chain
    # ... (rest of the setup code remains similar, but we need to ensure local variables are refreshed)
    
    # Re-initialize LLM and Embeddings to ensure fresh API keys are used if env changed
    # (Checking if it's cheap to re-init. It is.)
    try:
        llm = get_llm(llm_provider)
    except Exception as e:
        return {"error": f"Failed to initialize LLM: {e}"}
    
    try:
        embeddings = get_embeddings(embedding_provider)
        vectorstore = FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        return {"error": f"Failed to load index/embeddings: {e}"}
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    
    # Check for Env Var override if argument is missing
    if system_prompt is None:
        system_prompt = os.getenv("RAG_SYSTEM_PROMPT")
        # Also try to grab prompt_version from env if possible for logging/metadata
        env_prompt_ver = os.getenv("RAG_PROMPT_VERSION")
        if env_prompt_ver:
            prompt_version = env_prompt_ver

    if system_prompt:
        from langchain_core.prompts import ChatPromptTemplate
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", """Answer the question based only on the following context:
            {context}

            Question: {question}""")
        ])
    else:
        template = """Answer the question based only on the following context:
        {context}

        Question: {question}
        """
        prompt = PromptTemplate.from_template(template)
        
    output_parser = StrOutputParser()

    # 5. Define Formatting Helper
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # 6. Create Chain (LCEL)
    # We want to return source documents, so we use RunnableParallel to keep 'context'
    
    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
        | prompt
        | llm
        | output_parser
    )

    rag_chain_with_source = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    ).assign(answer=rag_chain_from_docs)

    # 4. Invoke with Retry Logic for Rate Limits
    import time
    max_retries = 5
    base_delay = 20 # seconds
    
    print(f"Invoking RAG chain for query: '{query}'")
    
    final_result = None
    for attempt in range(max_retries):
        try:
            final_result = rag_chain_with_source.invoke(query)
            break # Break loop if successful
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "Rate limit" in error_msg or "quota" in error_msg.lower():
                wait_time = base_delay * (attempt + 1) + 60 # Aggressive backoff: 80s, 100s, 120s...
                print(f"\n[WARNING] Rate limit hit. Sleeping {wait_time}s before retry {attempt+1}/{max_retries}...")
                print(f"Error details: {error_msg[:200]}...") # Print snippet only
                time.sleep(wait_time)
            else:
                # If it's not a rate limit, raise immediately
                raise e
    
    if final_result is None:
        raise Exception(f"Failed to process query after {max_retries} retries due to rate limits.")
    
    # Format output
    return {
        "query": query,
        "answer": final_result["answer"],
        "retrieved_chunks": [doc.page_content for doc in final_result["context"]],
        "retrieved_metadata": [doc.metadata for doc in final_result["context"]],
        "prompt_version": prompt_version
    }

if __name__ == "__main__":
    # Simple CLI test
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("query", type=str)
    parser.add_argument("--embedding_provider", default="offline")
    parser.add_argument("--llm_provider", default="openai") # Default to OpenAI for now
    args = parser.parse_args()

    res = rag(args.query, embedding_provider=args.embedding_provider, llm_provider=args.llm_provider)
    print(res)
