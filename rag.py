# Full RAG Example: Chatbot with Your Own Data

# Dependencies
# pip install langchain chromadb transformers sentence-transformers torch tqdm

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from tqdm import tqdm
import os
import sys

# Force CPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""
torch.set_num_threads(4)  # Limit CPU threads

try:
    # 1. Load your custom text file
    if not os.path.exists("my_data.txt"):
        with open("my_data.txt", "w") as f:
            f.write("This is a sample text file for testing RAG capabilities.")
        print("Created sample my_data.txt file")
    
    docs_path = "my_data.txt"
    loader = TextLoader(docs_path)
    documents = loader.load()
    print("✓ Loaded document")

    # 2. Split into smaller chunks
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    print("✓ Split document into chunks")

    # 3. Embed text chunks using sentence-transformers
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    vectorstore = Chroma.from_documents(chunks, embedding=embedding_model, persist_directory="db")
    print("✓ Created vector store")

    # 4. Load a smaller open-source LLM
    model_id = "microsoft/phi-2"  # Smaller model, ~2.7GB
    print(f"Loading model {model_id} on CPU...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # Load in 8bit mode to reduce memory usage
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map='cpu',
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )
    
    # Create a more focused prompt template
    prompt_template = """<|System|>You are a helpful AI assistant that gives clear and concise answers based on the provided context. Keep your answers focused and avoid redundancy.

<|Human|>Use this context to answer the question:
{context}

Question: {question}

<|Assistant|>"""

    llm_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=200,
        temperature=0,  # Deterministic output
        top_p=1,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    llm = HuggingFacePipeline(pipeline=llm_pipeline)
    print("✓ Loaded language model")

    # 5. Create the RAG chain with custom prompt
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # Simplest chain that stuffs all context into the prompt
        retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),  # Reduced number of retrieved docs
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": prompt_template
        }
    )
    print("✓ Created RAG chain")

    # 6. Ask a question
    query = "What is this document about?"
    print("\nQuestion:", query)
    result = qa_chain.invoke({"query": query})
    print("\nAnswer:", result["result"].strip())
    print("\nSources:", [doc.metadata for doc in result["source_documents"]])

except Exception as e:
    print(f"Error: {str(e)}", file=sys.stderr)
    sys.exit(1)
