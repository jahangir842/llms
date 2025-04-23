# RAG Environment Setup

## 🐍 Conda Environment Setup

### 1. Create the environment
```bash
conda create -n rag-env python=3.10
```
Replace `rag-env` with your preferred environment name.

### 2. Activate the environment
```bash
conda activate rag-env
```

### 3. Install dependencies
Since some packages are not available on conda-forge, use pip inside the Conda environment:
```bash
pip install -r requirements.txt
```

### 4. Run your script
```bash
python rag.py
```

### 5. Deactivate the environment
```bash
conda deactivate
```

---

## 🚀 Working with Transformers

### ✅ Step 1: **Brush Up on Python (Quick & Targeted)**
You don’t need to master Python fully — just enough to build and run scripts.

- Learn basics like:
  - `functions`, `lists`, `dictionaries`
  - `import`, `for` loops, `if/else`, `classes`
- Practice with:  
  👉 [Python Basics on Kaggle](https://www.kaggle.com/learn/python)  
  👉 [W3Schools Python](https://www.w3schools.com/python/)

---

### ✅ Step 2: **Understand Transformers & LLMs (High Level First)**

Focus on:
- What is a Transformer? (attention, tokens, embeddings)
- What is an LLM (like LLaMA, GPT, etc.)?
- What is the difference between a model and a fine-tuned model?

📺 Watch:
- [Jay Alammar’s “Visual Guide to Transformers”](https://jalammar.github.io/illustrated-transformer/)
- [YouTube: LLMs for Beginners (by AssemblyAI)](https://youtu.be/qbIk7-JPB2c)

---

### ✅ Step 3: **Start Playing with Open-Source Models (Hands-on)**

Use **Hugging Face Transformers** — it’s the easiest way to use LLaMA, Mistral, DeepSeek, etc.

Install with:
```bash
pip install transformers accelerate
```

Example to use LLaMA (if you have GPU/Colab):
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

inputs = tokenizer("Who are you?", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))
```

Or try on **Hugging Face Spaces** (no coding needed!):
👉 https://huggingface.co/spaces

---

### ✅ Step 4: **Learn How Chatbots Work**

Modern chatbots like ChatGPT use:
- **LLMs** (like LLaMA)
- A **memory** of the conversation (chat history)
- Optional **RAG pipeline** (adds external knowledge from files/documents)

Try this hands-on tutorial:
👉 [LangChain Chatbot in Colab](https://github.com/hwchase17/langchain/blob/master/docs/docs/modules/chains/index.ipynb)

---

### ✅ Step 5: **RAG (Retrieval-Augmented Generation)**

RAG = chatbot that can answer based on your custom data.

It uses:
1. **Vector databases** (like ChromaDB, FAISS, Weaviate)
2. **Embeddings** (convert text to vectors)
3. **LLM** to generate the final response

Hands-on project:
👉 [LangChain RAG Tutorial](https://docs.langchain.com/docs/use-cases/question-answering/)

---

### ✅ Step 6: **Explore Open-Source LLMs**

Explore these:
- **LLaMA 2**: Meta’s open LLM
- **Mistral / Mixtral**: Powerful lightweight models
- **DeepSeek / Yi**: Chinese-English hybrid LLMs
- **Phi / TinyLlama**: Small but efficient models

Hosted models: https://huggingface.co/models?pipeline_tag=text-generation

---

## 🛠️ Recommended Tools/Libraries

| Tool            | Purpose                          |
|------------------|----------------------------------|
| `transformers`   | Load and use any open-source LLM |
| `langchain`      | Framework for RAG/chatbots       |
| `chromadb/faiss` | Vector storage for documents     |
| `gradio/streamlit` | Build simple chatbot UIs       |

---

## 🤖 Simple RAG Chatbot Project Idea

- Upload your own PDF/doc
- Convert it to chunks
- Embed chunks with `sentence-transformers`
- Store in FAISS or Chroma
- Use LLaMA to answer based on retrieval

