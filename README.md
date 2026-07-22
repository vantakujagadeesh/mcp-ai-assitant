<p align="center">
  <img src="architecture.svg" alt="MCP AI Assistant Architecture" width="850"/>
</p>

<h1 align="center">MCP-Powered AI Assistant</h1>
<p align="center">
  <strong>Fine-Tuned Reasoning Core + RAG + n8n Workflow Automation</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue?logo=python" alt="Python 3.9+"/>
  <img src="https://img.shields.io/badge/Framework-FastAPI-009688?logo=fastapi" alt="FastAPI"/>
  <img src="https://img.shields.io/badge/LLM-Llama%203.1%208B-FF6F00?logo=meta" alt="Llama 3.1 8B"/>
  <img src="https://img.shields.io/badge/Fine--Tuning-QLoRA%20%7C%20Unsloth-FF6F00" alt="QLoRA + Unsloth"/>
  <img src="https://img.shields.io/badge/RAG-Sentence%20Transformers-FF6F00?logo=python" alt="RAG"/>
  <img src="https://img.shields.io/badge/Automation-n8n-green?logo=n8n" alt="n8n"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License MIT"/>
</p>

---

## 📋 Overview

An end-to-end AI system where a **QLoRA-fine-tuned LLM** acts as the reasoning core, retrieves context via **RAG**, exposes tools via an **MCP-style REST API**, and automates real workflows via **n8n**. A closed-loop, production-ready system with feedback capture for continuous improvement.

### What It Demonstrates

| Skill | What This Project Shows |
|-------|------------------------|
| **LLM Integration / Agent Building** | RAG pipeline + MCP tool orchestration |
| **ML Engineering** | QLoRA fine-tuning, dataset curation, eval |
| **Production Operations** | n8n workflow automation, feedback loops, logging |

---

## 🏗️ Architecture

```
User query ──► MCP Assistant (FastAPI REST API)
                  │
                  ├──► RAG Retriever (semantic / keyword search)
                  │       └─ data/knowledge_base/support_articles.jsonl
                  │
                  ├──► Fine-tuned LLM (QLoRA, 4-bit, domain-specialized)
                  │       └─ reasons + generates response
                  │
                  ├──► n8n triggers resulting action
                  │       ├─ Send email
                  │       ├─ Post Slack notification
                  │       ├─ Update ticket
                  │       └─ Log to spreadsheet/DB
                  │
                  └──► Feedback (👍/👎) captured
                          └─ stored in data/feedback/ for future re-tuning
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+** (3.9 works too)
- **n8n instance** (self-hosted or [n8n.cloud](https://n8n.cloud))
- **Google Colab account** (for fine-tuning — free tier works)

### 1. Clone & Setup

```bash
git clone <your-repo-url>
cd mcp-assistant

# Create virtual environment
python -m venv venv
source venv/bin/activate   # Linux/Mac
# venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the REST API Server

```bash
python src/mcp_server/rest_api.py
```

The server starts at `http://0.0.0.0:8000` with **6 endpoints**:

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check — model & RAG status |
| `POST` | `/api/classify_ticket` | Classify ticket → category + urgency |
| `POST` | `/api/search_kb` | RAG search over knowledge base |
| `POST` | `/api/generate_response` | Generate a support response |
| `POST` | `/api/resolve_ticket` | Resolve / summarize a ticket |
| `POST` | `/api/feedback_log` | Log 👍/👎 feedback for re-tuning |

### 3. Test the Endpoints

```bash
# Run with FastAPI TestClient (no server needed)
python test_endpoints.py

# Or start server and test with real HTTP
python live_test.py
```

> **Note:** The system works **without a GPU** — it gracefully falls back to keyword-based classification and template responses when no model is loaded.

### 4. Fine-Tune the Model (Colab)

Open and run the Colab notebook:

```bash
notebooks/finetune_llama_support.ipynb
```

This will:
1. Download 3 customer support datasets (45k+ samples)
2. Format them for instruction tuning
3. QLoRA fine-tune Llama 3.1 8B using Unsloth
4. Save adapter weights to Hugging Face / Google Drive

### 5. Deploy n8n Workflows

Import the workflow templates from `n8n_workflows/` into your n8n instance:

| Workflow | Description |
|----------|-------------|
| `ticket_workflow.json` | Ticket pipeline: webhook → classify → KB → respond |
| `ticket_workflow_slack.json` | Same + Slack notification |
| `ticket_workflow_email.json` | Same + email notification |
| `feedback_loop.json` | Smart feedback capture (only logs 👍/👎) |

---

## 📁 Project Structure

```
├── configs/                        # Configuration files
│   ├── training.yaml               #   QLoRA fine-tuning config
│   └── mcp_server.yaml             #   Server config (model, RAG)
│
├── data/
│   ├── knowledge_base/             #  15 seed support articles (JSONL)
│   ├── feedback/                   #  Captured feedback (gitignored)
│   ├── processed/                  #  Cleaned datasets (gitignored)
│   └── raw/                        #  Original downloads (gitignored)
│
├── notebooks/
│   └── finetune_llama_support.ipynb  # Colab fine-tuning notebook
│
├── src/
│   ├── mcp_server/
│   │   ├── rest_api.py             #  FastAPI HTTP endpoints (6 routes)
│   │   └── server.py               #  SupportAssistant (model wrapper)
│   ├── rag/
│   │   └── retriever.py            #  Knowledge base retriever (embeddings + keyword)
│   ├── finetune/
│   │   ├── prepare_data.py         #  Dataset loading & formatting
│   │   └── train.py                #  QLoRA training script
│   └── eval/
│       ├── benchmark.py            #  Base vs. fine-tuned comparison
│       └── metrics.py              #  ROUGE-L, BERTScore, accuracy
│
├── n8n_workflows/                  # Pre-built n8n workflow templates
│   ├── ticket_workflow.json
│   ├── ticket_workflow_slack.json
│   ├── ticket_workflow_email.json
│   └── feedback_loop.json
│
├── results/                        # Output artifacts (gitignored)
│   ├── models/                     #   Adapter weights
│   ├── benchmarks/                 #   Eval results
│   └── figures/                    #   Architecture diagrams
│
├── test_endpoints.py               # Test suite (FastAPI TestClient)
├── live_test.py                    # Live integration test
├── requirements.txt                # Python dependencies
├── .gitignore
├── LICENSE
└── README.md
```

---

## 🔬 Dataset Strategy

This project combines **3 datasets** for dual-task capability (classification + response generation):

| Dataset | Use | Size | License |
|---------|-----|------|---------|
| [Bitext Customer Support](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset) | Response generation | 30k+ conversations | Apache 2.0 |
| [ServiceNow Incidents](https://huggingface.co/datasets/6StringNinja/synthetic-servicenow-incidents) (synthetic) | Category + urgency classification | 10k+ incidents | MIT |
| [TriageIQ](https://huggingface.co/datasets/coldstart88/triageiq-dataset) | Sentiment + urgency + category | 5k+ samples | Apache 2.0 |

All data is unified into an instruction format:

```
System: You are a customer support assistant.
        Classify the ticket and generate a response.

User: Ticket: [ticket description]

Assistant: Category: [category]
           Urgency: [urgency]
           Response: [response]
```

---

## 🧠 Fine-Tuning Configuration

| Parameter | Value |
|-----------|-------|
| Base model | Llama 3.1 8B (or Qwen2.5 7B) |
| Method | QLoRA via Unsloth |
| Quantization | 4-bit NF4 |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Batch size | 2 (gradient accumulation 4) |
| Learning rate | 2e-4 |
| Max sequence length | 2048 |
| Training steps | 500–1000 |
| Hardware | Colab T4 (16GB VRAM) |

---

## 📊 Evaluation

Run the benchmark to compare base vs. fine-tuned model:

```bash
python src/eval/benchmark.py \
    --base-model unsloth/Llama-3.1-8B-Instruct-bnb-4bit \
    --finetuned-model ./llama-support-lora
```

**Metrics measured:**
- **Classification accuracy** (%)
- **Response quality** (ROUGE-L, BERTScore)
- **Inference latency** (ms)
- **Memory usage** (GB)
- **Cost per 1K tokens** ($)

---

## 🔄 Feedback Loop

The system captures 👍/👎 feedback from users and **selectively logs** it:

- **Logs only** when rating is `thumbs_down` OR confidence < 0.5
- **Skips** high-confidence `thumbs_up` results (not valuable for retraining)

All feedback is stored in `data/feedback/feedback_log.jsonl` and is ready for:
- Future DPO alignment training
- Low-confidence monitoring
- Performance analytics

---

## ✅ Deliverables

### Completed

- ✅ Project scaffold (structure, configs, README)
- ✅ Colab fine-tuning notebook (ready to run)
- ✅ Dataset preparation scripts
- ✅ FastAPI REST server (6 endpoints)
- ✅ RAG pipeline (embeddings + keyword fallback)
- ✅ n8n workflow templates (4 variants)
- ✅ Evaluation framework (benchmark + metrics)
- ✅ 15 seed knowledge base articles
- ✅ Keyword fallback (works without GPU)

### Yet to Do

- [ ] Run fine-tuning on Colab → push adapter to Hugging Face
- [ ] Run benchmark → fill in accuracy/latency numbers
- [ ] Generate architecture diagram PNG
- [ ] Wire fine-tuned model into MCP + n8n
- [ ] Capture feedback loop data
- [ ] Record demo video/GIF
- [ ] Publish Hugging Face model card

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  Built with FastAPI, Unsloth, n8n, and ❤️
</p>
