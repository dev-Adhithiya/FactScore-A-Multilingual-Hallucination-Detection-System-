# Hallucination Detection & Trust Scorer

A production-grade multilingual AI hallucination detection system that analyzes generated text, extracts atomic claims, retrieves evidence from a knowledge base, and produces per-claim trust scores with an interactive dashboard.

---

## Project Structure

```
hallucination-detection-trust-scorer/
│
├── core/                          # Core pipeline modules
│   ├── __init__.py
│   ├── generation.py              # LLM text generation (Phi-3, 4-bit quantized)
│   ├── claim_extraction.py        # Atomic claim extraction (spaCy + Stanza)
│   ├── retrieval.py               # Batch FAISS evidence retrieval (LaBSE)
│   ├── entailment.py              # Batch NLI entailment scoring (XLM-RoBERTa)
│   ├── drift.py                   # Cross-lingual semantic drift measurement
│   ├── scoring.py                 # Composite risk score computation
│   ├── pipeline.py                # Full pipeline orchestrator (batched)
│   └── model_server.py            # Persistent FastAPI model server
│
├── api/                           # REST API
│   ├── __init__.py
│   └── app.py                     # FastAPI endpoints (/analyze, /health)
│
├── dashboard/                     # Interactive UI
│   ├── __init__.py
│   └── dashboard.py               # Streamlit dashboard
│
├── scripts/                       # Knowledge base builders
│   ├── __init__.py
│   ├── build_kb.py                # General Wikipedia KB builder
│   └── build_fever_kb.py          # Targeted KB builder for Poly-FEVER evaluation
│
├── tests/                         # Tests and evaluation
│   ├── __init__.py
│   ├── test_system.py             # Unit tests (18 tests, no model loading)
│   └── test_poly_fever.py         # Accuracy evaluation on Poly-FEVER dataset
│
├── Poly-FEVER/                    # Dataset (not in git)
│   └── Poly-FEVER.tsv             # Multilingual fact verification dataset
│
├── data/                          # Generated at runtime (not in git)
│   └── knowledge_base/
│       ├── faiss.index            # FAISS vector index
│       └── passages.jsonl         # Knowledge base passages
│
├── logs/                          # Pipeline logs (not in git)
├── config.yaml                    # Central configuration
├── requirements.txt               # Python dependencies
├── Dockerfile                     # Container definition
├── docker-compose.yml             # Multi-service deployment
├── .gitignore
└── README.md
```

---

## How It Works

```
User Prompt
     │
     ▼
[1] LLM Generation (Phi-3-mini 4-bit)
     │  Generates factual response
     ▼
[2] Claim Extraction (spaCy / Stanza)
     │  Splits into atomic verifiable sentences
     ▼
[3] Cross-lingual Drift (LaBSE + Helsinki-NLP)
     │  Measures semantic shift across languages
     ▼
[4] Batch Evidence Retrieval (LaBSE + FAISS)
     │  Finds top-k supporting passages for ALL claims at once
     ▼
[5] Batch Entailment Scoring (XLM-RoBERTa-large-xnli)
     │  Scores ALL claim×passage pairs in one forward pass
     ▼
[6] Risk Score Computation
     │  Risk = 0.4×(1-Entailment) + 0.3×(1-Retrieval) + 0.3×Drift
     ▼
Dashboard / API Response
```

---

## Risk Score Formula

```
Risk = 0.4 × (1 - Entailment) + 0.3 × (1 - Retrieval) + 0.3 × Drift

Risk > 0.6  →  Hallucinated  🔴
Risk 0.4–0.6 →  Uncertain    🟡
Risk < 0.4  →  Reliable      🟢
```

---

## Tech Stack

| Component | Model / Library | Purpose |
|-----------|----------------|---------|
| Text Generation | microsoft/Phi-3-mini-4k-instruct (4-bit) | Generate factual responses |
| Embeddings | sentence-transformers/LaBSE | Multilingual semantic search |
| Entailment | joeddav/xlm-roberta-large-xnli | NLI-based claim verification |
| Translation | Helsinki-NLP/opus-mt-mul-en | Cross-lingual drift measurement |
| Vector Search | FAISS (GPU) | Fast evidence retrieval |
| NLP | spaCy + Stanza | Claim extraction & NER |
| Model Server | FastAPI + uvicorn | Persistent model hosting |
| Dashboard | Streamlit | Interactive UI |
| Quantization | bitsandbytes (4-bit NF4) | Fit models in 8GB VRAM |

---

## Supported Languages

English · Hindi · Tamil · Telugu · Bengali · Kannada · Marathi

---

## System Requirements

| Component | Minimum |
|-----------|---------|
| Python | 3.11 |
| RAM | 16 GB |
| GPU VRAM | 8 GB (NVIDIA) |
| Storage | 25–30 GB |
| CUDA | 11.8+ |

---

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/YourUsername/hallucination-detection-trust-scorer.git
cd hallucination-detection-trust-scorer
```

### 2. Create Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install PyTorch
```bash
# CUDA 12.1 (recommended)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# CPU only
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 5. Build Knowledge Base

**Option A — General Wikipedia (quick start):**
```bash
python scripts/build_kb.py --languages en hi ta --limit 1000
```

**Option B — Targeted KB for Poly-FEVER evaluation (recommended):**
```bash
python scripts/build_fever_kb.py --max_articles 5000
```
This fetches Wikipedia articles for the exact entities mentioned in Poly-FEVER claims, giving much better retrieval accuracy during evaluation.

---

## Running the System

### Step 1 — Start Model Server (Terminal 1)
```bash
python core/model_server.py
```
Wait until you see:
```
ALL MODELS LOADED. Server is ready.
```
Models load **once** and stay in memory. Do not close this terminal.

### Step 2 — Start Dashboard (Terminal 2)
```bash
python -m streamlit run dashboard/dashboard.py
```
Open **http://localhost:8501** in your browser.

---

## Running Tests

### Unit Tests (no models required, runs in ~5 seconds)
```bash
pytest tests/test_system.py -v
```

### Poly-FEVER Accuracy Evaluation (model server must be running)
```bash
# English claims
python tests/test_poly_fever.py --language en --max_samples 50

# Hindi claims
python tests/test_poly_fever.py --language hi --max_samples 50

# Tamil claims
python tests/test_poly_fever.py --language ta --max_samples 50
```

Available languages: `en`, `hi`, `ta`, `bn`, `ar`, `ja`, `ko`, `th`, `ka`, `am`

Results saved to `tests/poly_fever_results_{language}.json`.

---

## API Usage

```bash
curl -X POST http://localhost:8001/analyze \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain the causes of the Green Revolution in India.", "language": "English"}'
```

### Response Format
```json
{
  "generated_text": "...",
  "claims": [
    {
      "claim": "The Green Revolution began in the 1960s.",
      "risk_score": 0.21,
      "hallucinated": false,
      "entailment_score": 0.82,
      "retrieval_score": 0.74,
      "drift_score": 0.12,
      "evidence": [...]
    }
  ],
  "document_hallucination_rate": 0.25,
  "total_claims": 8,
  "hallucinated_claims": 2,
  "elapsed_sec": 28.3
}
```

---

## Performance (RTX 4060 Laptop 8GB, 16GB RAM)

| Stage | Time |
|-------|------|
| Model server startup (first time) | ~3-4 min |
| Model server startup (cached) | ~30-60s |
| Per-prompt analysis | ~20-30s |
| Knowledge base build (5000 articles) | ~20 min |

**Optimization applied:** Batch retrieval and batch entailment — all claims processed together in single forward passes instead of per-claim loops.

---

## Expected Accuracy on Poly-FEVER

| Knowledge Base | Accuracy |
|----------------|---------|
| 272 random articles | ~0% (no relevant evidence) |
| 5,000 targeted articles | ~55-65% |
| 50,000+ targeted articles | ~70-80% |

---

## Configuration

Edit `config.yaml` to customize models, thresholds, and paths:

```yaml
generation:
  model: microsoft/Phi-3-mini-4k-instruct
  temperature: 0.7
  max_new_tokens: 300

scoring:
  hallucination_threshold: 0.6   # Lower = stricter detection

retrieval:
  top_k: 5                       # Evidence passages per claim
```

---

## Docker Deployment

```bash
docker-compose up --build
```
- Dashboard: **http://localhost:8501**
- API: **http://localhost:8000**

---

## Notes

- Models are downloaded automatically from HuggingFace on first run (~6-8 GB total, cached after)
- Knowledge base accuracy depends on what articles were fetched — use `build_fever_kb.py` for Poly-FEVER evaluation
- A claim may be correct but flagged as hallucinated if the knowledge base lacks supporting evidence

---

## License

MIT License
