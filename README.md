# Hallucination Detection & Trust Scorer

A production-grade multilingual AI hallucination detection system that analyzes generated text, extracts atomic claims, retrieves evidence, and produces per-claim trust scores with an interactive dashboard.

---

## What It Does

1. Takes a prompt and generates a factual response using an LLM
2. Extracts individual verifiable claims from the response
3. Retrieves supporting evidence from a Wikipedia-based knowledge base
4. Scores each claim using entailment, retrieval similarity, and cross-lingual drift
5. Displays results in an interactive dashboard with color-coded highlights

---

## Risk Score Formula

```
Risk = 0.4 × (1 - Entailment) + 0.3 × (1 - Retrieval) + 0.3 × Drift

Risk > 0.6 → Hallucinated 🔴
Risk 0.4–0.6 → Uncertain  🟡
Risk < 0.4 → Reliable     🟢
```

---

## Tech Stack

| Component | Model / Library |
|-----------|----------------|
| Text Generation | microsoft/Phi-3-mini-4k-instruct |
| Embeddings | sentence-transformers/LaBSE |
| Entailment (NLI) | joeddav/xlm-roberta-large-xnli |
| Translation | Helsinki-NLP/opus-mt-mul-en |
| Vector Search | FAISS |
| NLP | spaCy + Stanza |
| API | FastAPI |
| Dashboard | Streamlit |

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

### 3. Install PyTorch (match your CUDA version)
```bash
# CUDA 12.1
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
Downloads Wikipedia articles, embeds them, and builds the FAISS search index.
```bash
python scripts/build_kb.py --languages en hi ta --limit 1000
```
- `--languages` : Wikipedia language codes (en, hi, ta, te, bn, etc.)
- `--limit` : Number of articles per language (start with 500–1000)

---

## Running the System

### Start the Dashboard (Direct Mode)
```bash
python -m streamlit run dashboard/dashboard.py
```
Open **http://localhost:8501** in your browser.

### Start the API Server (Optional)
```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
```
API docs available at **http://localhost:8000/docs**

---

## API Usage

```bash
curl -X POST http://localhost:8000/analyze \
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

## Docker Deployment

```bash
docker-compose up --build
```
- Dashboard: **http://localhost:8501**
- API: **http://localhost:8000**

---

## Project Structure

```
hallucination-detection-trust-scorer/
├── core/
│   ├── generation.py        # LLM text generation
│   ├── claim_extraction.py  # Atomic claim extraction
│   ├── retrieval.py         # FAISS evidence retrieval
│   ├── entailment.py        # NLI entailment scoring
│   ├── drift.py             # Cross-lingual drift
│   ├── scoring.py           # Risk score computation
│   └── pipeline.py          # Pipeline orchestrator
├── api/
│   └── app.py               # FastAPI server
├── dashboard/
│   └── dashboard.py         # Streamlit dashboard
├── scripts/
│   └── build_kb.py          # Knowledge base builder
├── tests/
│   └── test_system.py       # Unit tests
├── config.yaml              # Configuration
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

---

## Configuration

Edit `config.yaml` to customize:

```yaml
generation:
  model: microsoft/Phi-3-mini-4k-instruct
  temperature: 0.7
  max_new_tokens: 300

scoring:
  hallucination_threshold: 0.6   # Lower = stricter

retrieval:
  top_k: 5   # Evidence passages per claim
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Performance (RTX 4060 Laptop, 16GB RAM)

| Stage | Time |
|-------|------|
| First prompt (model loading) | ~60 sec |
| Subsequent prompts | ~20–30 sec |
| Knowledge base build (1000 articles) | ~10 min |

---

## Notes

- First run downloads models (~5–8 GB total). Cached after that.
- Knowledge base only contains what was fetched during `build_kb.py`. Add custom passages to `data/knowledge_base/passages.jsonl` for better coverage on specific topics.
- The system detects hallucinations based on retrieved evidence — a claim may be correct but flagged if the knowledge base lacks supporting passages.

---

## License

MIT License
