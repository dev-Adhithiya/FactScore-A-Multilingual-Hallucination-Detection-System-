"""
dashboard/dashboard.py - Streamlit interactive dashboard for.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import requests
import json

@st.cache_resource
def get_cached_pipeline(config_path: str = "config.yaml"):
    """
    Load pipeline once and cache it permanently in memory.
    Models will never reload as long as the dashboard is running.
    """
    from core.pipeline import Pipeline
    p = Pipeline(config_path)
    return p.run_pipeline
# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hallucination Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}

.main-title {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #FF4E50, #FC913A, #F9D423);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0;
}

.subtitle {
    font-size: 1rem;
    color: #888;
    margin-top: 0;
    font-family: 'JetBrains Mono', monospace;
}

.claim-card {
    border-radius: 10px;
    padding: 16px 20px;
    margin: 8px 0;
    border-left: 5px solid;
    font-size: 0.95rem;
    line-height: 1.6;
}

.claim-safe {
    background: #0f2818;
    border-color: #2ecc71;
    color: #a8f5c6;
}

.claim-warn {
    background: #2b2200;
    border-color: #f39c12;
    color: #ffe49a;
}

.claim-danger {
    background: #2b0a0a;
    border-color: #e74c3c;
    color: #ffa7a7;
}

.metric-box {
    background: #111;
    border: 1px solid #333;
    border-radius: 10px;
    padding: 20px;
    text-align: center;
}

.risk-badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    padding: 2px 8px;
    border-radius: 20px;
    font-weight: 600;
}

.badge-safe   { background: #1a4a2e; color: #2ecc71; }
.badge-warn   { background: #4a3800; color: #f39c12; }
.badge-danger { background: #4a1010; color: #e74c3c; }

.evidence-box {
    background: #0d0d0d;
    border: 1px solid #2a2a2a;
    border-radius: 6px;
    padding: 10px 14px;
    font-size: 0.85rem;
    color: #aaa;
    font-family: 'JetBrains Mono', monospace;
    margin-top: 6px;
}

.section-header {
    font-size: 1.1rem;
    font-weight: 700;
    color: #eee;
    border-bottom: 1px solid #333;
    padding-bottom: 6px;
    margin: 18px 0 10px 0;
}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
col_logo, col_title = st.columns([1, 8])
with col_title:
    st.markdown('<p class="main-title">FactScore: A Multilingual Hallucination Detection System</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">// Multilingual Hallucination Detection & Reliability Scoring</p>', unsafe_allow_html=True)

st.markdown("---")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    mode = st.radio("Mode", ["API Mode", "Direct Mode"], index=1,
                    help="API Mode calls the FastAPI server. Direct Mode runs pipeline in-process.")

    if mode == "API Mode":
        api_url = st.text_input("API Base URL", value="http://localhost:8000")
    else:
        config_path = st.text_input("Config Path", value="config.yaml")

    st.markdown("---")
    st.markdown("### 🌐 Languages")
    st.markdown("English · Hindi · Tamil · Telugu · Bengali · Kannada · Marathi")

    st.markdown("---")
    st.markdown("### 📊 Risk Thresholds")
    st.markdown("🟢 **Safe** — Risk < 0.4")
    st.markdown("🟡 **Warn** — 0.4 ≤ Risk < 0.6")
    st.markdown("🔴 **Hallucinated** — Risk ≥ 0.6")

# ── Input Form ────────────────────────────────────────────────────────────────
st.markdown('<p class="section-header">📝 Input</p>', unsafe_allow_html=True)

col_lang, col_prompt = st.columns([1, 4])
with col_lang:
    language = st.selectbox(
        "Language",
        ["English", "Hindi", "Tamil", "Telugu", "Bengali", "Kannada", "Marathi"],
    )
with col_prompt:
    prompt = st.text_area(
        "Prompt",
        placeholder="Explain the causes of the Green Revolution in India.",
        height=100,
    )

run_btn = st.button("🚀 Analyze", type="primary", use_container_width=True)

# ── Run Pipeline ──────────────────────────────────────────────────────────────
result = None

if run_btn and prompt.strip():
    with st.spinner("Running hallucination detection pipeline..."):
        if mode == "API Mode":
            try:
                resp = requests.post(
                    f"{api_url}/analyze",
                    json={"prompt": prompt, "language": language},
                    timeout=120,
                )
                resp.raise_for_status()
                result = resp.json()
            except requests.exceptions.ConnectionError:
                st.error(f"Cannot reach API at {api_url}. Is the FastAPI server running?")
            except Exception as e:
                st.error(f"API error: {e}")
        else:
            try:
                result = get_cached_pipeline(config_path)(prompt, language)
            except Exception as e:
                st.error(f"Pipeline error: {e}")
                import traceback
                st.code(traceback.format_exc())

elif run_btn:
    st.warning("Please enter a prompt.")

# ── Results ───────────────────────────────────────────────────────────────────
if result:
    claims = result.get("claims", [])
    doc_rate = result.get("document_hallucination_rate", 0.0)
    n_total = result.get("total_claims", len(claims))
    n_hallucinated = result.get("hallucinated_claims", sum(1 for c in claims if c["hallucinated"]))
    elapsed = result.get("elapsed_sec", "-")

    # ── KPI Row ───────────────────────────────────────────────────────────
    st.markdown('<p class="section-header">📊 Summary</p>', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)

    with k1:
        rate_color = "#e74c3c" if doc_rate > 0.5 else "#f39c12" if doc_rate > 0.25 else "#2ecc71"
        st.metric("Hallucination Rate", f"{doc_rate:.1%}")

    with k2:
        st.metric("Total Claims", n_total)

    with k3:
        st.metric("Hallucinated Claims", n_hallucinated)

    with k4:
        st.metric("Analysis Time", f"{elapsed}s")

    # ── Generated Text ────────────────────────────────────────────────────
    st.markdown('<p class="section-header">📄 Generated Text</p>', unsafe_allow_html=True)
    gen_text = result.get("generated_text", "")

    # Highlight hallucinated sentences in generated text
    highlighted = gen_text
    for claim in claims:
        if claim["hallucinated"]:
            highlighted = highlighted.replace(
                claim["claim"],
                f'<span style="background:#4a1010; color:#ffa7a7; border-radius:3px; padding:1px 2px;">{claim["claim"]}</span>'
            )
        else:
            highlighted = highlighted.replace(
                claim["claim"],
                f'<span style="background:#0f2818; color:#a8f5c6; border-radius:3px; padding:1px 2px;">{claim["claim"]}</span>'
            )

    st.markdown(
        f'<div style="background:#111; border:1px solid #333; border-radius:10px; padding:20px; line-height:1.8; font-size:1rem;">{highlighted}</div>',
        unsafe_allow_html=True,
    )

    # ── Charts ────────────────────────────────────────────────────────────
    if claims:
        st.markdown('<p class="section-header">📈 Risk Analysis</p>', unsafe_allow_html=True)
        chart_col1, chart_col2 = st.columns(2)

        df = pd.DataFrame([
            {
                "Claim": c["claim"][:60] + "..." if len(c["claim"]) > 60 else c["claim"],
                "Risk Score": c["risk_score"],
                "Entailment": c["entailment_score"],
                "Retrieval": c["retrieval_score"],
                "Drift": c["drift_score"],
                "Status": "Hallucinated" if c["hallucinated"] else "Reliable",
            }
            for c in claims
        ])

        with chart_col1:
            st.markdown("**Risk Scores per Claim**")
            risk_df = df[["Claim", "Risk Score"]].set_index("Claim")
            st.bar_chart(risk_df)

        with chart_col2:
            st.markdown("**Score Breakdown**")
            score_df = df[["Claim", "Entailment", "Retrieval", "Drift"]].set_index("Claim")
            st.bar_chart(score_df)

        # Scatter: Entailment vs Retrieval
        st.markdown("**Entailment vs Retrieval Score** (bubble = risk)")
        scatter_df = pd.DataFrame({
            "Entailment Score": [c["entailment_score"] for c in claims],
            "Retrieval Score": [c["retrieval_score"] for c in claims],
        })
        st.scatter_chart(scatter_df, x="Entailment Score", y="Retrieval Score")

    # ── Per-Claim Detail ──────────────────────────────────────────────────
    st.markdown('<p class="section-header">🔍 Claim Detail</p>', unsafe_allow_html=True)

    for i, claim in enumerate(claims):
        risk = claim["risk_score"]
        hallucinated = claim["hallucinated"]

        if risk >= 0.6:
            card_class = "claim-danger"
            badge = '<span class="risk-badge badge-danger">HALLUCINATED</span>'
        elif risk >= 0.4:
            card_class = "claim-warn"
            badge = '<span class="risk-badge badge-warn">UNCERTAIN</span>'
        else:
            card_class = "claim-safe"
            badge = '<span class="risk-badge badge-safe">RELIABLE</span>'

        with st.expander(f"Claim {i+1}: {claim['claim'][:80]}{'...' if len(claim['claim'])>80 else ''}", expanded=False):
            st.markdown(
                f"""<div class="claim-card {card_class}">
                {badge}&nbsp;&nbsp;<strong>Risk: {risk:.3f}</strong><br><br>
                {claim['claim']}
                </div>""",
                unsafe_allow_html=True,
            )

            score_c1, score_c2, score_c3 = st.columns(3)
            score_c1.metric("Entailment", f"{claim['entailment_score']:.3f}", delta=None)
            score_c2.metric("Retrieval", f"{claim['retrieval_score']:.3f}", delta=None)
            score_c3.metric("Drift", f"{claim['drift_score']:.3f}", delta=None)

            if claim.get("entities"):
                st.markdown(f"**Entities:** {', '.join(claim['entities'])}")
            if claim.get("numbers"):
                st.markdown(f"**Numbers:** {', '.join(claim['numbers'])}")

            # Evidence passages
            evidence = claim.get("evidence", [])
            if evidence:
                st.markdown("**Top Evidence Passages:**")
                for j, ev in enumerate(evidence[:3]):
                    sim = ev.get("similarity", 0)
                    src = ev.get("source", "Unknown")
                    lang = ev.get("language", "")
                    text = ev.get("text", "")[:300]
                    st.markdown(
                        f'<div class="evidence-box"><strong>#{j+1}</strong> | '
                        f'Source: {src} | Lang: {lang} | Similarity: {sim:.3f}<br><br>{text}…</div>',
                        unsafe_allow_html=True,
                    )

    # ── Raw JSON export ───────────────────────────────────────────────────
    st.markdown('<p class="section-header">⬇️ Export</p>', unsafe_allow_html=True)
    st.download_button(
        "Download Full Result JSON",
        data=json.dumps(result, indent=2, ensure_ascii=False),
        file_name="satya_vaani_result.json",
        mime="application/json",
    )

# ── Empty state ───────────────────────────────────────────────────────────────
elif not run_btn:
    st.markdown("""
    <div style="text-align:center; padding: 60px 20px; color: #555;">
        <div style="font-size: 3rem;">🔍</div>
        <div style="font-size: 1.2rem; margin-top:10px;">Enter a prompt above and click <strong>Analyze</strong> to begin.</div>
        <div style="font-size:0.9rem; margin-top:8px; font-family: monospace;">
            Multilingual · Claim-level scoring · Evidence retrieval · Cross-lingual drift
        </div>
    </div>
    """, unsafe_allow_html=True)
