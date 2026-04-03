# 🚦 TRAFFIX — Intelligent Traffic Decision Environment

TRAFFIX is a real-world OpenEnv environment where an AI agent learns to control a 4-way traffic junction in real time.

## 🚀 Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run Baseline Inference:
   ```bash
   export OPENAI_API_KEY="your-api-key"
   python inference.py
   ```

### 🔁 Reproducibility & Grading

TRAFFIX guarantees **100% deterministic reproducibility** for baseline task evaluation. The same agent logic against the same task seed will mathematically yield:
- **Exact same step sequences**
- **Exact same total rewards**
- **Exact same logs (`[START]`, `[STEP]`, `[END]` tags)**

Furthermore, explicit task evaluation is handled in `graders/` which hosts precise metric-based functions for `easy_grader.py`, `medium_grader.py`, and `hard_grader.py` ensuring mathematically fair scoring!

3. Run Interactive UI (Gradio):
   ```bash
   python app.py
   ```
   Open `http://localhost:7860` in your browser.

## 🧠 Tasks

- **🟢 Easy**: Green Wave Timing
- **🟡 Medium**: Congestion Reduction
- **🔴 Hard**: Full Optimization
