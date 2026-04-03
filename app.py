import gradio as gr
from fastapi import FastAPI
from pydantic import BaseModel
from env import TraffixEnv
from models import Observation, StepResponse

app = FastAPI(title="TRAFFIX Environment API")
_env = TraffixEnv()

# ---- REST API Endpoints required by OpenEnv spec ----

class ResetRequest(BaseModel):
    task: str = "easy"

class StepRequest(BaseModel):
    action: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/reset", response_model=Observation)
def api_reset(req: ResetRequest):
    return _env.reset(req.task)

@app.post("/step", response_model=StepResponse)
def api_step(req: StepRequest):
    return _env.step(req.action)

@app.get("/state", response_model=Observation)
def api_state():
    return _env.state()

# ---- Gradio UI endpoints ----

initial_obs = _env.reset("easy")

def reset_env(task_id):
    obs = _env.reset(task_id)
    return (
        obs.north_queue, obs.south_queue, obs.east_queue, obs.west_queue,
        obs.signal_state, round(obs.avg_wait, 2), obs.violations, "0.00"
    )

def step_env(action):
    resp = _env.step(action)
    obs = resp.observation
    return (
        obs.north_queue, obs.south_queue, obs.east_queue, obs.west_queue,
        obs.signal_state, round(obs.avg_wait, 2), obs.violations, f"{resp.reward:.2f}"
    )

with gr.Blocks(title="TRAFFIX Visualization", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🚦 TRAFFIX — Intelligent Traffic Decision Environment")
    
    with gr.Row():
        task_dropdown = gr.Dropdown(choices=["easy", "medium", "hard"], value="easy", label="Select Task")
        reset_btn = gr.Button("🔄 Reset Environment", scale=0)
        
    with gr.Row():
        n_q = gr.Number(label="North Queue (🏎️)", value=initial_obs.north_queue, interactive=False)
        s_q = gr.Number(label="South Queue (🏎️)", value=initial_obs.south_queue, interactive=False)
        e_q = gr.Number(label="East Queue (🏎️)", value=initial_obs.east_queue, interactive=False)
        w_q = gr.Number(label="West Queue (🏎️)", value=initial_obs.west_queue, interactive=False)
        
    with gr.Row():
        signal = gr.Textbox(label="Current Signal State (🟢/🔴)", value=initial_obs.signal_state, interactive=False)
        avg_wait = gr.Number(label="Avg Wait Time", value=round(initial_obs.avg_wait, 2), interactive=False)
        violations = gr.Number(label="Violations 🛑", value=initial_obs.violations, interactive=False)
        reward = gr.Textbox(label="Last Step Reward (0-1)", value="0.00", interactive=False)
        
    with gr.Row():
        action_dropdown = gr.Dropdown(
            choices=["SWITCH_SIGNAL", "KEEP_SIGNAL", "PRIORITIZE_LANE_N", "PRIORITIZE_LANE_S", "PRIORITIZE_LANE_E", "PRIORITIZE_LANE_W"],
            value="KEEP_SIGNAL",
            label="Select Action"
        )
        step_btn = gr.Button("⚡ Take Action", variant="primary")
        
    # Wire events
    reset_outputs = [n_q, s_q, e_q, w_q, signal, avg_wait, violations, reward]
    reset_btn.click(fn=reset_env, inputs=task_dropdown, outputs=reset_outputs)
    step_btn.click(fn=step_env, inputs=action_dropdown, outputs=reset_outputs)

# Mount Gradio over FastAPI root (makes space fully compliant + UI)
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=7860)
