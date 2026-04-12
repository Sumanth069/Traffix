import os
import json
from openai import OpenAI
from env import TraffixEnv
from models import Action
from config import TASK_SEEDS

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: str) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

def run_inference():
    api_base = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
    model_name = os.environ.get("MODEL_NAME", "gpt-4o-mini")
    hf_token = os.environ.get("HF_TOKEN")
    
    client = OpenAI(
        base_url=api_base,
        api_key=hf_token
    )
    
    env_instance = TraffixEnv()
    tasks = list(TASK_SEEDS.keys())
    
    for task_id in tasks:
        log_start(task=task_id, env="traffix", model=model_name)
        obs = env_instance.reset(task_id)
        
        done = False
        step_count = 0
        rewards = []
        
        while not done:
            step_count += 1
            error_msg = None
            
            prompt = f"""
            You are a traffic signal control agent. Select the best action to minimize wait times and violations.
            Current Observation:
            {obs.model_dump_json()}
            
            Actions available:
            "SWITCH_SIGNAL": Switch the light from NS to EW, or EW to NS.
            "KEEP_SIGNAL": Keep the light as is.
            "PRIORITIZE_LANE_N": Prioritize North lane flow if NS is green.
            "PRIORITIZE_LANE_S": Prioritize South lane flow if NS is green.
            "PRIORITIZE_LANE_E": Prioritize East lane flow if EW is green.
            "PRIORITIZE_LANE_W": Prioritize West lane flow if EW is green.
            
            Respond only with the exact string of the chosen action.
            """
            
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0
                )
                action_str = response.choices[0].message.content.strip().strip('"\'')
                
                valid_actions = ["SWITCH_SIGNAL", "KEEP_SIGNAL", "PRIORITIZE_LANE_N", 
                               "PRIORITIZE_LANE_S", "PRIORITIZE_LANE_E", "PRIORITIZE_LANE_W"]
                if action_str not in valid_actions:
                    action_str = "KEEP_SIGNAL"
                    
            except Exception as e:
                action_str = "KEEP_SIGNAL"
                error_msg = str(e).replace('\n', ' ')
                
            action: Action = action_str
            
            try:
                step_resp = env_instance.step(action)
                obs = step_resp.observation
                reward = step_resp.reward
                done = step_resp.done
            except Exception as e:
                error_msg = str(e).replace('\n', ' ')
                reward = 0.0
                done = True
                
            rewards.append(reward)
            
            log_step(step=step_count, action=action_str, reward=reward, done=done, error=error_msg)
            
        score_val = min(0.99, max(0.01, sum(rewards) / max(1, step_count)))
        success = score_val > 0.5 
        log_end(success=success, steps=step_count, score=score_val, rewards=rewards)

if __name__ == "__main__":
    run_inference()
