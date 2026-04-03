import random
from models import Observation, StepResponse, Action
from config import TASK_SEEDS, MAX_WAIT, MAX_VIOLATIONS, MAX_STEPS

class TraffixEnv:
    def __init__(self):
        self.rng = random.Random()
        self.task_id = "easy"
        self.reset("easy")

    def reset(self, task_id: str) -> Observation:
        self.task_id = task_id
        seed = TASK_SEEDS.get(task_id, 42)
        self.rng.seed(seed)
        
        self.time_step = 0
        self.signal_state = "NS_GREEN"
        self.avg_wait = 0.0
        self.violations = 0
        self.total_wait_time = 0.0
        self.stable_steps = 0
        
        # Task specific queue initialization
        if task_id == "easy":
            # Easy: one lane heavy
            self.queues = {"N": 15, "S": 0, "E": 0, "W": 0}
        elif task_id == "medium":
            # Medium: all lanes have traffic
            self.queues = {"N": 5, "S": 5, "E": 5, "W": 5}
        elif task_id == "hard":
            # Hard: Heavy traffic
            self.queues = {"N": 10, "S": 10, "E": 8, "W": 8}
        else:
            self.queues = {"N": 5, "S": 5, "E": 5, "W": 5}
            
        return self._get_obs()

    def _get_obs(self) -> Observation:
        return Observation(
            north_queue=self.queues["N"],
            south_queue=self.queues["S"],
            east_queue=self.queues["E"],
            west_queue=self.queues["W"],
            signal_state=self.signal_state,
            avg_wait=self.avg_wait,
            violations=self.violations,
            time_step=self.time_step
        )

    def state(self) -> Observation:
        return self._get_obs()

    def step(self, action: str) -> StepResponse:
        self.time_step += 1
        penalty = 0.0
        
        # Action Validation Layer
        valid_actions = ["SWITCH_SIGNAL", "KEEP_SIGNAL", "PRIORITIZE_LANE_N", "PRIORITIZE_LANE_S", "PRIORITIZE_LANE_E", "PRIORITIZE_LANE_W"]
        if action not in valid_actions:
            penalty += 0.1
            action = "KEEP_SIGNAL" # default to a safe action
        
        # Process Action
        if action == "SWITCH_SIGNAL":
            if self.signal_state == "NS_GREEN":
                self.signal_state = "EW_GREEN"
            else:
                self.signal_state = "NS_GREEN"
            
            # small chance of violation when switching signals abruptly in hard mode
            if self.rng.random() < 0.2 and self.task_id == "hard":
                self.violations += 1
                
        elif action == "KEEP_SIGNAL":
            pass
            
        elif action.startswith("PRIORITIZE_LANE_"):
            lane = action.split("_")[-1]
            if (lane in ["N", "S"] and self.signal_state != "NS_GREEN") or \
               (lane in ["E", "W"] and self.signal_state != "EW_GREEN"):
                # Invalid prioritization (lane doesn't have greed light)
                penalty += 0.1
                # Can cause violation
                if self.rng.random() < 0.2:
                    self.violations += 1

        # Process traffic flows
        flow = 0
        if self.signal_state == "NS_GREEN":
            # Green flow
            cleared_n = min(self.queues["N"], self.rng.randint(2, 4))
            cleared_s = min(self.queues["S"], self.rng.randint(2, 4))
            self.queues["N"] -= cleared_n
            self.queues["S"] -= cleared_s
            flow += cleared_n + cleared_s
            
            # Red wait time increase
            waiting = self.queues["E"] + self.queues["W"]
            self.total_wait_time += waiting
            
        elif self.signal_state == "EW_GREEN":
            # Green flow
            cleared_e = min(self.queues["E"], self.rng.randint(2, 4))
            cleared_w = min(self.queues["W"], self.rng.randint(2, 4))
            self.queues["E"] -= cleared_e
            self.queues["W"] -= cleared_w
            flow += cleared_e + cleared_w
            
            # Red wait time increase
            waiting = self.queues["N"] + self.queues["S"]
            self.total_wait_time += waiting

        # Random arrivals
        arrival_rate = {"easy": 0.0, "medium": 0.3, "hard": 0.7}[self.task_id]
        for mode in ["N", "S", "E", "W"]:
            if self.rng.random() < arrival_rate:
                self.queues[mode] += self.rng.randint(1, 2)
                
        # Spontaneous violations in hard mode
        if self.task_id == "hard" and self.rng.random() < 0.1:
            self.violations += 1

        # Update running stats
        total_q = sum(self.queues.values())
        if total_q > 0:
            self.avg_wait = self.total_wait_time / max(1, self.time_step) # Simple avg

        # Reward Calculation as specified
        flow_score = 1.0 - min(self.avg_wait / MAX_WAIT, 1.0)
        violation_score = 1.0 - min(self.violations / MAX_VIOLATIONS, 1.0)
        
        raw_reward = (flow_score * 0.5) + (violation_score * 0.4) - penalty
        reward = max(0.0, min(1.0, raw_reward))

        # Check terminal
        done = False
        if self.time_step >= MAX_STEPS:
            done = True
        
        # stabilized -> early stop + bonus
        if total_q == 0:
            self.stable_steps += 1
            if self.stable_steps >= 3:
                done = True
                reward = 1.0
        else:
            self.stable_steps = 0

        obs = self._get_obs()
        return StepResponse(
            observation=obs,
            reward=reward,
            done=done,
            info={"flow": flow, "total_queue": total_q}
        )
