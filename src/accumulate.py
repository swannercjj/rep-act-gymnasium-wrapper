import gymnasium as gym
# import numpy as np    

class AccumulateSteps(gym.Wrapper):
    """This wrapper allows multiple steps to be taken at once, 
    accumulating all reward into the final step. 
    """
    def __init__(self, env: gym.Env, batch_key="batch"):
        super().__init__(env)
        self._batch_key = batch_key

    def step(self, action, repeat=1):
        obs = None
        acc_reward = 0
        terminated = False
        truncated = False
        info = {}
        prev_steps = []
        num_steps = 0

        for _ in range(repeat):
            obs, reward, terminated, truncated, info = super().step(action)
            acc_reward += reward
            num_steps += 1

            if terminated or truncated:
                break

            prev_steps.append((obs, reward))
        
        assert self._batch_key not in info

        info[self._batch_key] = {
            "prev_steps": prev_steps,
            "num_steps": num_steps
        }

        return obs, acc_reward, terminated, truncated, info
