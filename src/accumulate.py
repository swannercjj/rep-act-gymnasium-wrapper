import gymnasium as gym

class AccumulateSteps(gym.Wrapper):
    """This wrapper allows multiple steps to be taken at once, 
    accumulating all reward into the final step. 
    """
    def __init__(self, env: gym.Env, info_key="repeats"):
        super().__init__(env)
        self._info_key = info_key

    def step(self, action, repeat=4):
        acc_reward = 0
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

        info[self._info_key] = {
            "prev_steps": prev_steps,
            "num_steps": num_steps
        }

        return obs, acc_reward, terminated, truncated, info


if __name__=="__main__":
    env = gym.make("Pendulum-v1")
    env = AccumulateSteps(env)

    observation, info = env.reset()
    repeat = 4

    for _ in range(10):
        action = env.action_space.sample()  # agent policy that uses the observation and info
        observation, reward, terminated, truncated, info = env.step(action, repeat)


        acc_reward = 0

        for i in range(repeat):
            acc_reward += info['intermediate']['prev_steps'][i][1]
        
        assert acc_reward == reward
        print(observation, info["intermediate"]['prev_steps'][repeat - 1][0])

        if terminated or truncated:
            observation, info = env.reset()
        

    env.close()
