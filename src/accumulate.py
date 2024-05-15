import gymnasium as gym
# from gymnasium.vector.vector_env import ArrayType, VectorEnv, VectorWrapper

class RepeatAction(gym.Wrapper):
    """This wrapper allows multiple steps to be taken at once, 
    accumulating all reward into the final step. 
    """
    def __init__(self, env: gym.Env, repeats:list, info_key="repeats"):
        # include possible repeats by modifying the environment
        # gymnasium.spaces.Tuple
        super().__init__(env)
        self.env.action_space.n *= len(repeats)
        self.repeats = repeats
        self._info_key = info_key

    def step(self, action):
        acc_reward = 0
        info = {}
        prev_steps = []
        num_steps = 0

        # decode the action, to get the actual action and repeat number
        # actual action = action % repeat space, repeat = action / repeat space?

        # [a0 with r1 repeats, a0 with r2 repeats, a1 with r1 repeats, a1 with r2 repeats, ...]
        repeat = self.repeats[action % len(self.repeats)]
        action = action // len(self.repeats)

        # print(action, repeat)

        for _ in range(repeat):
            obs, reward, terminated, truncated, info = super().step(action)
            acc_reward += reward
            num_steps += 1

            prev_steps.append((obs, reward))

            if terminated or truncated:
                break
        
        assert self._info_key not in info

        info[self._info_key] = {
            "prev_steps": prev_steps,
            "num_steps": num_steps
        }

        return obs, acc_reward, terminated, truncated, info


# class RepeatActionVector(VectorWrapper):
#     def __init__(self, env: VectorEnv, info_key="repeats"):
#         super().__init__(env)
#         self._info_key = info_key

#     def step(self, actions, repeat=4):
#         (
#             observations,
#             rewards, 
#             terminations, 
#             truncations, 
#             infos
#         ) = super().step(actions)

#         acc_rewards = []
#         info = {}
#         prev_steps = []
#         num_steps = []

        # for _ in range(repeat):
            


if __name__=="__main__":
    env = gym.make("MountainCar-v0")
    env = RepeatAction(env, repeats=[1, 4])

    observation, info = env.reset()
    # repeat = 4

    for _ in range(10):
        action = env.action_space.sample()  # agent policy that uses the observation and info
        observation, reward, terminated, truncated, info = env.step(action)

        acc_reward = 0

        # for i in range(repeat):
        #     acc_reward += info['repeats']['prev_steps'][i][1]
        
        # assert acc_reward == reward
        # print(info["repeats"]['num_steps'])
        # print(observation, info["repeats"]['prev_steps'][repeat - 1][0])

        if terminated or truncated:
            observation, info = env.reset()
        

    env.close()
