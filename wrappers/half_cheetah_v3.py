import gym
import numpy as np


class HalfCheetahGoalEnv:

    def __init__(self, seed=0, max_goal_distance=10,
                 goal_distance_threshold=0.1, reward_type='sparse'):
        """Initializes the wrapper.

        Args:
            seed (int, optional): Random seed. Defaults to 0.
            max_goal_distance (int, optional): Range for goal sampling will be
                [-max_goal_distance, max_goal_distance]. Defaults to 10.
            goal_distance_threshold (float, optional): If x position becomes
                less than or equal to threshold, reward is given and episode is
                terminated. Defaults to 0.1.
            reward_type (str, optional): In ['sparse', 'dense']. If 'dense',
                reward is negative L2 distance to goal. Defaults to 'sparse'.
        """
        self._env = gym.make('HalfCheetah-v3')
        self._max_goal_distance = max_goal_distance
        self._goal_distance_threshold = goal_distance_threshold
        self._reward_type = reward_type
        self.sample_goal()

    @property
    def observation_space(self):
        observation_shape = (self._env.observation_space.shape[0] + 1,)
        space = gym.spaces.Box(
            low=-float('inf'),
            high=float('inf'),
            shape=observation_shape)
        return space

    @property
    def action_space(self):
        return self._env.action_space

    def reset(self):
        obs = self._env.reset()
        self.sample_goal()
        return {
            'observation': self.concatenate_goal(obs),
            'achieved_goal': [0],
        }

    def step(self, action):
        observation, original_reward, done, info = self._env.step(action)
        observation = self.concatenate_goal(observation)
        observation = {
            'observation': observation,
            'achieved_goal': [info['x_position']],
        }
        goal_distance = np.linalg.norm(info['x_position'] - self.goal)
        if goal_distance < self._goal_distance_threshold:
            done = True
        reward = 0
        if self._reward_type == 'sparse':
            reward = 1 if goal_distance < self._goal_distance_threshold else 0
        elif self._reward_type == 'dense':
            reward = -goal_distance
        else:
            NotImplementedError(self._reward_type)
        return observation, reward, done, info

    def render(self, *args, **kwargs):
        return self._env.render(*args, **kwargs)

    def sample_goal(self):
        self.goal = np.random.uniform(low=-self._max_goal_distance,
                                      high=self._max_goal_distance,
                                      size=1)

    def concatenate_goal(self, obs):
        return np.concatenate([obs, self.goal], axis=-1)
