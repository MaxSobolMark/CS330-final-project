import gym
import numpy as np


class HierarchicalEnv:

    def __init__(self, env, diayn_agent, z_dim):
        """initializes the wrapper.

        Args:
            env (gym environment): The environment to wrap around.
            diayn_agent (DIAYNAgent): the agent that executes the actual
                behavior.
            z_dim (int): The dimensionality of the latent variable.
        """
        self._env = env
        self._diayn_agent = diayn_agent
        self._z_dim = z_dim
        self._last_observation = self._env.reset()

    @property
    def observation_space(self):
        return self._env.observation_space

    @property
    def action_space(self):
        space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(self._z_dim,))
        return space

    def reset(self):
        self._last_observation = self._env.reset()['observation']
        return self._last_observation

    def step(self, z):
        # self._diayn_agent.set_z(np.argmax(z))
        action, entropy = self._diayn_agent.select_action(self._last_observation, z=z)
        observation, reward, done, info = self._env.step(action)
        self._last_observation = observation['observation']
        return observation['observation'], np.float32(reward), done, info

    def render(self, *args, **kwargs):
        return self._env.render(*args, **kwargs)
