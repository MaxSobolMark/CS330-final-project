"""ACME agent that wraps around DMPO and adds a discriminator.
"""

import numpy as np
from models import make_feed_forward_networks
from acme import specs
from acme import types
from acme.agents.tf import dmpo
from acme.tf import utils as tf2_utils
from acme.utils import loggers
from tensorflow_probability import distributions as tfd
import tensorflow as tf
import sonnet as snt
import dm_env


class DIAYNAgent:

    def __init__(self, environment_spec: specs.EnvironmentSpec,
                 action_spec: specs.BoundedArray, z_dim: int) -> None:
        self._z_dim = z_dim
        z_spec = specs.BoundedArray((z_dim,), np.float64, minimum=0, maximum=1)
        # Modify the environment_spec to also include the latent variable
        # observation  (z)
        self._obs_space = environment_spec.observations
        assert (len(self._obs_space.shape) == 1), f"Only vector observations are supported for now. Observations shape passed: {obs_shape}"
        updated_observations = specs.BoundedArray(
            (self._obs_space.shape[0] + z_dim,),
            dtype=environment_spec.observations.dtype,
            name=environment_spec.observations.name,
            minimum=np.append(
                environment_spec.observations.minimum, [0]*z_dim),
            maximum=np.append(
                environment_spec.observations.maximum, [0]*z_dim),
            )
        environment_spec = specs.EnvironmentSpec(
          observations=updated_observations,
          actions=environment_spec.actions,
          rewards=environment_spec.rewards,
          discounts=environment_spec.discounts,
        )
        self._agent_networks = make_feed_forward_networks(action_spec, z_spec)
        self._agent = dmpo.DistributionalMPO(
            environment_spec=environment_spec,
            policy_network=self._agent_networks['policy'],
            critic_network=self._agent_networks['critic'],
            observation_network=self._agent_networks['observation'],  # pytype: disable=wrong-arg-types
            extra_modules_to_save={
                'discriminator': self._agent_networks['discriminator'],
            },
            return_action_entropy=True,
        )

        self._z_distribution = tfd.Categorical([1] * z_dim)
        self._current_z = self._z_distribution.sample()

        # Create discriminator optimizer.
        self._discriminator_optimizer = snt.optimizers.Adam(1e-4)
        self._discriminator_logger = loggers.make_default_logger(
            'discriminator')

        # Create variables for the discriminator.
        tf2_utils.create_variables(self._agent_networks['discriminator'],
                                   [self._obs_space])

    def sample_z(self):
        self._current_z = self._z_distribution.sample()
        return self._current_z

    def set_z(self, z):
        self._current_z = z

    def _concat_latent_variable(self, obs):
        return tf.concat(
            [obs, tf.one_hot(self._current_z, self._z_dim)],
            axis=-1)

    def select_action(self, observation, z=None):
        if z is not None:
            inputs = tf.concat([
                observation.astype(np.float32), z.astype(np.float32)], axis=-1)
        else:
            inputs = self._concat_latent_variable(observation)
        return self._agent._actor.select_action(inputs)

    def observe(self, action: types.NestedArray,
                next_timestep: dm_env.TimeStep):
        # Create a new timestep with the latent variable in the observation
        new_timestep = dm_env.TimeStep(
            step_type=next_timestep.step_type,
            reward=next_timestep.reward,
            discount=next_timestep.discount,
            observation=self._concat_latent_variable(next_timestep.observation),
        )
        return self._agent._actor.observe(action, new_timestep)

    def observe_first(self, timestep: dm_env.TimeStep):
        # Create a new timestep with the latent variable in the observation
        new_timestep = dm_env.TimeStep(
            step_type=timestep.step_type,
            reward=timestep.reward,
            discount=timestep.discount,
            observation=self._concat_latent_variable(timestep.observation),
        )
        return self._agent._actor.observe_first(new_timestep)

    def learner_step(self):
        return self._agent._learner.step()

    def discriminator_step(self):
        fetches = self._discriminator_step()
        # Write to logger
        self._discriminator_logger.write(fetches)

    @tf.function
    def _discriminator_step(self):
        it = self._agent._learner._iterator
        inputs = next(it)
        o_tm1, a_tm1, r_t, d_t, o_t = inputs.data  # tm1 means t - 1.
        # Get latent variable from observation
        z = o_t[:, -self._z_dim:]
        # Remove the latent variable from the observation so that the
        # discriminator doesn't have access to it.
        s_t = o_t[:, :-self._z_dim]

        with tf.GradientTape() as tape:
            s_t = self._agent_networks['observation'](s_t)
            predicted_z = self._agent_networks['discriminator'](s_t)
            loss = tf.nn.softmax_cross_entropy_with_logits(
                logits=predicted_z, labels=z)
            loss = tf.reduce_mean(loss)

        trainable_variables = (
            self._agent_networks['discriminator'].trainable_variables)
        gradients = tape.gradient(loss, trainable_variables)
        # Apply gradients.
        self._discriminator_optimizer.apply(gradients, trainable_variables)

        return {
            'discriminator_loss': loss,
        }

    @tf.function
    def predict_z(self, s_t):
        # Assume s_t doesn't contain latent variable.
        x = self._agent_networks['observation'](s_t)
        x = self._agent_networks['discriminator'](x)
        return tf.nn.softmax(x, axis=-1)

    @tf.function
    def get_log_prior_of_z(self):
        return self._z_distribution.log_prob(self._current_z)

    @tf.function
    def get_diayn_reward(self, next_state):
        next_state = tf.reshape(next_state, [1, self._obs_space.shape[0]])
        return (tf.math.log(self.predict_z(next_state)[0, self._current_z]) -
                self.get_log_prior_of_z())
