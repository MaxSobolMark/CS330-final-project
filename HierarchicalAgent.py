"""ACME agent that wraps around DMPO and adds a discriminator.
"""

from typing import Optional
from DIAYNAgent import DIAYNAgent
import numpy as np
from models import make_feed_forward_networks
from acme import specs
from acme import types
from acme.agents.tf import dmpo
from acme.tf import utils as tf2_utils
import dm_env
import DIAYNAgent
from acme.adders import reverb as adders
import reverb
from acme import datasets


class HierarchicalAgent:

    def __init__(self, DIAYN_agent: DIAYNAgent.DIAYNAgent,
                 environment_spec: specs.EnvironmentSpec,
                 action_spec: specs.BoundedArray, z_dim: int,
                 replay_table_name: str = adders.DEFAULT_PRIORITY_TABLE,
                 replay_server_port: Optional[int] = None,
                 ) -> None:
        self._z_dim = z_dim
        z_spec = specs.BoundedArray((z_dim,), np.float64, minimum=0, maximum=1)
        self._environment_spec = environment_spec
        # Modify the environment_spec to also include the latent variable
        # observation  (z)
        self._obs_space = environment_spec.observations
        assert (len(self._obs_space.shape) == 1), f"Only vector observations are supported for now. Observations shape passed: {obs_shape}"
        self._agent_networks = make_feed_forward_networks(action_spec, z_spec)
        self._agent = dmpo.DistributionalMPO(
            environment_spec=environment_spec,
            policy_network=self._agent_networks['policy'],
            critic_network=self._agent_networks['critic'],
            observation_network=self._agent_networks['observation'],  # pytype: disable=wrong-arg-types
            extra_modules_to_save={
                'hierarchical_controller': self._agent_networks['hierarchical_controller'],
            },
            checkpoint_name='hierarchical_dmpo',
            replay_table_name=replay_table_name,
            replay_server_port=replay_server_port,
            return_action_entropy=True,
        )

        self._DIAYN_agent = DIAYN_agent

        # Create variables for the discriminator.
        tf2_utils.create_variables(
            self._agent_networks['hierarchical_controller'],
            [self._obs_space])

    def select_action(self, observation):
        return self._agent._actor.select_action(observation)

    def observe(self, action: types.NestedArray,
                next_timestep: dm_env.TimeStep):
        return self._agent._actor.observe(action, next_timestep)

    def observe_first(self, timestep: dm_env.TimeStep):
        return self._agent._actor.observe_first(timestep)

    def learner_step(self):
        return self._agent._learner.step()

    def reset_replay_table(self, name='new_replay_table'):
        replay_table = reverb.Table(
            name=name,
            sampler=reverb.selectors.Uniform(),
            remover=reverb.selectors.Fifo(),
            max_size=1000000,
            rate_limiter=reverb.rate_limiters.MinSize(min_size_to_sample=1),
            signature=adders.NStepTransitionAdder.signature(self._environment_spec))
        port = self._agent._server.port
        del self._agent._server
        self._agent._server = reverb.Server([replay_table], port=port)
        dataset = datasets.make_reverb_dataset(
            table=name,
            server_address=f'localhost:{port}',
            batch_size=256,
            prefetch_size=4,
        )
        self._agent._learner._iterator = iter(dataset)
