"""Definitions of models used for the project.
We follow the specs of Diversity is All You Need
Implementation based on ACME samples.
https://github.com/deepmind/acme/blob/master/examples/control_suite/run_dmpo.py

Returns:
    [type]: [description]
"""
from typing import Dict, Sequence, Tuple
import numpy as np
import sonnet as snt
from acme.tf import utils as tf2_utils
from acme import specs
from acme import types
from acme.tf import networks

def make_feed_forward_networks(
    action_spec: specs.BoundedArray,
    z_spec: specs.BoundedArray,
    policy_layer_sizes: Tuple[int, ...] = (256, 256),
    critic_layer_sizes: Tuple[int, ...] = (256, 256),
    discriminator_layer_sizes: Tuple[int, ...] = (256, 256),
    hierarchical_controller_layer_sizes: Tuple[int, ...] = (256, 256),
    vmin: float = -150.,  # Minimum value for the Critic distribution.
    vmax: float = 150.,  # Maximum value for the Critic distribution.
    num_atoms: int = 51,  # Number of atoms for the discrete value distribution.
) -> Dict[str, types.TensorTransformation]:
    num_dimensions = np.prod(action_spec.shape, dtype=int)
    z_dim = np.prod(z_spec.shape, dtype=int)

    observation_network = tf2_utils.batch_concat

    policy_network = snt.Sequential([
        networks.LayerNormMLP(policy_layer_sizes),
        networks.MultivariateNormalDiagHead(num_dimensions)
    ])

    critic_multiplexer = networks.CriticMultiplexer(
        critic_network=networks.LayerNormMLP(critic_layer_sizes),
        action_network=networks.ClipToSpec(action_spec)
    )

    critic_network = snt.Sequential([
        critic_multiplexer,
        networks.DiscreteValuedHead(vmin, vmax, num_atoms),
    ])

    # The discriminator in DIAYN uses the same architecture as the critic.
    discriminator_network = networks.LayerNormMLP(
        discriminator_layer_sizes + (z_dim,))

    hierarchical_controller_network = networks.LayerNormMLP(
        hierarchical_controller_layer_sizes + (z_dim,))

    return {
        'policy': policy_network,
        'critic': critic_network,
        'observation': observation_network,
        'discriminator': discriminator_network,
        'hierarchical_controller': hierarchical_controller_network,
    }