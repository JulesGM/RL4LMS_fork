"""
Code adapted from https://github.com/DLR-RM/stable-baselines3
"""



from typing import Generator, NamedTuple, Optional, Union


import accelerate
import numpy as np
import torch
import torch as th
from gym import spaces
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer
from stable_baselines3.common.type_aliases import TensorDict
from stable_baselines3.common.vec_env import VecNormalize



class MaskableRolloutBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    action_masks: th.Tensor


class MaskableDictRolloutBufferSamples(MaskableRolloutBufferSamples):
    observations: TensorDict
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    action_masks: th.Tensor



class MaskableRolloutBuffer(RolloutBuffer):
    """
    Rollout buffer that also stores the invalid action masks associated with each observation.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device:
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param gamma: Discount factor
    :param n_envs: Number of parallel environments
    """

    def __init__(self, *args, **kwargs):
        self.action_masks = None
        super().__init__(*args, **kwargs)

    def reset(self) -> None:
        if isinstance(self.action_space, spaces.Discrete):
            mask_dims = self.action_space.n
        elif isinstance(self.action_space, spaces.MultiDiscrete):
            mask_dims = sum(self.action_space.nvec)
        elif isinstance(self.action_space, spaces.MultiBinary):
            mask_dims = 2 * self.action_space.n  # One mask per binary outcome
        else:
            raise ValueError(
                f"Unsupported action space {type(self.action_space)}")

        self.mask_dims = mask_dims
        self.action_masks = np.ones(
            (self.buffer_size, self.n_envs, self.mask_dims), dtype=np.float32)

        super().reset()

    def add(self, *args, action_masks: Optional[np.ndarray] = None, **kwargs) -> None:
        """
        :param action_masks: Masks applied to constrain the choice of possible actions.
        """
        if action_masks is not None:
            self.action_masks[self.pos] = action_masks.reshape(
                (self.n_envs, self.mask_dims))

        super().add(*args, **kwargs)

    def get(self, batch_size: Optional[int] = None) -> Generator[MaskableRolloutBufferSamples, None, None]:
        """
        
        What this does: 
        # 1 shuffle the indices of the buffer
        # 2 swap and flatten the data if it hasn't been done yet
        # 3 yield the data in minibatches if batch_size is not None

        So it's basically a dataloader. 

        """
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:
            for tensor in [
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
                "action_masks",
            ]:
                self.__dict__[tensor] = self.swap_and_flatten(
                    self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx: start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> MaskableRolloutBufferSamples:
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
            self.action_masks[batch_inds].reshape(-1, self.mask_dims),
        )
        return MaskableRolloutBufferSamples(*map(self.to_torch, data))


class MaskableDictRolloutBuffer(DictRolloutBuffer):
    """
    Dict Rollout buffer used in on-policy algorithms like A2C/PPO.
    Extends the RolloutBuffer to use dictionary observations

    It corresponds to ``buffer_size`` transitions collected
    using the current policy.
    This experience will be discarded after the policy update.
    In order to use PPO objective, we also store the current value of each state
    and the log probability of each taken action.

    The term rollout here refers to the model-free notion and should not
    be used with the concept of rollout used in model-based RL or planning.
    Hence, it is only involved in policy and value function training but not action selection.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device:
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param gamma: Discount factor
    :param n_envs: Number of parallel environments
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "cpu",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):
        self.action_masks = None
        super().__init__(buffer_size, observation_space,
                         action_space, device, gae_lambda, gamma, n_envs=n_envs)

    def reset(self) -> None:
        if isinstance(self.action_space, spaces.Discrete):
            mask_dims = self.action_space.n
        elif isinstance(self.action_space, spaces.MultiDiscrete):
            mask_dims = sum(self.action_space.nvec)
        elif isinstance(self.action_space, spaces.MultiBinary):
            mask_dims = 2 * self.action_space.n  # One mask per binary outcome
        else:
            raise ValueError(
                f"Unsupported action space {type(self.action_space)}")

        self.mask_dims = mask_dims
        self.action_masks = np.ones(
            (self.buffer_size, self.n_envs, self.mask_dims))  # .to(self.device)

        super().reset()

    def add(self, *args, action_masks: Optional[torch.Tensor] = None, **kwargs) -> None:
        """
        :param action_masks: Masks applied to constrain the choice of possible actions.
        """
        if action_masks is not None:
            self.action_masks[self.pos] = action_masks.reshape(
                (self.n_envs, self.mask_dims))

        super().add(*args, **kwargs)

    def get(self, batch_size: Optional[int] = None) -> Generator[MaskableDictRolloutBufferSamples, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:

            for key, obs in self.observations.items():
                self.observations[key] = self.swap_and_flatten(obs)

            _tensor_names = ["actions", "values", "log_probs",
                             "advantages", "returns", "action_masks"]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(
                    self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx: start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> MaskableDictRolloutBufferSamples:

        return MaskableDictRolloutBufferSamples(
            observations={key: self.to_torch(obs[batch_inds]) for (key, obs) in self.observations.items()},
            actions=self.to_torch(self.actions[batch_inds]),
            old_values=self.to_torch(self.values[batch_inds].flatten()),
            old_log_prob=self.to_torch(self.log_probs[batch_inds].flatten()),
            advantages=self.to_torch(self.advantages[batch_inds].flatten()),
            returns=self.to_torch(self.returns[batch_inds].flatten()),
            action_masks=self.to_torch(self.action_masks[batch_inds].reshape(-1, self.mask_dims)),
        )


class MaskableDictRolloutDataset(MaskableDictRolloutBuffer, torch.utls.data.Dataset):
    """Contains the data iterated on by the Dataloader.

    Accelerate:
    This class does not handle any parallelism. 
    That is handled by the Dataloader & its Accelerate wrapper. 
    It doesn't know if we're using accelerate.

    """

    def get(self, *args, **kwargs):
        raise RuntimeError(
            "Should not be called. "
            "Use __getitem__ or the Dataloader version."
        )

    def _prep_data(self):
        assert self.full, ""
        # Prepare the data
        if not self.generator_ready:

            for key, obs in self.observations.items():
                self.observations[key] = self.swap_and_flatten(obs)

            _tensor_names = [
                "actions", 
                "values", 
                "log_probs",
                "advantages", 
                "returns", 
                "action_masks"
            ]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(
                    self.__dict__[tensor])

            self.generator_ready = True

    def __len__(self):
        target_len = self.buffer_size * self.n_envs
        _tensor_names = [
                "actions", 
                "values", 
                "log_probs",
                "advantages", 
                "returns", 
                "action_masks"
            ]

        for name in _tensor_names:
            assert len(self.__dict__(name)) == target_len, (
                f"Tensor size is unexpected: {name = }, "
                f"{len(self.__dict__(name)) = }, {target_len = }"
            )

        return target_len

    def __getitem__(self, idx):
        assert self.generator_ready
        return self._get_samples(idx)


    class MaskableDictRolloutDataloader(torch.utils.data.DataLoader):
        """The class to replace MaskableDictRolloutBuffer for Accelerate.

        The class is also functional without an Accelerator instance.

        Accelerate:
        This class knows about the accelerator.
        Its job is to keep the buffers in sync and to be wrapped by `Accelerator.prepare`
        or `Accelerator.prepare_data_loader`.

        """
        
        def __init__(self, *args, accelerator: Optional[accelerate.Accelerator], **kwargs):
            super().__init__(*args, **kwargs)
            self.accelerator = accelerator

        def reset(self):
            """Empties the buffers.

            Accelerate: 
            Each process can release its own buffers. 
            No change needed.

            """
            self.dataset.reset()

        def add(self, *args, **kwargs):
            """Adds new data to the buffer.

            Accelerate: 
            We send the new data to each process.            

            """
            if self.accelerator is not None:
                for i, v in enumerate(args):
                    if isinstance(v, torch.Tensor):
                        args[i] = self.accelerator.gather(v)
                for k, v in kwargs.items():
                    if isinstance(v, torch.Tensor):
                        kwargs[k] = self.accelerator.gather(v)

            self.dataset.add(*args, **kwargs)

        def get(self, batch_size):
            """Returns a generator that yields batches.

            Accelerate:
            We return outself, which is a dataloader that is wrapped
            by Accelerator.
            The only issue is if batch_size is `None`.

            # TODO: fix this.

            """
            assert batch_size is not None, "A batch size of None is not currently supported."

            return self

        def __iter__(self):
            """Returns a generator that yields batches, prepares the data if needed.
            
            Accelerate: 
            We prepare the data then let the wrapped 
            dataloader handle the rest.

            """
            self.dataset._prep_data()
            return super().__iter__()

        @property
        def rewards(self):
            """Return the rollout buffer rewards.
            
            Accelerate:
            They're all the same on the different processes, so we can just
            return the process-local one.
            
            """
            return self.dataset.rewards

        @rewards.setter
        def rewards(self, new_val):
            """Set the rollout buffer rewards.
            
            Accelerate:
            We sync them across the processes.
            
            """
            if self.accelerator is not None:
                self.dataset.rewards = self.accelerator.gather(new_val)
            else:
                self.dataset.rewards = new_val

        @property
        def actions(self):
            return self.dataset.actions
        
        @property
        def values(self):
            return self.dataset.values
        
        @property
        def log_probs(self):
            return self.dataset.log_probs

        @property
        def advantages(self):
            return self.dataset.advantages

        @property
        def returns(self):
            return self.dataset.returns
    
        @property
        def action_masks(self):
            return self.dataset.action_masks
        
        @property
        def full(self):
            return self.dataset.full