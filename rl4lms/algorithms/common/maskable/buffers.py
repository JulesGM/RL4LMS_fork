"""
Code adapted from https://github.com/DLR-RM/stable-baselines3
"""



from typing import Generator, NamedTuple, Optional, Union

import more_itertools

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
        ensure_in_dataset=False,
    ):
        assert ensure_in_dataset, "MaskableDictRolloutBuffer is being used without ensure_in_dataset."

        self.action_masks = None
        super().__init__(
            buffer_size, 
            observation_space,
            action_space, 
            device, 
            gae_lambda, 
            gamma, 
            n_envs=n_envs,
        )

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
        assert not self.generator_ready

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

            _tensor_names = [
                "actions", 
                "values", 
                "log_probs",
                "advantages", 
                "returns", 
                "action_masks",
            ]

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

    def _get_samples(
        self, 
        batch_inds: np.ndarray, 
        env: Optional[VecNormalize] = None
    ) -> MaskableDictRolloutBufferSamples:

        assert self.generator_ready

        return MaskableDictRolloutBufferSamples(
            observations ={key: self.to_torch(obs[batch_inds]) for (key, obs) in self.observations.items()},
            actions      =self.to_torch(self.actions     [batch_inds]),
            old_values   =self.to_torch(self.values      [batch_inds].flatten()),
            old_log_prob =self.to_torch(self.log_probs   [batch_inds].flatten()),
            advantages   =self.to_torch(self.advantages  [batch_inds].flatten()),
            returns      =self.to_torch(self.returns     [batch_inds].flatten()),
            action_masks =self.to_torch(self.action_masks[batch_inds].reshape(-1, self.mask_dims)),
        )


class MaskableDictRolloutDataset(torch.utils.data.Dataset):
    """Contains the data iterated on by the Dataloader.

    Accelerate:
        - This class does not handle any parallelism. 
        - That is handled by the Dataloader & its Accelerate wrapper. 
        - It doesn't know if we're using accelerate.

    self._rollout_buffer.generator_ready:
        - indicates if the data is ready to be iterated on.



    """

    def __init__(self, **rollout_buffer_kwargs):
        self._rollout_buffer = MaskableDictRolloutBuffer(
            **rollout_buffer_kwargs, ensure_in_dataset=True
        )

    def get(self, *args, **kwargs):
        raise RuntimeError(
            "Should not be called. "
            "Use __getitem__ or the Dataloader version."
        )

    def _prep_data(self):
        assert self._rollout_buffer.full, "The data is only preped if the buffer is full"
        if not self._rollout_buffer.generator_ready:

            for key, obs in self._rollout_buffer.observations.items():
                self._rollout_buffer.observations[key] = self.swap_and_flatten(obs)

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

            self._rollout_buffer.generator_ready = True

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
            assert len(self._rollout_buffer.__dict__(name)) == target_len, (
                f"Tensor size is unexpected: {name = }, "
                f"{len(self._rollout_buffer.__dict__(name)) = }, {target_len = }"
            )

        return target_len

    def __getitem__(self, idx):
        self._prep_data()
        return self._get_samples(idx)

    def __iter__(self):
        """Returns a generator that yields batches, prepares the data if needed.
        
        Accelerate: 
        We prepare the data then let the wrapped 
        dataloader handle the rest.

        """
        return self

    @property
    def full(self):
        return self._rollout_buffer.full

    def reset(self):
        self._rollout_buffer.reset()


class MaskableDictRolloutDataloaderBuilder:
    """The class to replace MaskableDictRolloutBuffer for Accelerate.

    Creates a dataloader that is wrapped by the Accelerator when .get is called.
    The class is also functional without an Accelerator instance, in which case
    it just returns a dataloader.

    Spec:
        - Needs to be compatible with accelerator.prepare_data_loader being called on it, or to call it itself with .get
        - `.get` should return an iterable that yields batches of data, of size `batch_size`.
        - Needs to return `rollout_buffer.full`
        - Needs to have `rollout_buffer.reset`
        
    Accelerate:
        - This class knows about the accelerator.
        - Its job is to keep the buffers in sync and to be wrapped by `Accelerator.prepare`
            or `Accelerator.prepare_data_loader`.


    """
    
    def __init__(
        self,
        dataset: MaskableDictRolloutDataset, 
        accelerator: Optional[accelerate.Accelerator],
        # batch_size is not optional for the dataloader, so we make it required here.
        **dataloader_kwargs,
    ):
        # Validate args
        fixed_args = {
            "shuffle": False,
            "num_workers": 0,
        }

        for k, v in fixed_args.items():
            if k in dataloader_kwargs:
                assert dataloader_kwargs[k] == v, (
                    f"{k} is fixed to {v} for the dataloader. "
                    f"Got {dataloader_kwargs[k]}, which is not supported."
                )

        # Store args
        self._dataset = dataset
        self._accelerator = accelerator
        self._dataloader_kwargs = dataloader_kwargs


    def reset(self):
        """Empties the buffers.

        Accelerate: 
        Each process can release its own buffers. 
        No change needed.

        """
        self._dataset.reset()

    def add(self, *args, **kwargs):
        """Adds new data to the buffer.

        Accelerate: 
        We send the new data to each process.            

        """
        if self._accelerator:
            for i, v in enumerate(args):
                if isinstance(v, torch.Tensor):
                    args[i] = self._accelerator.gather(v)
            for k, v in kwargs.items():
                if isinstance(v, torch.Tensor):
                    kwargs[k] = self._accelerator.gather(v)

        self._dataset.add(*args, **kwargs)

    def get(self, batch_size):
        """Returns a generator that yields batches.

        Accelerate:
        We return outself, which is a dataloader that is wrapped
        by Accelerator.
        The only issue is if batch_size is `None`.

        """
        assert batch_size is not None, "A batch size of None is not currently supported."
        assert "batch_size" not in self._dataloader_kwargs, "batch_size is already set."

        dataloader = torch.utils.data.DataLoader(
            self._dataset,
            batch_size=batch_size,
            **self._dataloader_kwargs
        )

        if self._accelerator:
            assert False
            yield from self._accelerator.prepare_data_loader(dataloader)
        else:
            yield from dataloader

    @property
    def full(self):
        return self._dataset.full

    def reset(self):
        self._dataset.reset()