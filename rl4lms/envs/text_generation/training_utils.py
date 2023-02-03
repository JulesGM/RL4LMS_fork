from functools import partial
import logging
from typing import Any, Dict, List, Optional

import numpy as np
from rl4lms.data_pools.text_generation_pool import Sample
from rl4lms.envs.text_generation.env import TextGenEnv
from rl4lms.envs.text_generation.evaluation_utils import evaluate_on_samples
from rl4lms.envs.text_generation.utils_supervised import (
    evaluate_on_samples as evaluate_supervised,
)
from rl4lms.envs.text_generation.dataset_utils import create_dataloader
from rl4lms.envs.text_generation.logging_utils import Tracker
from rl4lms.envs.text_generation.registry import (
    DataPoolRegistry,
    MetricRegistry,
    RewardFunctionRegistry,
    PolicyRegistry,
    AlgorithmRegistry,
    WrapperRegistry,
)
from rl4lms.envs.text_generation.reward import RewardFunction
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq,
)
from accelerate import Accelerator

from rl4lms.envs.text_generation.utils_supervised import (
    get_datasets_for_causal,
    get_datasets_for_seq2seq,
    tokenize_causal,
    tokenize_seq2seq,
    EvalCallack,
)
from rl4lms.envs.text_generation.warm_start import TrainerWarmStartMixin

LOGGER = logging.getLogger(__name__)
DEEPSPEED_KEY = "supervised_deepspeed"
OUTPUT_DIR_KEY = "supervised_output_dir"

def build_tokenizer(tokenizer_config: Dict[str, Any]):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_config["model_name"])
    if tokenizer.pad_token is None and tokenizer_config.get(
        "pad_token_as_eos_token", True
    ):
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = tokenizer_config.get("padding_side", "left")
    tokenizer.truncation_side = tokenizer_config.get("truncation_side", "left")
    return tokenizer


def build_reward_fn(reward_config: Dict[str, Any]):
    reward_fn = RewardFunctionRegistry.get(
        reward_config["id"], reward_config.get("args", {},)
    )
    return reward_fn


def build_metrics(metric_configs: List[Dict[str, Any]]):
    metrics = [
        MetricRegistry.get(metric_config["id"], metric_config.get("args", {}))
        for metric_config in metric_configs
    ]
    return metrics


def build_datapool(datapool_config: Dict[str, Any]):
    def _get_datapool_by_split(split: str):
        kwargs = datapool_config.get("args", {})
        kwargs["split"] = split
        dp_split = DataPoolRegistry.get(datapool_config["id"], kwargs)
        return dp_split

    train_datapool = _get_datapool_by_split("train")
    val_datapool = _get_datapool_by_split("val")
    test_datapool = _get_datapool_by_split("test")

    samples_by_split = {
        "train": [(sample, weight) for sample, weight in train_datapool],
        "val": [sample for sample, _ in val_datapool],
        "test": [sample for sample, _ in test_datapool],
    }
    return samples_by_split


def build_env(
    env_config: Dict[str, Any],
    reward_fn: RewardFunction,
    tokenizer: AutoTokenizer,
    train_samples: List[Sample],
):
    # vectoried env
    env_kwargs = {
        "reward_function": reward_fn,
        "tokenizer": tokenizer,
        "samples": train_samples,
    }
    env_kwargs = {**env_kwargs, **env_config.get("args", {})}
    env = make_vec_env(
        TextGenEnv,
        n_envs=env_config.get("n_envs", 1),
        vec_env_cls=DummyVecEnv,
        env_kwargs=env_kwargs,
    )
    return env


def build_alg(
    accelerator: Accelerator,
    alg_config: Dict[str, Any],
    env: TextGenEnv,
    tracker: Tracker,
    policy_state: Dict[str, Any],
    alg_state: Dict[str, Any],
):
    policy_config = alg_config["policy"]
    policy_cls = PolicyRegistry.get(policy_config["id"])
    alg_cls = AlgorithmRegistry.get(alg_config["id"])

    policy_args = policy_config["args"]
    policy_args["state_dict"] = policy_state
    alg_kwargs = {
        "policy": policy_cls,
        "env": env,
        "policy_kwargs": policy_args,
    }
    alg_kwargs = {**alg_kwargs, **alg_config.get("args")}
    wrapper = WrapperRegistry.get(alg_config["id"])
    alg = wrapper(
        accelerator=accelerator,
        alg_class=alg_cls,
        alg_kwargs=alg_kwargs,
        kl_coeff=alg_config["kl_div"]["coeff"],
        tracker=tracker,
        target_kl=alg_config["kl_div"].get("target_kl", None),
        norm_reward=alg_config["kl_div"].get("norm_reward", False),
        dataloader_kwargs=alg_config.get("dataloader_kwargs", {}),
    )
    alg.load_from_dict(alg_state)
    return alg


class OnPolicyTrainer(TrainerWarmStartMixin):
    """
    A generic trainer for training LMs with onpolicy algorithms from SB3
    """

    def __init__(
        self,
        *,
        datapool_config: Dict[str, Any],
        env_config: Dict[str, Any],
        experiment_name: str = "",
        on_policy_alg_config: Dict[str, Any],
        reward_config: Dict[str, Any],
        tokenizer_config: Dict[str, Any],
        tracker: Tracker = None,
        train_eval_config: Dict[str, Any],
    ):
        self._datapool_config = datapool_config
        self._env_config = env_config
        self._experiment_name = experiment_name
        self._on_policy_alg_config = on_policy_alg_config
        self._reward_config = reward_config
        self._tokenizer_config = tokenizer_config
        self._tracker = tracker
        self._train_eval_config = train_eval_config
        self._setup()

    def _setup(self):

        # load trainer state from available previous checkpoint if available
        self.load_trainer_state(self._tracker)

        # build components
        self._tokenizer = build_tokenizer(self._tokenizer_config)
        self._reward_fn = build_reward_fn(self._reward_config)
        self._metrics = build_metrics(self._train_eval_config.get("metrics", []))
        self._samples_by_split = build_datapool(self._datapool_config)
        self._env = build_env(
            self._env_config,
            self._reward_fn,
            self._tokenizer,
            self._samples_by_split["train"],
        )

        # extract train params
        self._max_episode_length = self._env_config["args"]["max_episode_length"]
        self._max_prompt_length = self._env_config["args"]["max_prompt_length"]
        self._eval_batch_size = self._train_eval_config["eval_batch_size"]
        self._n_iters = int(self._train_eval_config["n_iters"])

        # gen kwargs for evaluation (if it is different from rollout gen kwargs)
        self._eval_gen_kwargs = self._train_eval_config.get("generation_kwargs", None)


        self._accelerator = Accelerator()

        # prepare for accelerate
        self._alg = build_alg(
            accelerator=self._accelerator,
            alg_config=self._on_policy_alg_config,
            env=self._env,
            tracker=self._tracker,
            policy_state=self._policy_state_dict,
            alg_state=self._alg_state_dict,
        )
        self._n_steps_per_iter = self._env.num_envs * self._alg.n_steps

        assert isinstance(self._alg.policy, BasePolicy), type(self._alg.policy).mro()
        self._prepare_accelerate()
        assert isinstance(self._alg.policy, BasePolicy), type(self._alg.policy).mro()

        

    def _prepare_accelerate(self):

        assert isinstance(self._alg.policy, BasePolicy), type(self._alg.policy).mro()
        # create optimizer first
        optimizer = self._alg.policy.setup_optimizer()

        # prepare dataloaders
        self._dataloaders = {
            "val": create_dataloader(self._samples_by_split["val"], self._eval_batch_size),
            "test": create_dataloader(self._samples_by_split["test"], self._eval_batch_size),
        }
        assert isinstance(self._accelerator.unwrap_model(self._alg.policy), BasePolicy), (
            type(self._alg.policy).mro()
        )

        # prepare policy, optimizer and dataloader
        (
            self._alg.policy,
            self._alg.optimizer,
            self._dataloaders["val"],
            self._dataloaders["test"],
            *accelerated_models,
        ) = self._accelerator.prepare(self._alg.policy,
                                      optimizer, 
                                      self._dataloaders["val"], 
                                      self._dataloaders["test"],
                                      *self._reward_fn._models.values())

        assert isinstance(self._accelerator.unwrap_model(self._alg.policy), BasePolicy), (
            type(self._alg.policy).mro()
        )

        # Dicts are orderd, this is fine
        for key, accelerated in zip(self._reward_fn._models, accelerated_models):
            self._reward_fn._models[key] = accelerated



    def _evaluate_on_datapools(self, epoch: int, splits: List[str] = ["val", "test"]):
        
        LOGGER.debug(f"[blue bold]OnPolicyTrainer._evaluate_on_datapools: [white]Entry - {splits}")

        assert isinstance(self._alg.policy, BasePolicy), type(self._alg.policy).mro()

        for split in splits:
            evaluate_on_samples(
                policy=self._alg.policy,
                tokenizer=self._tokenizer,
                dataloader=self._dataloaders[split],
                max_prompt_length=self._max_prompt_length,
                metrics=self._metrics,
                epoch=epoch,
                split_name=split,
                accelerator=self._accelerator,
                tracker=self._tracker,
                gen_kwargs=self._eval_gen_kwargs,
            )

        LOGGER.debug(f"[blue bold]OnPolicyTrainer._evaluate_on_datapools: [white]Exit - {splits}")
    

    def train_and_eval(self):
        # evaluate on val and test set before fine-tuning once
        iter_start = self._trainer_state["current_iter"]
        self._evaluate_on_datapools(epoch=iter_start)

        assert False

        # train for given number of iters
        for epoch in range(iter_start, self._n_iters):
            # current state
            self._trainer_state["current_iter"] = epoch

            # inner rollout and learn loop for on-policy algorithm
            self._alg.learn(self._n_steps_per_iter)

            # save the policy checkpoint
            if (epoch + 1) % self._train_eval_config.get("save_every", 20) == 0:
                self.save_trainer_state(
                    self._tracker, self._alg.policy, self._trainer_state)

            # evaluate on val set in the given intervals
            if (epoch + 1) % self._train_eval_config["eval_every"] == 0:
                self._evaluate_on_datapools(epoch=epoch, splits=["val"])

        # finally evaluate on val and test samples
        self._evaluate_on_datapools(epoch=epoch)

        # save model here - we save only the language model
        if self._tracker is not None:
            self._tracker.save_auto_model(
                self._alg.policy.get_language_model())


class SupervisedTrainer:
    """
    A supervised trainer to train LMs (causal and seq2seq) on text generation tasks (wrapper on HF trainer)
    """

    def __init__(
        self,
        *,
        tokenizer_config: Dict[str, Any],
        datapool_config: Dict[str, Any],
        train_eval_config: Dict[str, Any],
        alg_config: Dict[str, Any],
        tracker: Tracker = None,
    ):
        self._tokenizer_config = tokenizer_config
        self._datapool_config = datapool_config
        self._train_eval_config = train_eval_config
        self._alg_config = alg_config
        self._tracker = tracker
        self._setup()

    def _evaluate_on_datapools(
        self, 
        epoch: int, 
        splits: List[str] = ["val", "test"]
    ):
        for split in splits:
            evaluate_supervised(
                model=self._model,
                tokenizer=self._tokenizer,
                samples=self._samples_by_split[split],
                batch_size=self._eval_batch_size,
                max_prompt_length=self._max_prompt_length,
                metrics_config_dict=self._metrics_config_dict,
                epoch=epoch,
                split_name=split,
                tracker=self._tracker,
                generation_kwargs=self._gen_kwargs,
            )

    def _setup(self):
        self._tokenizer = build_tokenizer(self._tokenizer_config)
        self._metrics_config_dict = self._train_eval_config.get("metrics")
        self._samples_by_split = build_datapool(self._datapool_config)
        self._train_dataset = (
            get_datasets_for_causal(self._samples_by_split["train"])
            if self._alg_config["model_type"] == "causal"
            else get_datasets_for_seq2seq(self._samples_by_split["train"])
        )
        preprocess_fn = (
            tokenize_causal
            if self._alg_config["model_type"] == "causal"
            else tokenize_seq2seq
        )
        preprocess_fn = partial(preprocess_fn, tokenizer=self._tokenizer)
        self._tokenized_dataset = self._train_dataset.map(
            preprocess_fn, 
            batched=True, 
            remove_columns=self._train_dataset.column_names
        )
        model_cls = (
            AutoModelForCausalLM
            if self._alg_config["model_type"] == "causal"
            else AutoModelForSeq2SeqLM
        )
        self._gen_kwargs = self._alg_config["generation_kwargs"]
        self._model = model_cls.from_pretrained(self._alg_config["model_name"])
        self._eval_batch_size = self._train_eval_config["eval_batch_size"]

        # setting max prompt length
        self._max_prompt_length = self._tokenizer_config.get(
            "max_length", self._tokenizer.model_max_length
        )

        if (self._alg_config["model_type"] == "causal") and (
            (self._max_prompt_length + self._gen_kwargs["max_new_tokens"])
            > self._tokenizer.model_max_length
        ):
            self._max_prompt_length = (
                self._max_prompt_length - self._gen_kwargs["max_new_tokens"]
            )

        self._eval_callback = EvalCallack(
            self._samples_by_split["val"],
            self._gen_kwargs,
            self._eval_batch_size,
            self._tokenizer,
            self._metrics_config_dict,
            self._max_prompt_length,
            self._tracker,
        )
        train_args = self._alg_config["training_args"]
        train_args["output_dir"] = self._tracker.checkpoint_base_path
        train_args["seed"] = np.random.randint(1e2)  # random seed
        data_collator = (
            DataCollatorForLanguageModeling(self._tokenizer, mlm=False)
            if self._alg_config["model_type"] == "causal"
            else DataCollatorForSeq2Seq(self._tokenizer, self._model)
        )
        if (
            DEEPSPEED_KEY in self._train_eval_config and 
            self._train_eval_config[DEEPSPEED_KEY]
        ):
            LOGGER.debug("[bold red]SupervisedTrainer: [bold white]Using deepspeed")
            train_args["output_dir"] = self._train_eval_config[OUTPUT_DIR_KEY]
            train_args["deepspeed"] = self._train_eval_config[DEEPSPEED_KEY]
        else:
            self._model.parallelize()

        assert train_args["deepspeed"], train_args["deepspeed"]
        self._train_args = TrainingArguments(**train_args)
        self._trainer = Trainer(
            model=self._model,
            tokenizer=self._tokenizer,
            args=self._train_args,
            data_collator=data_collator,
            train_dataset=self._tokenized_dataset,
            callbacks=[self._eval_callback],
        )

    def train_and_eval(self):
        # evaluate on val and test set before fine-tuning once
        self._evaluate_on_datapools(epoch=0)

        assert False

        # train using HF trainer
        self._trainer.train()

        # finally evaluate on val and test samples
        self._evaluate_on_datapools(epoch=self._train_args.num_train_epochs)

        # save model here - we save only the language model
        if self._tracker is not None:
            self._tracker.save_auto_model(self._model)
