import hydra
import ray

from .dataset import AgentDataset
from .trainer import AgentFlowTrainer
from verl.trainer.ppo.reward import load_reward_manager
from verl.trainer.main_ppo import create_rl_sampler


@hydra.main(config_path="pkg://agentflow/verl", config_name="config", version_base=None)
def main(config):
    run_ppo(config)


def run_ppo(config) -> None:
    import sys
    print("[entrypoint] Starting run_ppo...", flush=True)

    if not ray.is_initialized():
        # this is for local ray cluster
        print("[entrypoint] Initializing Ray...", flush=True)
        ray.init(
            runtime_env={
                "env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN", "VLLM_LOGGING_LEVEL": "WARN"}
            },
        )

    # Run training directly in main process (not as Ray remote task) to see output
    print("[entrypoint] Running training directly in main process...", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    run_training(config)
    print("[entrypoint] Training completed.", flush=True)


def run_training(config):
    import sys
    import traceback

    try:
        print("[Training] Starting...", flush=True)

        # print initial config
        from pprint import pprint
        from omegaconf import OmegaConf
        from verl.utils.fs import copy_to_local

        print("[Training] Config:", flush=True)
        pprint(OmegaConf.to_container(config, resolve=True))
        sys.stdout.flush()
        OmegaConf.resolve(config)

        # download the checkpoint from hdfs
        print(f"[Training] Downloading model: {config.actor_rollout_ref.model.path}", flush=True)
        local_path = copy_to_local(config.actor_rollout_ref.model.path)
        print(f"[Training] Model downloaded to: {local_path}", flush=True)

        # instantiate tokenizer
        from verl.utils import hf_processor, hf_tokenizer

        print("[Training] Loading tokenizer...", flush=True)
        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, use_fast=True)
        print("[Training] Tokenizer loaded.", flush=True)

        # define worker classes
        print("[Training] Setting up worker classes...", flush=True)
        if config.actor_rollout_ref.actor.strategy in ["fsdp", "fsdp2"]:
            assert config.critic.strategy in ["fsdp", "fsdp2"]
            from verl.single_controller.ray import RayWorkerGroup
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker

            actor_rollout_cls = (
                AsyncActorRolloutRefWorker
                if config.actor_rollout_ref.rollout.mode == "async"
                else ActorRolloutRefWorker
            )
            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == "megatron":
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
            from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker

            actor_rollout_cls = ActorRolloutRefWorker
            ray_worker_group_cls = NVMegatronRayWorkerGroup

        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(actor_rollout_cls),
            Role.Critic: ray.remote(CriticWorker),
        }

        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }

        # reward model setup
        if config.reward_model.enable:
            if config.reward_model.strategy in ["fsdp", "fsdp2"]:
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        # use reference model
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        print("[Training] Loading reward functions...", flush=True)
        reward_fn = load_reward_manager(
            config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {})
        )
        val_reward_fn = load_reward_manager(
            config, tokenizer, num_examine=1, **config.reward_model.get("reward_kwargs", {})
        )
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        from verl.utils.dataset.rl_dataset import collate_fn

        # Use our special dataset
        print("[Training] Loading datasets...", flush=True)
        train_dataset = AgentDataset(
            data_files=config.data.train_files,
            tokenizer=tokenizer,
            processor=processor,
            config=config.data,
        )
        val_dataset = AgentDataset(
            data_files=config.data.val_files,
            tokenizer=tokenizer,
            processor=processor,
            config=config.data,
        )
        print(f"[Training] Train dataset: {len(train_dataset)} samples", flush=True)
        print(f"[Training] Val dataset: {len(val_dataset)} samples", flush=True)

        train_sampler = create_rl_sampler(config.data, train_dataset)
        print("[Training] Creating trainer...", flush=True)
        trainer = AgentFlowTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
        )
        print("[Training] Initializing workers...", flush=True)
        trainer.init_workers()
        print("[Training] Starting training (fit)...", flush=True)
        trainer.fit()
        print("[Training] Training completed!", flush=True)

    except Exception as e:
        print(f"[Training] ERROR: {e}", flush=True)
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise


if __name__ == "__main__":
    main()
