import multiprocessing

from grokker_trainer import train_grokker
from trainer_config import TrainerConfig


def expt_runner_background(config: TrainerConfig, prefix: str = ""):
    process = multiprocessing.get_context("spawn").Process(
        target=train_grokker,
        args=(config,),
        kwargs={"prefix": prefix},
    )
    process.start()
    return process


def all_expts_config():
    configs = []
    config_prefixes = []

    for optimizer in ["sgd", "adam"]:
        for lr in [1e-3, 1e-2]:
            for weight_decay in [0.0, 2e-3]:#, 2e-2, 2e-1]:
                cfg = TrainerConfig(lr=lr, weight_decay=weight_decay, optimizer=optimizer)
                configs.append(cfg)
                prefix = f"{optimizer}_lr{lr}_wd{weight_decay}"
                config_prefixes.append(prefix)
    return configs, config_prefixes


if __name__ == "__main__":
    expt_configs, config_prefixes = all_expts_config()
    for i, cfg in enumerate(expt_configs):
        print(f"Starting experiment {i+1}/{len(expt_configs)} with config:\n{cfg.to_json()}\n")
        prefix = config_prefixes[i]
        expt_runner_background(cfg, prefix=prefix)