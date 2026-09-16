# rl/train_signal_agents_sb3.py

from __future__ import annotations
import argparse
from pathlib import Path

import pandas as pd
import yaml
from stable_baselines3 import PPO as SB3PPO
from stable_baselines3.common.monitor import Monitor

from .signal_envs import LongSignalEnv, ShortSignalEnv


def main():
    parser = argparse.ArgumentParser(description="Train long/short signal PPO agents on futures dataset.")
    parser.add_argument(
        "--contract",
        type=str,
        required=True,
        help="Futures contract code used in processed_dataset_{contract}.csv",
    )
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    contract = args.contract

    root_dir = Path(__file__).resolve().parents[1]  # futures-agent/
    rl_dir = root_dir / "rl"

    # Load env configs
    with open(rl_dir / "configs" / "env_configs.yaml", "r") as f:
        env_configs = yaml.safe_load(f)

    # Load dataset
    data_path = root_dir / "data" / "rl" / f"processed_dataset_{contract}.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}. "
                                f"Run: python -m rl.create_futures_dataset --contract {contract}")

    dataset = pd.read_csv(data_path, index_col=0)

    start_day = int(dataset.index.min())
    full_end = int(dataset.index.max())
    if not 0.0 < args.train_ratio < 1.0:
        raise ValueError("--train-ratio must be between 0 and 1")
    end_day = start_day + int((full_end - start_day) * args.train_ratio)
    seed = int(args.seed if args.seed is not None else env_configs.get("seed", 42))

    long_env = LongSignalEnv(dataset, start_day=start_day, end_day=end_day)
    short_env = ShortSignalEnv(dataset, start_day=start_day, end_day=end_day)

    # Wrap with Monitor
    long_env = Monitor(long_env)
    short_env = Monitor(short_env)

    models_dir = root_dir / "models" / "rl" / "signal_agents"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Long-signal PPO
    long_model = SB3PPO(
        "MlpPolicy",
        long_env,
        verbose=1,
        n_steps=2048,
        batch_size=256,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        seed=seed,
    )
    long_model.learn(total_timesteps=300_000, progress_bar=True)
    long_out = models_dir / f"ppo_long_signal_{contract}.zip"
    long_model.save(long_out)
    print(f"[train_signal_agents_sb3] Saved long-signal model to {long_out}")

    # Short-signal PPO
    short_model = SB3PPO(
        "MlpPolicy",
        short_env,
        verbose=1,
        n_steps=2048,
        batch_size=256,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        seed=seed,
    )
    short_model.learn(total_timesteps=300_000, progress_bar=True)
    short_out = models_dir / f"ppo_short_signal_{contract}.zip"
    short_model.save(short_out)
    print(f"[train_signal_agents_sb3] Saved short-signal model to {short_out}")


if __name__ == "__main__":
    main()
