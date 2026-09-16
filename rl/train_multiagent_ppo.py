# rl/train_multiagent_ppo.py

from __future__ import annotations
import argparse
from pathlib import Path

import pandas as pd
import yaml
from stable_baselines3 import PPO as SB3PPO
from stable_baselines3.common.monitor import Monitor

from .stock_env import StockEnvConfig
from .trading_env_with_signals import TradingEnvWithSignals


def main():
    parser = argparse.ArgumentParser(description="Train multiagent PPO trading agent with signals.")
    parser.add_argument(
        "--contract",
        type=str,
        required=True,
        help="Futures contract code used in processed_dataset_{contract}.csv and signal models.",
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

    config = StockEnvConfig(
        initial_balance=float(env_configs.get("initial_balance", 1_000_000.0)),
        transaction_fee=float(env_configs.get("transaction_fee", 0.001)),
        hmax=int(env_configs.get("hmax", 100)),
        reward_scaling=float(env_configs.get("reward_scaling", 1e-4)),
        nb_stock=int(env_configs.get("nb_stock", 1)),
        multiplier=float(env_configs.get("multiplier", 10.0)),
        margin_ratio=float(env_configs.get("margin_ratio", 0.21)),
    )

    # Load dataset
    data_path = root_dir / "data" / "rl" / f"processed_dataset_{contract}.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}. "
                                f"Run: python -m rl.create_futures_dataset --contract {contract}")

    dataframe = pd.read_csv(data_path, index_col=0)

    start_day = int(dataframe.index.min())
    full_end = int(dataframe.index.max())
    if not 0.0 < args.train_ratio < 1.0:
        raise ValueError("--train-ratio must be between 0 and 1")
    end_day = start_day + int((full_end - start_day) * args.train_ratio)
    seed = int(args.seed if args.seed is not None else env_configs.get("seed", 42))

    # Load pre-trained signal models
    models_dir = root_dir / "models" / "rl"
    long_model_path = models_dir / "signal_agents" / f"ppo_long_signal_{contract}.zip"
    short_model_path = models_dir / "signal_agents" / f"ppo_short_signal_{contract}.zip"

    if not long_model_path.exists() or not short_model_path.exists():
        raise FileNotFoundError(
            f"Signal models not found. Expected:\n"
            f"  {long_model_path}\n"
            f"  {short_model_path}\n"
            f"Run: python -m rl.train_signal_agents_sb3 --contract {contract}"
        )

    long_model = SB3PPO.load(long_model_path)
    short_model = SB3PPO.load(short_model_path)

    # Build trading env with signals
    env = TradingEnvWithSignals(
        dataframe=dataframe,
        config=config,
        long_model=long_model,
        short_model=short_model,
        start_day=start_day,
        end_day=end_day,
    )
    env = Monitor(env)

    trader_model = SB3PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=2048,
        batch_size=256,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        seed=seed,
    )
    trader_model.learn(total_timesteps=300_000, progress_bar=True)

    trader_dir = models_dir / "trader"
    trader_dir.mkdir(parents=True, exist_ok=True)
    trader_out = trader_dir / f"ppo_trader_multiagent_{contract}.zip"
    trader_model.save(trader_out)
    print(f"[train_multiagent_ppo] Saved trader model to {trader_out}")


if __name__ == "__main__":
    main()
