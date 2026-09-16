# Hierarchy Agents Futures

A compact research repository for minute-level Chinese futures backtesting and
a hierarchical PPO experiment. The learned system combines long- and
short-signal policies with a higher-level trader policy.

> Research software only. It is not investment advice and is not connected to
> a broker or live exchange.

## Architecture

```text
SQLite ticks -> OHLCV bars -> technical indicators
                              |-> long-signal PPO  --|
                              |-> short-signal PPO --|-> trader PPO
                                                       |
strategy target <- bounded position [-max_position, max_position]
       |
futures backtester -> trades.csv, equity.csv, summary.json, charts
```

The deterministic engine supports long and short positions, contract
multipliers, margin checks, tick-based slippage, and amount- or ratio-based
fees. The bundled market metadata currently covers **RB only**.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For development, run `make setup` and then `make check`.

Place a per-contract SQLite database at `data/db/<contract>.db`. It must contain
a `ticks` table with `ts_event_ms`, `last_price`, and cumulative `volume`.
Databases, generated datasets, reports, and trained models are intentionally
ignored by Git. Original data could be provided by reaching out to zixuanzhou2002@gmail.com .

## Baseline backtest

```bash
python -m engine.run_backtest \
  --contract rb2406 \
  --start 20240201 \
  --end 20240228 \
  --strategy ma_cross \
  --step-min 1
```

Available baseline strategies are `ma_cross` and `cash_weighted_ma`.

## Train the hierarchical PPO experiment

```bash
python -m rl.create_futures_dataset --contract rb2407
python -m rl.train_signal_agents_sb3 --contract rb2407 --train-ratio 0.6
python -m rl.train_multiagent_ppo --contract rb2407 --train-ratio 0.6
```

Evaluate only on dates after the chronological training boundary:

```bash
python -m engine.run_backtest \
  --contract rb2407 \
  --start YYYYMMDD \
  --end YYYYMMDD \
  --strategy ppo_multiagent \
  --strategy-param trader_model_path=models/rl/trader/ppo_trader_multiagent_rb2407.zip \
  --strategy-param long_model_path=models/rl/signal_agents/ppo_long_signal_rb2407.zip \
  --strategy-param short_model_path=models/rl/signal_agents/ppo_short_signal_rb2407.zip \
  --strategy-param max_position=4
```

The training environment remains deliberately lightweight. Before applying it
to another contract, update `data/meta/contracts.json` and the RL contract
settings in `rl/configs/env_configs.yaml`.

## Tests

```bash
pytest
```

## Data and rule assumptions

`data/market_rules/ini/20230101.ini` is a set of illustrative project
assumptions here, not authoritative historical exchange data. Replace it with a
dated and sourced rules file. 

As pre-claimed, original futures data could be provided by reaching out to zixuanzhou2002@gmail.com .

## Attribution

The team prototype was adapted from an MIT-licensed open-source PPO trading
project and later migrated. See
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## License

MIT. See [`LICENSE`](LICENSE).
