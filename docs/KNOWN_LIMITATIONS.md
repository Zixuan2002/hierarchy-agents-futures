# Known limitations

- Only RB contract metadata is bundled.
- The recovered market-rule file contains illustrative assumptions rather than
  independently verified historical exchange rules.
- The PPO observation uses four technical indicators and unnormalized account
  features; this is a research baseline, not a production policy.
- Training uses a simple chronological split but does not yet include walk-forward
  validation, hyperparameter selection, or statistical significance tests.
- The RL environment approximates fees with a ratio and does not model separate
  open/close-today fee schedules, limit moves, queue priority, or forced liquidation.
- Model archives and proprietary tick databases are not distributed.
