# version 0.3
"""
Command‑line interface for the α‑aware Higgs entropy *grid‑sweep* package.

Simply parses CLI flags (via `config.parse_args`) and hands the resulting
`Config` to `sweeps.main`.
"""

from config import parse_args
from sweeps import main as run_sweep


def main_cli() -> None:
    """Entry point: parse args, run the grid sweep."""
    cfg = parse_args()
    run_sweep(cfg)


if __name__ == "__main__":
    main_cli()
