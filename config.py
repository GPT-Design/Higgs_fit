# version 0.4
"""
Configuration for the α‑aware Higgs entropy fit – now supports 2‑D (κ, c) sweeps.
"""
from dataclasses import dataclass
from pathlib import Path
import argparse


@dataclass
class Config:
    """Container for all tunable parameters used by the sweep pipeline."""

    # I/O --------------------------------------------------------------------
    data_path: Path = None  # None means batch process data directory
    output: Path = Path("results")
    calibration: bool = False  # Use toy.csv for calibration
    batch_mode: bool = False  # Process all files in data directory

    # Entropy scaling --------------------------------------------------------
    alpha: float = 0.04      # coupling α
    alpha_err: float = 0.0005
    S_scalar: float = 0.10   # base S₀

    # κ sweep ---------------------------------------------------------------
    kappa_min: float = 0.0
    kappa_max: float = 2.0
    kappa_steps: int = 50

    # c sweep (log‑running coefficient) -------------------------------------
    c_min: float = 0.0
    c_max: float = 1.5
    c_steps: int = 25

    # Physics model ----------------------------------------------------------
    model: str = "breit_wigner_entropy"
    entropy_shape: str = "log"  # "constant", "powerlaw", "log"
    phangs_mode: bool = False  # Use PHANGS shock fitting

    # -----------------------------------------------------------------------
    # Validation helpers
    # -----------------------------------------------------------------------
    def validate(self) -> None:
        # Handle different modes
        if self.phangs_mode:
            # PHANGS mode - set model and skip standard validation
            self.model = "phangs_shock"
            return  # Skip standard file validation for PHANGS
        elif self.calibration:
            # Use toy.csv for calibration
            self.data_path = Path("toy.csv")
            if not self.data_path.exists():
                raise ValueError("toy.csv not found for calibration mode")
        elif self.data_path is None:
            # Batch mode - process data directory
            self.batch_mode = True
            data_dir = Path("data")
            if not data_dir.exists():
                raise ValueError("Data directory 'data' not found for batch processing")
        else:
            # Single file mode - check if file exists
            if not self.data_path.exists():
                data_dir_path = Path("data") / self.data_path.name
                if data_dir_path.exists():
                    self.data_path = data_dir_path
                else:
                    raise ValueError(f"Data file not found: {self.data_path} (also checked data/{self.data_path.name})")

        if self.kappa_min > self.kappa_max:
            raise ValueError(f"kappa_min ({self.kappa_min}) must be <= kappa_max ({self.kappa_max})")
        if self.c_min > self.c_max:
            raise ValueError(f"c_min ({self.c_min}) must be <= c_max ({self.c_max})")

        if self.kappa_steps < 1 or self.c_steps < 1:
            raise ValueError(f"step counts must be ≥ 1, got kappa_steps={self.kappa_steps}, c_steps={self.c_steps}")

        if self.alpha < 0:
            raise ValueError(f"alpha must be non‑negative, got {self.alpha}")

        self.output.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------------
# CLI argument parsing
# ----------------------------------------------------------------------------

def parse_args() -> "Config":
    p = argparse.ArgumentParser("alpha-aware Higgs entropy grid-sweep")

    # optional data file - if not provided, batch process data directory
    p.add_argument("data", nargs="?", type=Path, 
                   help="Data file (CSV/Parquet/HDF5). If not provided, batch processes all files in data/ directory")
    
    # calibration mode
    p.add_argument("--calibration", action="store_true", 
                   help="Use toy.csv for calibration (overrides data argument)")
    
    # PHANGS mode
    p.add_argument("--phangs", action="store_true",
                   help="Use PHANGS shock fitting mode")

    # entropy scalars
    p.add_argument("--alpha", type=float, default=0.04)
    p.add_argument("--alpha-err", type=float, default=0.0005)
    p.add_argument("--S", dest="S_scalar", type=float, default=0.10)

    # kappa range/steps
    p.add_argument("--kappa-range", nargs=2, type=float, metavar=("MIN", "MAX"), default=(0.0, 2.0))
    p.add_argument("--kappa-steps", type=int, default=50)

    # c range/steps
    p.add_argument("--c-range", nargs=2, type=float, metavar=("MIN", "MAX"), default=(0.0, 1.5))
    p.add_argument("--c-steps", type=int, default=25)

    # misc
    p.add_argument("--model", default="breit_wigner_entropy")
    p.add_argument("--entropy-shape", dest="entropy_shape", default="log", 
                   choices=["constant", "powerlaw", "log"],
                   help="Entropy profile type")
    p.add_argument("--output", type=Path, default=Path("results"))

    a = p.parse_args()

    cfg = Config(
        data_path=a.data,
        calibration=a.calibration,
        phangs_mode=a.phangs,
        alpha=a.alpha,
        alpha_err=a.alpha_err,
        S_scalar=a.S_scalar,
        kappa_min=a.kappa_range[0],
        kappa_max=a.kappa_range[1],
        kappa_steps=a.kappa_steps,
        c_min=a.c_range[0],
        c_max=a.c_range[1],
        c_steps=a.c_steps,
        model=a.model,
        entropy_shape=a.entropy_shape,
        output=a.output,
    )

    cfg.validate()
    return cfg


if __name__ == "__main__":  # simple smoke
    print(parse_args())
