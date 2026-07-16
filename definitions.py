import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# Root of the data layout documented in README.md. Override with the HEAT_DATA_DIR
# env var to point at data stored outside the repo (e.g. on another drive/volume).
DATA_DIR = Path(os.environ.get("HEAT_DATA_DIR", PROJECT_ROOT / "data"))

# Fixed subdirectory names created under each run's output directory (the Hydra
# run dir; see run_output_dir below). These are output-layout conventions, not
# user-tunable config, so they live here rather than in conf/config.yaml.
TEST_DATA_DIRNAME = "TestData"
MODEL_DIRNAME = "Model"
METRICS_DIRNAME = "Metrics"
PREDICTIONS_DIRNAME = "Predictions"
TRUTH_PREDICTION_DIRNAME = "TruthPrediction"


def run_output_dir() -> str:
    """Return the current Hydra run directory, with a trailing separator.

    All run artifacts are written beneath this directory. It is only defined
    while a Hydra job is active (Hydra sets it up in ``main.main``), which is
    the only context in which the output-writing helpers run.
    """
    from hydra.core.hydra_config import HydraConfig
    return HydraConfig.get().runtime.output_dir + "/"
