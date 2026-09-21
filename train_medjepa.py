"""Train LeJEPA and then generate the configured PCA and probe reports."""

import argparse
from core.config import load_run_config
from core.training import run_training


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-config", required=True)
    args = parser.parse_args()

    result = run_training(args.run_config)
    # All ranks have finished training and destroyed the process group here.
    # Only rank zero receives the checkpoint/model and runs analysis.
    if result is not None and load_run_config(args.run_config)["paths"].get("analysis_config"):
        from core.features import run_analysis

        checkpoint, model = result
        run_analysis(checkpoint, args.run_config, model=model)


if __name__ == "__main__":
    main()
