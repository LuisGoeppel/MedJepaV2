"""Generate PCA and linear-probe reports for an existing checkpoint."""

import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--run-config", required=True, help="Training JSON providing dataset paths")
    parser.add_argument("--analysis-config", help="Override the referenced analysis JSON")
    parser.add_argument("--output-dir", help="Default: training output_dir/analysis")
    args = parser.parse_args()
    from core.features import run_analysis

    run_analysis(args.checkpoint, args.run_config, args.analysis_config, args.output_dir)


if __name__ == "__main__":
    main()
