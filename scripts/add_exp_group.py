"""CLI utility: add exp_group tag to a saved test case config.

Usage:
    python scripts/add_exp_group.py <problem> <framework> <case_id> <tag>

Example:
    python scripts/add_exp_group.py poisson1d 01_pinn 0 "SampleNumTest"
"""

import argparse
import json
import sys
from pathlib import Path

from unipinn.io.workspace import get_results_dir


def main():
    parser = argparse.ArgumentParser(description="Add exp_group tag to test case config")
    parser.add_argument("problem", type=str, help="Problem name (e.g., poisson1d)")
    parser.add_argument("framework", type=str, help="Framework name (e.g., 01_pinn)")
    parser.add_argument("case_id", type=int, help="Test case index")
    parser.add_argument("tag", type=str, help="Tag to add to exp_group")
    args = parser.parse_args()

    results_dir = get_results_dir(args.problem, args.framework)
    config_file = results_dir / f"test_case_{args.case_id}" / "config.json"

    print(f"Config file: {config_file}")
    print(f"Tag to add: {args.tag}")

    if not config_file.exists():
        print(f"Error: config file not found: {config_file}")
        sys.exit(1)

    response = input("Continue? (y/n): ").strip().lower()
    if response not in ("", "y"):
        print("Aborted.")
        sys.exit(0)

    with open(config_file, "r", encoding="utf-8") as f:
        config_data = json.load(f)

    if "exp_group" not in config_data or not isinstance(config_data["exp_group"], list):
        config_data["exp_group"] = []

    if args.tag not in config_data["exp_group"]:
        config_data["exp_group"].append(args.tag)
        print(f"'{args.tag}' added to exp_group")
    else:
        print(f"'{args.tag}' already exists in exp_group")

    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config_data, f, ensure_ascii=False, indent=4)

    print("Config updated.")


if __name__ == "__main__":
    main()
