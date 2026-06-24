"""Generate a JSON config file from a Poisson1DVanillaPINNConfig instance.

Usage:
    python scripts/generate_config.py --output <path> [--modify key=value ...]

Example:
    python scripts/generate_config.py --output configs/test_0.json --modify optimizer=lbfgs epochs=4000

This script creates a default config, applies any --modify overrides, and writes JSON.
"""

import argparse
import json
from dataclasses import asdict

from unipinn.config.poisson1d import Poisson1DVanillaPINNConfig
from unipinn.io.artifacts import SafeJSONEncoder


def apply_overrides(cfg_dict: dict, overrides: list) -> dict:
    """Apply key=value overrides to a config dictionary.

    Handles type coercion for numeric fields.
    """
    for override in overrides:
        if "=" not in override:
            print(f"Warning: skipping invalid override '{override}' (expected key=value)")
            continue
        key, value = override.split("=", 1)
        key = key.strip()
        value = value.strip()

        # Attempt type coercion
        if value.lower() in ("true", "false"):
            value = value.lower() == "true"
        else:
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    pass  # Keep as string

        if key in cfg_dict:
            old = cfg_dict[key]
            cfg_dict[key] = value
            print(f"  {key}: {old} -> {value}")
        else:
            print(f"  Warning: unknown key '{key}', adding anyway")
            cfg_dict[key] = value

    return cfg_dict


def main():
    parser = argparse.ArgumentParser(description="Generate experiment config JSON")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output JSON path")
    parser.add_argument("--modify", "-m", nargs="*", default=[], help="key=value overrides")
    args = parser.parse_args()

    cfg = Poisson1DVanillaPINNConfig()
    cfg_dict = asdict(cfg)

    if args.modify:
        print("Applying overrides:")
        cfg_dict = apply_overrides(cfg_dict, args.modify)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(cfg_dict, f, indent=2, cls=SafeJSONEncoder, ensure_ascii=False)

    print(f"\nConfig written to: {args.output}")


if __name__ == "__main__":
    main()
