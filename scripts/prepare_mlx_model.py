"""Export a standalone MLX checkpoint without modifying the source weights."""

import argparse
import fcntl
import json
import shutil
from pathlib import Path

from mlx_lm.utils import save

from plamo_translate.servers.mlx.loader import load_translation_model
from plamo_translate.servers.mlx.release import write_inference_assets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--precision", choices=["4bit", "6bit", "8bit", "mixed", "bf16"], default="4bit")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("The output directory already exists; choose a new directory")
    model, tokenizer, config = load_translation_model(args.model, args.precision, optimize=False)
    save(args.output, args.model, model, tokenizer, config)
    write_inference_assets(args.output, config, sorted(tokenizer.eos_token_ids))
    source = Path(args.model)
    if source.is_dir():
        if (source / "LICENSE").is_dir():
            shutil.copytree(source / "LICENSE", args.output / "LICENSE")
        if (source / "README.md").is_file():
            shutil.copy2(source / "README.md", args.output / "SOURCE_README.md")
    (args.output / "plamo-translate-source.json").write_text(
        json.dumps(
            {
                "source": args.model,
                "precision": args.precision,
                "backend": "plamo_translate.servers.mlx",
                "rope_from_checkpoint": True,
                "standalone_model_file": "modeling_mlx_plamo2.py",
            },
            indent=2,
        )
    )
    print(f"Saved {args.precision} checkpoint to {args.output}")


if __name__ == "__main__":
    with open("/tmp/plamo-translate-benchmark.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        main()
