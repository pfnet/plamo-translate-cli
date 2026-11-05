"""Convert local PLaMo 2 or PLaMo 3 weights using an existing llama.cpp checkout and Python environment.

Official PLaMo 3 GGUFs (pfnet/plamo-3-translate-*-GGUF) need no conversion; use this for other checkpoints.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--llama-cpp-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--python", default=sys.executable, help="Python with llama.cpp conversion dependencies")
    parser.add_argument(
        "--quantization",
        nargs="+",
        choices=["Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q5_0", "Q4_0", "Q4_1", "Q3_K_M", "Q4_0-ssm-f16"],
        help="Default: Q4_0-ssm-f16 for PLaMo 2, Q4_K_M for PLaMo 3",
    )
    parser.add_argument("--imatrix", type=Path, help="Importance matrix from llama-imatrix, passed to llama-quantize")
    parser.add_argument("--name", help="Output file prefix (default: plamo-2-translate, or the PLaMo 3 directory name)")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    source, llama, output = (p.expanduser().resolve() for p in (args.model_dir, args.llama_cpp_dir, args.output_dir))
    config = json.loads((source / "config.json").read_text())
    model_type = config.get("model_type")
    if model_type not in ("plamo2", "plamo3"):
        parser.error("Expected a local PLaMo 2 or PLaMo 3 model directory")
    plamo2 = model_type == "plamo2"
    quantizations = args.quantization or (["Q4_0-ssm-f16"] if plamo2 else ["Q4_K_M"])
    if not plamo2 and "Q4_0-ssm-f16" in quantizations:
        parser.error("Q4_0-ssm-f16 keeps Mamba matrices in F16; PLaMo 3 has none")
    if args.imatrix is not None and not args.imatrix.is_file():
        parser.error(f"{args.imatrix} does not exist")
    # Name PLaMo 3 outputs after the repository: local copies are often named <user>--<repo>, and Hugging Face
    # snapshots are models--<user>--<repo>/snapshots/<commit>. Keep symlinks, which point into blobs.
    given = Path(os.path.abspath(args.model_dir.expanduser()))
    if given.parent.name == "snapshots" and given.parent.parent.name.startswith("models--"):
        given = given.parent.parent
    name = args.name or ("plamo-2-translate" if plamo2 else given.name.split("--")[-1])
    if args.threads <= 0:
        parser.error("--threads must be positive")
    converter = llama / "convert_hf_to_gguf.py"
    quantizer = llama / "build/bin/llama-quantize"
    if not converter.is_file() or not quantizer.is_file():
        parser.error("Build llama-quantize in the specified llama.cpp checkout first")
    output.mkdir(parents=True, exist_ok=True)
    commands = []

    def run(command, destination):
        if destination.exists():
            parser.error(f"Refusing to overwrite {destination}; choose a new output directory")
        temporary = destination.with_suffix(".partial.gguf")
        if temporary.exists():
            parser.error(f"Remove the incomplete file {temporary} before retrying")
        command = [str(arg) if arg != destination else str(temporary) for arg in command]
        commands.append(command)
        with (output / (destination.stem + ".log")).open("w") as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
        temporary.replace(destination)
        print(destination, flush=True)

    # PLaMo 3 checkpoints are BF16, which F16 cannot represent exactly.
    outtype = "f16" if plamo2 else "bf16"
    original = output / f"{name}-{outtype.upper()}.gguf"
    run([args.python, converter, source, "--outtype", outtype, "--outfile", original], original)
    models = [original]
    for quantization in quantizations:
        model = output / f"{name}-{quantization}.gguf"
        extra = ["--tensor-type", "ssm_out=f16"] if quantization == "Q4_0-ssm-f16" else []
        if args.imatrix is not None:
            extra += ["--imatrix", args.imatrix]
        kind = "Q4_0" if quantization == "Q4_0-ssm-f16" else quantization
        run([quantizer, *extra, original, model, kind, args.threads], model)
        models.append(model)
    if (source / "LICENSE").is_dir():
        shutil.copytree(source / "LICENSE", output / "LICENSE", dirs_exist_ok=True)
    elif (source / "LICENSE").is_file():
        shutil.copy2(source / "LICENSE", output / "LICENSE")
    revision = subprocess.check_output(["git", "-C", str(llama), "rev-parse", "HEAD"], text=True).strip()
    patch = subprocess.check_output(["git", "-C", str(llama), "diff", "HEAD"], text=True)
    (output / "llama-cpp.patch").write_text(patch, encoding="utf-8")
    files = [
        source / "config.json",
        source / "tokenizer_config.json",
        source / "tokenizer.jsonl",
        *sorted(source.glob("chat_template.jinja")),
        *sorted(source.glob("*.safetensors")),
        *([args.imatrix.resolve()] if args.imatrix is not None else []),
        *models,
    ]
    manifest = {
        "source": str(source),
        "llama_cpp_revision": revision,
        "llama_cpp_patch_sha256": hashlib.sha256(patch.encode()).hexdigest(),
        "commands": commands,
        "files": {str(p): {"bytes": p.stat().st_size, "sha256": digest(p)} for p in files},
    }
    (output / "conversion.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
