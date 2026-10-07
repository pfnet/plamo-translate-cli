"""Publish a validated MLX release atomically, then verify the remote bytes."""

import argparse
import json
from pathlib import Path

from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi, ModelCard, hf_hub_download

from plamo_translate.servers.mlx.release import file_digest, inference_files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--validation", required=True, type=Path)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--upload", action="store_true", help="Publish after preflight checks; otherwise only check locally")
    args = parser.parse_args()
    validation = json.loads(args.validation.read_text())
    assert validation["runs"] and all(
        run["full_output_exact_match"] and run["finish_reason"] == "stop" for run in validation["runs"]
    )
    files = inference_files(args.model)
    assert set(files) == set(validation["validated_files"]), "Inference file set changed after validation"
    manifest = {name: file_digest(args.model / name) for name in files}
    assert manifest == validation["validated_files"], "Inference files changed after validation"
    # Explicit allowlist keeps local paths, source copies and evaluation text private.
    for name in ["README.md", "evaluation.json", "LICENSE/en", "LICENSE/ja"]:
        manifest[name] = file_digest(args.model / name)
    evaluation = json.loads((args.model / "evaluation.json").read_text())
    assert evaluation == {key: value for key, value in validation.items() if key != "validated_files"}
    card = ModelCard.load(args.model / "README.md")
    assert card.data.library_name == "mlx"
    assert card.data.base_model == "pfnet/plamo-2-translate"
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "upload-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if not args.upload:
        print(json.dumps({"preflight": "passed", "files": len(manifest), "bytes": sum(f["bytes"] for f in manifest.values())}))
        return
    api = HfApi()
    api.auth_check(repo_id=args.repo, repo_type="model")
    info = api.model_info(args.repo, files_metadata=True)
    assert info.sha == args.expected_head, "Remote changed; inspect before retrying"
    card.validate()
    # mlx-lm globs all model*.safetensors: stale shards must not survive a release.
    stale = sorted(f.rfilename for f in info.siblings
                   if f.rfilename.startswith("model") and f.rfilename.endswith(".safetensors")
                   and f.rfilename not in manifest)
    operations = [CommitOperationAdd(path_in_repo=name, path_or_fileobj=args.model / name) for name in sorted(manifest)]
    operations.extend(CommitOperationDelete(path_in_repo=name) for name in stale)
    commit = api.create_commit(
        repo_id=args.repo, repo_type="model", revision="main", parent_commit=args.expected_head,
        operations=operations,
        commit_message=f"Refresh {validation['precision']} weights and corrected standalone MLX inference",
        commit_description=(
            f"Source: pfnet/plamo-2-translate at {validation['source_revision']}.\n\n"
            "Preserve BF16 floating weights, include configured RoPE bases and reference-compatible SSM, "
            "and correct BOS/EOS handling. Validate full output against same-precision direct inference; "
            "document the single-example scope and runtime requirements."
        ),
    )
    published = {"repo_id": args.repo, "revision": commit.oid, "commit_url": commit.commit_url,
                 "previous_head": args.expected_head, "deleted_stale_shards": stale}
    (args.output / "published.json").write_text(json.dumps(published, indent=2) + "\n")
    print(json.dumps(published), flush=True)
    remote = {f.rfilename: f for f in api.model_info(args.repo, revision=commit.oid, files_metadata=True).siblings}
    for name, expected in manifest.items():
        actual = remote[name]
        assert actual.size == expected["bytes"], name
        if actual.lfs:
            assert actual.lfs.sha256 == expected["sha256"], name
        else:
            path = Path(hf_hub_download(args.repo, name, revision=commit.oid))
            assert file_digest(path) == expected, name
    assert not set(stale) & set(remote)
    assert api.model_info(args.repo).sha == commit.oid, "Main changed after publication; pinned revision is verified"
    result = {**published, "all_file_hashes_matched": True, "files": manifest}
    (args.output / "remote-verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"Verified all {len(manifest)} remote file hashes", flush=True)


if __name__ == "__main__":
    main()
