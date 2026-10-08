#!/usr/bin/env python3
"""Download a pinned official Qwen GGUF or build a separate llama.cpp runtime.

No Python packages, global compilers, ROS files or system services are changed.
All artifacts are under third_party/llm_* in this project.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
MODEL_REPOSITORY = "Qwen/Qwen3-4B-GGUF"
MODEL_REVISION = "bc640142c66e1fdd12af0bd68f40445458f3869b"
MODEL_FILE = "Qwen3-4B-Q4_K_M.gguf"
MODEL_SHA256 = "7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5"
MODEL_BYTES = 2497280256
RUNTIME_TAG = "v0.6.0"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def request(url: str):
    return urllib.request.Request(url, headers={"User-Agent": "opti-project-local-llm-setup"})


def download(url: str, destination: Path, expected_sha: str | None = None) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        actual = sha256(destination)
        if expected_sha and actual != expected_sha:
            raise RuntimeError(f"Existing file has an unexpected SHA256: {destination}")
        print(f"Verified existing file: {destination.name}", flush=True)
        return actual
    partial = destination.with_suffix(destination.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    req = request(url)
    if offset:
        req.add_header("Range", f"bytes={offset}-")
    start = time.monotonic()
    with urllib.request.urlopen(req, timeout=120) as response:
        append = offset > 0 and response.status == 206
        total = offset if append else 0
        announced = total
        with partial.open("ab" if append else "wb") as stream:
            while block := response.read(8 * 1024 * 1024):
                stream.write(block)
                total += len(block)
                if total - announced >= 256 * 1024 * 1024:
                    print(f"{destination.name}: {total} bytes received", flush=True)
                    announced = total
    actual = sha256(partial)
    if expected_sha and actual != expected_sha:
        raise RuntimeError(f"SHA256 mismatch for {destination.name}: {actual}")
    partial.replace(destination)
    print(f"Downloaded {destination.name}: {destination.stat().st_size} bytes, "
          f"SHA256={actual}, elapsed={time.monotonic() - start:.1f}s", flush=True)
    return actual


def setup_model() -> None:
    directory = ROOT / "third_party" / "llm_models" / "qwen3_4b_q4_k_m"
    url = f"https://huggingface.co/{MODEL_REPOSITORY}/resolve/{MODEL_REVISION}/{MODEL_FILE}"
    digest = download(url, directory / MODEL_FILE, MODEL_SHA256)
    if (directory / MODEL_FILE).stat().st_size != MODEL_BYTES:
        raise RuntimeError("Model size differs from official Hugging Face LFS metadata")
    for filename in ["LICENSE", "README.md"]:
        download(f"https://huggingface.co/{MODEL_REPOSITORY}/resolve/{MODEL_REVISION}/{filename}",
                 directory / filename)
    provenance = {"repository": MODEL_REPOSITORY, "revision": MODEL_REVISION,
                  "file": MODEL_FILE, "sha256": digest, "bytes": MODEL_BYTES,
                  "license": "Apache-2.0", "official_distribution": True,
                  "source_url": f"https://huggingface.co/{MODEL_REPOSITORY}",
                  "training_stage": "pretraining and post-training", "quantization": "Q4_K_M",
                  "additional_project_training": False}
    (directory / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Model ready: {directory / MODEL_FILE}", flush=True)


def setup_runtime(cuda: bool, cuda_architecture: str) -> None:
    directory = ROOT / "third_party" / "llm_runtime"
    directory.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(request(
            f"https://api.github.com/repos/ggml-org/llama.cpp/git/ref/tags/{RUNTIME_TAG}"), timeout=30) as response:
        tag = json.load(response)
    obj = tag["object"]
    if obj["type"] == "tag":
        with urllib.request.urlopen(request(obj["url"]), timeout=30) as response:
            obj = json.load(response)["object"]
    commit = obj["sha"]
    archive = directory / f"llama_cpp_{RUNTIME_TAG}.tar.gz"
    digest = download(f"https://codeload.github.com/ggml-org/llama.cpp/tar.gz/{commit}", archive)
    source = directory / f"llama.cpp-{commit}"
    if not source.exists():
        with tarfile.open(archive) as tar:
            for member in tar.getmembers():
                target = (directory / member.name).resolve()
                if not target.is_relative_to(directory.resolve()):
                    raise RuntimeError("Archive contains an unsafe path")
                if member.issym() or member.islnk():
                    link_target = (target.parent / member.linkname).resolve()
                    if not link_target.is_relative_to(directory.resolve()):
                        raise RuntimeError("Archive contains an unsafe link")
            tar.extractall(directory)
    build = directory / ("build_cuda" if cuda else "build_cpu")
    command = ["cmake", "-S", str(source), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release",
               "-DLLAMA_CURL=OFF", "-DLLAMA_BUILD_TESTS=OFF", "-DGGML_NATIVE=OFF",
               "-DGGML_CUDA=" + ("ON" if cuda else "OFF")]
    if cuda:
        compiler = shutil.which("nvcc")
        if compiler is None:
            raise RuntimeError("CUDA build requested, but nvcc was not found on PATH")
        command += [f"-DCMAKE_CUDA_COMPILER={compiler}",
                    f"-DCMAKE_CUDA_ARCHITECTURES={cuda_architecture}", "-DGGML_CUDA_FA_ALL_QUANTS=OFF"]
    start = time.monotonic()
    subprocess.run(command, check=True, cwd=directory)
    subprocess.run(["cmake", "--build", str(build), "--parallel", "4", "--target", "llama-server"],
                   check=True, cwd=directory)
    executable = build / "bin" / "llama-server"
    result = subprocess.run([str(executable), "--version"], capture_output=True, text=True, check=True)
    provenance = {"repository": "ggml-org/llama.cpp", "tag": RUNTIME_TAG, "commit": commit,
                  "source_archive_sha256": digest, "license": "MIT",
                  "cuda_requested": cuda, "cuda_architecture": cuda_architecture if cuda else None,
                  "build_command": command, "build_elapsed_seconds": time.monotonic() - start,
                  "executable": str(executable.relative_to(ROOT)),
                  "executable_sha256": sha256(executable), "version_output": result.stdout + result.stderr}
    (directory / ("provenance_cuda.json" if cuda else "provenance_cpu.json")).write_text(
        json.dumps(provenance, indent=2) + "\n")
    print(f"Runtime ready: {executable}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("component", choices=["model", "runtime"])
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--cuda-architecture", default="89",
                        help="CMake CUDA architecture; 89 is the tested RTX 4060 Ti target")
    args = parser.parse_args()
    if args.component == "model":
        setup_model()
    else:
        setup_runtime(args.cuda, args.cuda_architecture)


if __name__ == "__main__":
    main()
