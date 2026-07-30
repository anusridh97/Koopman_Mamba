"""Dependency-light integrity tests for Phase 2a frozen inputs."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import pytest
except ImportError:  # Keeps direct unittest execution dependency-free.
    pytest = None

if pytest is not None:
    pytestmark = pytest.mark.correctness


ROOT = Path(__file__).resolve().parents[1]
PHASE2 = ROOT / "koopman_lm" / "experiments" / "phase2"
sys.path.insert(0, str(PHASE2))

from preflight import _data_contract_failures  # noqa: E402
from runtime_fingerprint import (  # noqa: E402
    _CRITICAL_IMPORTS,
    build_runtime_lock,
    verify_runtime_lock,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _healthy_runtime() -> dict:
    return {
        "python": {
            "implementation": "CPython",
            "version": "3.11.9",
            "executable": "/approved/venv/bin/python",
        },
        "platform": {
            "system": "Linux",
            "release": "test-kernel",
            "machine": "x86_64",
            "libc": ["glibc", "2.31"],
        },
        "packages": {
            "koopman_lm": "0.5.0",
            "torch": "2.6.0",
            "numpy": "2.0.0",
            "mamba-ssm": "2.2.4",
            "causal-conv1d": "1.5.0",
            "triton": "3.2.0",
            "transformers": "4.48.0",
            "datasets": "3.2.0",
            "safetensors": "0.5.0",
            "wandb": "0.19.0",
            "PyYAML": "6.0.2",
            "optuna": "4.2.0",
            "scikit-learn": "1.6.0",
            "psycopg": "3.2.0",
        },
        "critical_imports": {
            module_name: {"ok": True} for module_name in _CRITICAL_IMPORTS
        },
        "torch": {
            "available": True,
            "version": "2.6.0+cu124",
            "git_version": "abc",
            "cuda_build": "12.4",
            "hip_build": None,
            "cuda_available": True,
            "device_count": 1,
            "devices": [
                {
                    "name": "NVIDIA B200",
                    "compute_capability": [10, 0],
                    "total_memory_bytes": 1000,
                }
            ],
            "compiled_cuda_arches": ["sm_100"],
            "cudnn_version": 90100,
            "cxx11_abi": True,
            "gpu_smoke": {
                "ok": True,
                "operation": "cuda_float32_matmul_2x2",
                "result": 16.0,
            },
        },
        "cuda_toolkit": {
            "cuda_home": "/usr/local/cuda",
            "cuda_path": None,
            "nvcc_version_output": "Cuda compilation tools, release 12.4",
            "driver_versions": ["550.54"],
        },
    }


class RuntimeLockTests(unittest.TestCase):
    def test_runtime_lock_checks_self_hash_and_active_runtime(self) -> None:
        runtime = _healthy_runtime()
        lock = build_runtime_lock(runtime)
        self.assertEqual(
            verify_runtime_lock(lock, current_fingerprint=runtime),
            [],
        )

        changed = copy.deepcopy(runtime)
        changed["packages"]["triton"] = "different"
        failures = verify_runtime_lock(lock, current_fingerprint=changed)
        self.assertIn("ACTIVE_RUNTIME_MISMATCH", {item["code"] for item in failures})
        self.assertTrue(any("packages.triton" in item["detail"] for item in failures))

        corrupt = copy.deepcopy(lock)
        corrupt["fingerprint_sha256"] = "0" * 64
        failures = verify_runtime_lock(corrupt, current_fingerprint=runtime)
        self.assertIn(
            "RUNTIME_LOCK_SELF_HASH_MISMATCH",
            {item["code"] for item in failures},
        )

    def test_runtime_lock_rejects_a_reproducible_but_cpu_only_stack(self) -> None:
        runtime = _healthy_runtime()
        runtime["torch"]["cuda_available"] = False
        runtime["torch"]["device_count"] = 0
        runtime["torch"]["devices"] = []
        failures = verify_runtime_lock(
            build_runtime_lock(runtime),
            current_fingerprint=runtime,
        )
        self.assertIn("CUDA_RUNTIME_UNAVAILABLE", {item["code"] for item in failures})

    def test_runtime_lock_rejects_more_than_one_visible_gpu(self) -> None:
        runtime = _healthy_runtime()
        runtime["torch"]["device_count"] = 2
        runtime["torch"]["devices"].append(
            copy.deepcopy(runtime["torch"]["devices"][0])
        )
        failures = verify_runtime_lock(
            build_runtime_lock(runtime),
            current_fingerprint=runtime,
        )
        self.assertIn(
            "EXACTLY_ONE_VISIBLE_CUDA_DEVICE_REQUIRED",
            {item["code"] for item in failures},
        )

    def test_runtime_lock_rejects_broken_extension_or_cuda_smoke(self) -> None:
        runtime = _healthy_runtime()
        runtime["critical_imports"]["causal_conv1d_cuda"] = {
            "ok": False,
            "error_type": "ImportError",
        }
        runtime["torch"]["gpu_smoke"]["ok"] = False
        failures = verify_runtime_lock(
            build_runtime_lock(runtime),
            current_fingerprint=runtime,
        )
        codes = {item["code"] for item in failures}
        self.assertIn("CRITICAL_RUNTIME_IMPORT_FAILED", codes)
        self.assertIn("CUDA_SMOKE_TEST_FAILED", codes)


class FrozenArtifactTests(unittest.TestCase):
    def test_all_three_artifacts_are_hashed_and_sized(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train.tokens"
            validation = root / "validation.tokens"
            mqar = root / "mqar.jsonl"
            train.write_bytes(b"frozen-train")
            validation.write_bytes(b"frozen-validation")
            mqar.write_bytes(b"frozen-mqar")
            auxiliary_paths: dict[str, Path] = {}
            for index, name in enumerate(
                (
                    "tokenizer_fingerprint",
                    "training_shard_order_manifest",
                    "mqar_oracle_fixture",
                    "mqar_sample_ids",
                    "mqar_vocabulary_identity",
                    "mqar_token_map",
                    "birdie_selective_copy_fixture",
                    "birdie_infilling_fixture",
                    "birdie_sample_identity_manifest",
                )
            ):
                path = root / f"{name}.json"
                path.write_text(
                    json.dumps(
                        {
                            "fixture_name": name,
                            "fixture_revision": index,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                auxiliary_paths[name] = path

            spec = {
                "protocol": {
                    "tokenizer": "mistralai/Mistral-7B-v0.1",
                    "tokenizer_revision": "tok-rev",
                    "train_dataset": "FineWeb-Edu",
                    "train_dataset_revision": "train-rev",
                    "validation_dataset": "WikiText-103",
                    "validation_dataset_revision": "validation-rev",
                    "sequence_length": 2048,
                    "loss_normalization": (
                        "mean_over_supervised_tokens_then_weighted_objective_sum"
                    ),
                    "data_stream": {
                        "sampler": "deterministic_global_sequence_index",
                        "sampler_revision": "sampler-v1",
                        "shuffle_seed": 42,
                        "shard_order_policy": "frozen-manifest-order",
                        "shard_order_manifest_sha256": _sha256(
                            auxiliary_paths["training_shard_order_manifest"]
                        ),
                        "epoch_policy": "single-pass-then-repeat",
                        "repeat_policy": "restart-identical-order",
                        "worker_partitioning": (
                            "global_sequence_index modulo world_size after "
                            "deterministic ordering"
                        ),
                    },
                }
            }
            manifest = {
                "schema_version": 1,
                "tokenizer": {
                    "name": "mistralai/Mistral-7B-v0.1",
                    "revision": "tok-rev",
                    "vocab_size": 32000,
                    "fingerprint_sha256": _sha256(
                        auxiliary_paths["tokenizer_fingerprint"]
                    ),
                },
                "training": {
                    "dataset": "FineWeb-Edu",
                    "revision": "train-rev",
                    "split": "train",
                    "document_range": "0:100",
                    "packing": {
                        "sequence_length": 2048,
                        "eos_between_documents": True,
                        "drop_remainder": True,
                    },
                    "stream": {
                        "sampler": "deterministic_global_sequence_index",
                        "sampler_revision": "sampler-v1",
                        "shuffle_seed": 42,
                        "shard_order_policy": "frozen-manifest-order",
                        "shard_order_manifest_sha256": _sha256(
                            auxiliary_paths["training_shard_order_manifest"]
                        ),
                        "epoch_policy": "single-pass-then-repeat",
                        "repeat_policy": "restart-identical-order",
                        "worker_partitioning": (
                            "global_sequence_index modulo world_size after "
                            "deterministic ordering"
                        ),
                    },
                    "token_file": str(train),
                    "token_count": 2048,
                    "byte_size": train.stat().st_size,
                    "sha256": _sha256(train),
                },
                "wikitext_validation": {
                    "dataset": "WikiText-103",
                    "revision": "validation-rev",
                    "split": "validation",
                    "maximum_tokens": 2048,
                    "packing": "contiguous",
                    "token_file": str(validation),
                    "byte_size": validation.stat().st_size,
                    "sha256": _sha256(validation),
                },
                "mqar_screening": {
                    "generator": "pinned oracle",
                    "generator_revision": "oracle-v1",
                    "oracle_fixture_sha256": _sha256(
                        auxiliary_paths["mqar_oracle_fixture"]
                    ),
                    "sequence_lengths": [256, 512, 1024, 2048],
                    "key_value_pairs": [4, 8, 16, 32, 64],
                    "samples_per_cell": 8,
                    "seed": 42,
                    "sample_ids_sha256": _sha256(
                        auxiliary_paths["mqar_sample_ids"]
                    ),
                    "artifact_file": str(mqar),
                    "byte_size": mqar.stat().st_size,
                    "sha256": _sha256(mqar),
                    "vocabulary": {
                        "size": 128,
                        "identity_sha256": _sha256(
                            auxiliary_paths["mqar_vocabulary_identity"]
                        ),
                    },
                    "token_map": {
                        "revision": "token-map-v1",
                        "sha256": _sha256(
                            auxiliary_paths["mqar_token_map"]
                        ),
                    },
                },
                "birdie_training_objectives": {
                    "mixer_semantics": (
                        "static Optuna mixture fixed for the whole trial"
                    ),
                    "mixer_revision": "mixer-v1",
                    "selective_copy": {
                        "generator": "copy-v1",
                        "generator_revision": "copy-rev",
                        "seed_policy": "trial-seed-derived",
                        "fixture_sha256": _sha256(
                            auxiliary_paths["birdie_selective_copy_fixture"]
                        ),
                    },
                    "infilling": {
                        "generator": "infill-v1",
                        "generator_revision": "infill-rev",
                        "seed_policy": "trial-seed-derived",
                        "fixture_sha256": _sha256(
                            auxiliary_paths["birdie_infilling_fixture"]
                        ),
                    },
                    "sample_identity_manifest_sha256": _sha256(
                        auxiliary_paths["birdie_sample_identity_manifest"]
                    ),
                    "loss_normalization": (
                        "mean_over_supervised_tokens_then_weighted_objective_sum"
                    ),
                    "equal_token_accounting": True,
                },
                "auxiliary_artifacts": {
                    name: {
                        "path": str(path),
                        "byte_size": path.stat().st_size,
                        "sha256": _sha256(path),
                    }
                    for name, path in auxiliary_paths.items()
                },
            }
            for path in (train, validation, mqar, *auxiliary_paths.values()):
                path.chmod(0o444)
            self.assertEqual(_data_contract_failures(spec, manifest), [])

            aliased = copy.deepcopy(manifest)
            aliased["wikitext_validation"].update(
                {
                    "token_file": str(train),
                    "byte_size": train.stat().st_size,
                    "sha256": _sha256(train),
                }
            )
            failures = _data_contract_failures(spec, aliased)
            self.assertIn(
                "DATA_SPLIT_ALIAS",
                {item["code"] for item in failures},
            )

            tokenizer_fixture = auxiliary_paths["tokenizer_fingerprint"]
            auxiliary_alias = copy.deepcopy(manifest)
            tokenizer_entry = copy.deepcopy(
                auxiliary_alias["auxiliary_artifacts"]["tokenizer_fingerprint"]
            )
            auxiliary_alias["auxiliary_artifacts"]["mqar_sample_ids"] = (
                tokenizer_entry
            )
            auxiliary_alias["mqar_screening"]["sample_ids_sha256"] = (
                tokenizer_entry["sha256"]
            )
            failures = _data_contract_failures(spec, auxiliary_alias)
            self.assertIn(
                "DATA_SPLIT_ALIAS",
                {item["code"] for item in failures},
            )

            tokenizer_fixture.chmod(0o644)
            failures = _data_contract_failures(spec, manifest)
            self.assertIn(
                "WRITABLE_AUXILIARY_DATA_FILE",
                {item["code"] for item in failures},
            )
            tokenizer_fixture.chmod(0o444)

            original_tokenizer_fixture = tokenizer_fixture.read_bytes()
            tokenizer_fixture.chmod(0o644)
            tokenizer_fixture.write_bytes(b"tampered-tokenizer-fixture")
            failures = _data_contract_failures(spec, manifest)
            self.assertIn(
                "AUXILIARY_DATA_HASH_MISMATCH",
                {item["code"] for item in failures},
            )
            tokenizer_fixture.write_bytes(original_tokenizer_fixture)
            tokenizer_fixture.chmod(0o444)

            mqar.chmod(0o644)
            mqar.write_bytes(b"changedmqar")
            failures = _data_contract_failures(spec, manifest)
            codes = {item["code"] for item in failures}
            self.assertIn("DATA_ARTIFACT_HASH_MISMATCH", codes)
            self.assertNotIn("DATA_ARTIFACT_SIZE_MISMATCH", codes)

            mqar.write_bytes(b"changed-mqar-size")
            failures = _data_contract_failures(spec, manifest)
            self.assertIn(
                "DATA_ARTIFACT_SIZE_MISMATCH",
                {item["code"] for item in failures},
            )
            mqar.chmod(0o444)


if __name__ == "__main__":
    unittest.main()
