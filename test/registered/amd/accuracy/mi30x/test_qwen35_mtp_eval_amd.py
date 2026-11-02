"""AMD Qwen 3.5 FP8 MTP Test (8-GPU)

Tests Qwen/Qwen3.5-397B-A17B-FP8 with MTP speculative decoding (EAGLE, 3 steps,
4 draft tokens) on MI325/MI300X, and asserts the speculative accept length.

A broken draft model does not change the output: the target model verifies
every draft token, so GSM8K accuracy stays the same and only the accept length
drops (to 1.0 when every draft is rejected). The accuracy-only Qwen 3.5 test
cannot catch that, so this test checks the accept length directly.

Registry: nightly-amd-accuracy-8-gpu-qwen35 suite
"""

import os
import unittest
from types import SimpleNamespace

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.send_one import BenchArgs, send_one_prompt
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)

register_amd_ci(est_time=1800, suite="nightly-amd-accuracy-8-gpu-qwen35", nightly=True)

QWEN35_FP8_MODEL_PATH = os.environ.get(
    "QWEN35_FP8_MODEL_PATH", "Qwen/Qwen3.5-397B-A17B-FP8"
)
SERVER_LAUNCH_TIMEOUT = 3600
TP_SIZE = 8

# Healthy MTP accepts about 3 tokens per step on MI300X; a draft whose weights
# are missing or wrong accepts 1.0. Calibrate from nightly results.
MIN_ACCEPT_LENGTH = 2.5
MIN_GSM8K_ACCURACY = 0.93


class TestQwen35MTPEvalAMD(CustomTestCase):
    """Qwen 3.5 FP8 MTP accept length and GSM8K accuracy for AMD MI325/MI300X."""

    @classmethod
    def setUpClass(cls):
        cls.model = QWEN35_FP8_MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--tp",
            str(TP_SIZE),
            "--attention-backend",
            "aiter",
            "--mamba-ssm-dtype",
            "bfloat16",
            "--trust-remote-code",
            "--speculative-algorithm",
            "EAGLE",
            "--speculative-num-steps",
            "3",
            "--speculative-eagle-topk",
            "1",
            "--speculative-num-draft-tokens",
            "4",
            "--model-loader-extra-config",
            '{"enable_multithread_load": true}',
            "--watchdog-timeout",
            "1200",
        ]
        env = os.environ.copy()
        env["SGLANG_USE_AITER"] = "1"
        env["SGLANG_ENABLE_SPEC_V2"] = "1"
        # Real acceptance is the point of this test.
        env.pop("SGLANG_SIMULATE_ACC_LEN", None)
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=other_args,
            env=env,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_a_gsm8k(
        self,
    ):  # Append an "a" to make this test run first (alphabetically) to warm up the server
        requests.get(self.base_url + "/flush_cache")

        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=200,
            max_new_tokens=512,
            parallel=128,
            host="127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval_few_shot_gsm8k(args)
        print(f"{metrics=}")

        server_info = requests.get(self.base_url + "/server_info")
        avg_spec_accept_length = server_info.json()["internal_states"][0][
            "avg_spec_accept_length"
        ]
        print(f"{avg_spec_accept_length=}")

        if is_in_ci():
            write_github_step_summary(
                f"### test_gsm8k (qwen3.5-fp8 mtp)\n"
                f'{metrics["accuracy"]=:.3f}\n'
                f"{avg_spec_accept_length=:.2f}\n"
            )
        self.assertGreaterEqual(metrics["accuracy"], MIN_GSM8K_ACCURACY)
        self.assertGreater(avg_spec_accept_length, MIN_ACCEPT_LENGTH)

    def test_bs_1_accept_length(self):
        args = BenchArgs(port=int(self.base_url.split(":")[-1]), max_new_tokens=2048)
        acc_length, speed = send_one_prompt(args)

        print(f"{acc_length=:.2f} {speed=:.2f}")

        if is_in_ci():
            write_github_step_summary(
                f"### test_bs_1_accept_length (qwen3.5-fp8 mtp)\n"
                f"{acc_length=:.2f}\n"
                f"{speed=:.2f} token/s\n"
            )
        self.assertGreater(acc_length, MIN_ACCEPT_LENGTH)


if __name__ == "__main__":
    unittest.main()
