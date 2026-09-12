"""Read-only GPU preflight contracts using synthetic NVIDIA resource metadata."""

import os
import subprocess
import unittest
from unittest.mock import patch

from src.training.benchmark_track_a import GPUPreflightError, inspect_device


class GPUPreflightTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1"})
        self.environment.start()
        self.addCleanup(self.environment.stop)
        available = patch("torch.cuda.is_available", return_value=True)
        count = patch("torch.cuda.device_count", return_value=1)
        self.available = available.start()
        self.count = count.start()
        self.addCleanup(available.stop)
        self.addCleanup(count.stop)

    def query(self, occupancy="", memory=37, utilization=0, inventory=None):
        if inventory is None:
            inventory = f'1, GPU-selected, "NVIDIA Test, GPU", {memory}, 24000, {utilization}\n'
        return patch("subprocess.check_output", side_effect=[inventory, occupancy])

    def test_graphics_residency_is_available_without_special_memory_value(self):
        for memory in (7, 15, 37, 128):
            with self.subTest(memory=memory), self.query(memory=memory) as query:
                result = inspect_device("cuda:0", 1)
                self.assertEqual(result["preflight_status"], "AVAILABLE")
                self.assertEqual(result["preflight_memory_used_mib"], memory)
                self.assertEqual(result["preflight_memory_free_mib"], 24000)
                self.assertEqual(result["compute_process_count"], 0)
                for call in query.call_args_list:
                    self.assertIn("--id=1", call.args[0])
                    self.assertEqual(call.kwargs["timeout"], 10)

    def test_zero_compute_zero_or_low_utilization_is_available(self):
        for utilization in (0, 1, 3):
            with self.subTest(utilization=utilization), self.query(memory=37, utilization=utilization):
                self.assertEqual(inspect_device("cuda:0", 1)["preflight_status"], "AVAILABLE")

    def test_zero_memory_does_not_override_compute_occupancy(self):
        with self.query(occupancy="GPU-selected, 12345\n", memory=0):
            with self.assertRaises(GPUPreflightError) as caught:
                inspect_device("cuda:0", 1)
            self.assertEqual(caught.exception.status, "OCCUPIED")

    def test_other_user_compute_is_occupied_without_private_process_inspection(self):
        with self.query(occupancy="GPU-selected, 424242\n"), \
             patch("pathlib.Path.stat", side_effect=AssertionError("No process ownership read needed")), \
             patch("os.getuid", side_effect=AssertionError("No owner-based exemption allowed")):
            with self.assertRaises(GPUPreflightError) as caught:
                inspect_device("cuda:0", 1)
            self.assertEqual(caught.exception.status, "OCCUPIED")

    def test_current_user_job_and_current_process_have_no_implicit_exemption(self):
        for pid in (54321, os.getpid()):
            with self.subTest(pid=pid), self.query(occupancy=f"GPU-selected, {pid}\n"):
                with self.assertRaises(GPUPreflightError) as caught:
                    inspect_device("cuda:0", 1)
                self.assertEqual(caught.exception.status, "OCCUPIED")

    def test_failed_queries_are_unknown_not_hardware_failure(self):
        errors = [FileNotFoundError("nvidia-smi"), subprocess.TimeoutExpired("nvidia-smi", 10),
                  subprocess.CalledProcessError(9, "nvidia-smi", stderr="restricted context")]
        inventory = "1, GPU-selected, NVIDIA GPU, 37, 24000, 0\n"
        for error in errors:
            for side_effect in ([error], [inventory, error]):
                with self.subTest(error=type(error).__name__, query_count=len(side_effect)), \
                     patch("subprocess.check_output", side_effect=side_effect):
                    with self.assertRaises(GPUPreflightError) as caught:
                        inspect_device("cuda:0", 1)
                    self.assertEqual(caught.exception.status, "UNKNOWN")
                    self.assertIn("does not prove a broken GPU/driver", str(caught.exception))
                    self.assertIn("approved server execution context", str(caught.exception))

    def test_ambiguous_compute_output_is_unknown(self):
        for output in ("Not Supported\n", "GPU-selected, N/A\n", "GPU-selected, 0\n",
                       "GPU-other, 12345\n", "GPU-selected, 12345, extra\n", " \n"):
            with self.subTest(output=output), self.query(occupancy=output):
                with self.assertRaises(GPUPreflightError) as caught:
                    inspect_device("cuda:0", 1)
                self.assertEqual(caught.exception.status, "UNKNOWN")

    def test_invalid_or_missing_inventory_is_unknown(self):
        for inventory in ("", "0, GPU-selected, NVIDIA GPU, 37, 24000, 0\n",
                          "1, GPU-selected, NVIDIA GPU, N/A, 24000, 0\n",
                          "1, GPU-selected, NVIDIA GPU, nan, 24000, 0\n",
                          "1, GPU-selected, NVIDIA GPU, 37, 24000, 101\n"):
            with self.subTest(inventory=inventory), self.query(inventory=inventory):
                with self.assertRaises(GPUPreflightError) as caught:
                    inspect_device("cuda:0", 1)
                self.assertEqual(caught.exception.status, "UNKNOWN")

    def test_correct_host_and_process_local_mapping(self):
        with self.query():
            result = inspect_device("cuda:0", 1)
        self.assertEqual((result["host_gpu"], result["CUDA_VISIBLE_DEVICES"], result["device"]),
                         (1, "1", "cuda:0"))
        for mapping, device, host in (("1", "cuda:1", 1), ("0", "cuda:0", 1),
                                      ("0,1", "cuda:0", 1), ("1", "cuda:0", True)):
            with self.subTest(mapping=mapping, device=device, host=host), \
                 patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": mapping}), \
                 patch("subprocess.check_output") as query:
                with self.assertRaises(ValueError):
                    inspect_device(device, host)
                query.assert_not_called()

    def test_missing_cuda_visibility_stops(self):
        self.available.return_value = False
        with self.query():
            with self.assertRaises(GPUPreflightError) as caught:
                inspect_device("cuda:0", 1)
            self.assertEqual(caught.exception.status, "UNKNOWN")
        self.available.return_value = True
        self.count.return_value = 2
        with self.query():
            with self.assertRaises(GPUPreflightError) as caught:
                inspect_device("cuda:0", 1)
            self.assertEqual(caught.exception.status, "UNKNOWN")

    def test_cpu_path_has_no_nvidia_query(self):
        with patch("subprocess.check_output") as query:
            self.assertEqual(inspect_device("cpu")["device"], "cpu")
            query.assert_not_called()

    def test_cuda_runtime_query_exception_is_unknown(self):
        self.available.side_effect = RuntimeError("restricted runtime")
        with self.query():
            with self.assertRaises(GPUPreflightError) as caught:
                inspect_device("cuda:0", 1)
            self.assertEqual(caught.exception.status, "UNKNOWN")


if __name__ == "__main__":
    unittest.main()
