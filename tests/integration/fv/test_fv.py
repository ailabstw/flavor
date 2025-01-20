import os
import subprocess
import tempfile
import unittest

import torch

from examples.fv.main import Net

# Global variable for a list of processes to be tested
test_proc = ["examples/fv/main.py"]


class TestFVProcessCompletion(unittest.TestCase):
    def test_process_completes_successfully(self):
        for proc in test_proc:
            with tempfile.TemporaryDirectory() as tempdir:
                input_path = os.path.join(tempdir, "input")
                output_path = os.path.join(tempdir, "output")
                weight_path = os.path.join(tempdir, "weight", "weight.ckpt")
                log_path = os.path.join(tempdir, "log")

                # Create necessary directories
                os.makedirs(os.path.dirname(weight_path), exist_ok=True)

                # Create a dummy weight file
                model = Net()
                state_dict = model.state_dict()
                torch.save({"state_dict": state_dict}, weight_path)

                # Set up environment variables
                os.environ["INPUT_PATH"] = input_path
                os.environ["OUTPUT_PATH"] = output_path
                os.environ["WEIGHT_PATH"] = weight_path
                os.environ["LOG_PATH"] = log_path

                # Main script
                main_script = f"poetry run python {proc}"

                # Command to run the check-fv script with the current process
                command = ["poetry", "run", "check-fv", "--main", main_script, "--yes"]

                # Run the command
                result = subprocess.run(
                    command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
                )

                if result.stderr:
                    print(f"Error occurred while running {proc}: {result.stderr}")

                # Check if the process has completed successfully
                self.assertEqual(
                    result.returncode, 0, f"Process {proc} did not complete successfully"
                )
