import os
import subprocess
import tempfile
import unittest

# Global variable for a list of processes to be tested
test_proc = ["examples/fl-client/pytorch/main.py"]


class TestFLProcessCompletion(unittest.TestCase):
    def test_process_completes_successfully(self):
        for proc in test_proc:
            with tempfile.TemporaryDirectory() as tempdir:
                input_path = os.path.join(tempdir, "input")
                output_path = os.path.join(tempdir, "output")
                log_path = os.path.join(tempdir, "log")

                # Set up environment variables
                local_model_path = os.path.join(tempdir, "local_model.pth")
                global_model_path = os.path.join(tempdir, "global_model.pth")

                os.environ["INPUT_PATH"] = input_path
                os.environ["OUTPUT_PATH"] = output_path
                os.environ["LOG_PATH"] = log_path
                os.environ["LOCAL_MODEL_PATH"] = local_model_path
                os.environ["GLOBAL_MODEL_PATH"] = global_model_path

                # Main script
                main_script = f"poetry run python {proc}"

                # Command to run the check-fl script with the current process
                command = [
                    "poetry",
                    "run",
                    "check-fl",
                    "--main",
                    main_script,
                    "--yes",
                    "--ignore-ckpt",
                ]

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
