import subprocess
from pathlib import Path
from typing import Sequence

from simrt.core.processor import PlatformInfo
from simrt.core.task import TaskInfo
from simrt.utils.schedulability_test import ExactTest


class GFPTest(ExactTest):
    def __init__(self, file_path) -> None:
        super().__init__()

    def test(self, Gamma: Sequence[TaskInfo], processors: PlatformInfo) -> bool:
        return super().test(Gamma, processors)


# Prepare input data
inputs = [
    "2",  # Number of processors
    "3",  # Number of tasks
    "1",  # Tasks have implicit deadlines
    "1",  # C[0]
    "3",  # P[0]
    "1",  # C[1]
    "4",  # P[1]
    "1",  # C[2]
    "2",  # P[2]
    "0",  # Dynamically optimize memory usage
    "0",  # Verbose
]

inputs_str = "\n".join(inputs) + "\n"  # Ensure there's a newline at the end

# Start the C++ program
process = subprocess.run(
    "./gfp_test_p1",
    # check=True,
    capture_output=True,
    input=inputs_str.encode(),  # Encode the string to bytes
)

# Print process details and output
print(f"Return code: {process.returncode}")
print("SCHED" if process.returncode == 1 else "UNSCHED")