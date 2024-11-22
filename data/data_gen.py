import random
import subprocess
import time
from math import ceil
from pathlib import Path
from typing import Sequence

from simrt.core.processor import PlatformInfo
from simrt.core.task import PeriodicTask, TaskInfo
from simrt.generator.task_factory import UtilizationGenerationAlgorithm
from simrt.generator.taskset_generator import (
    TasksetFactory,
    TasksetGenerator,
    TaskSubsetFactory,
)
from simrt.utils.schedulability_analyzer import SchedulabilityAnalyzer
from simrt.utils.schedulability_test import ExactTest, TestFactory
from simrt.utils.schedulability_test_executor import (
    ParallelStrategy,
    SchedulabilityTestExecutor,
    SerialStrategy,
    SqlitePersistence,
)
from simrt.utils.task_storage import TaskStorage
from tqdm import tqdm, trange


class GFPTest(ExactTest):

    def __init__(self, file_path: Path) -> None:
        self.file_path = file_path

    def test(self, Gamma: Sequence[TaskInfo], processors: PlatformInfo) -> bool:
        if not processors.is_homogeneous:
            raise ValueError("processors is not homogeneous.")

        num_processors = len(processors.speed_list)
        num_task = len(Gamma)
        # Prepare input data
        inputs = [
            str(num_processors),  # Number of processors
            str(num_task),  # Number of tasks
            "1",  # Tasks have implicit deadlines
            # "1",  # C[0]
            # "3",  # P[0]
            # "1",  # C[1]
            # "4",  # P[1]
            # "1",  # C[2]
            # "2",  # P[2]
            # "0",  # Dynamically optimize memory usage
            # "0",  # Verbose
        ]

        for taskinfo in Gamma:
            inputs.append(str(taskinfo.wcet))
            inputs.append(str(taskinfo.period))

        inputs.append("0")  # Dynamically optimize memory usage
        inputs.append("0")  # Verbose

        inputs_str = "\n".join(inputs) + "\n"  # Ensure there's a newline at the end

        # Start the C++ program
        process = subprocess.run(
            self.file_path.as_posix(),
            # check=True,
            capture_output=True,
            input=inputs_str.encode(),  # Encode the string to bytes
        )
        return process.returncode == 1


if __name__ == "__main__":
    # 可调度性分析器
    analyzer = SchedulabilityAnalyzer()
    # analyzer.set_exact_test(
    #     GFPTest(Path("/home/polyarc/Development/schedulability-prediction/gfp_test_p1"))
    # )
    # analyzer.set_sufficient_test(TestFactory.create_test("GlobalFPTest"))
    analyzer.set_exact_test(TestFactory.create_test("SimulationTest", cutoff=1000000))
    analyzer.set_sufficient_test(TestFactory.create_test("GlobalEDFTest"))

    # 执行策略
    execution = ParallelStrategy(num_process=20, chunksize=1, show_progress=True)
    # execution = SerialStrategy()

    # 数据持久化策略
    current_time = time.localtime()
    formatted_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    persistence = SqlitePersistence(data_path=Path(f"./data/") / formatted_time)

    # 可调度性测试执行器
    executor = SchedulabilityTestExecutor(
        test_analyzer=analyzer,
        execution_strategy=execution,
        persistence_strategy=persistence,
    )

    # 数据集生成器
    platform = PlatformInfo([4, 3, 2, 1])
    period_bound = (10, 40)
    num_task = 2000

    task_db = TaskStorage(Path(f"./data/") / formatted_time / "data.sqlite")
    task_db.insert_metadata(platform.speed_list, period_bound, num_task)
    task_db.commit()
    task_db.close()

    generator = (
        TasksetGenerator()
        .set_task_type(PeriodicTask)
        # .set_utilization_algorithm(UtilizationGenerationAlgorithm.UScaling)
        .set_taskset_factory(TaskSubsetFactory)
        .set_platform_info(platform)
        .set_period_bound(period_bound)
        .set_num_task(num_task)
        .setup()
    )

    # 生成数据集
    # totol_num_taskset = 20000

    tasksets = []
    for taskset_size in range(5, 9):
        for i in range(20000):
            taskset = generator.generate_taskset(num_task=taskset_size)
            tasksets.append(taskset)
    random.shuffle(tasksets)
    print("任务集生成完毕")

    # 并行测试大量数据集的可调度性
    executor.execute(tasksets, platform)
