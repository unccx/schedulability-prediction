import bisect
import random
import subprocess
import time
from math import ceil
from pathlib import Path
from typing import Sequence

import numpy as np
from simrt.core.processor import PlatformInfo
from simrt.core.task import PeriodicTask, TaskInfo
from simrt.generator.task_factory import UtilizationGenerationAlgorithm
from simrt.generator.taskset_generator import (
    Taskset,
    TasksetFactory,
    TasksetGenerator,
    TaskSubsetFactory,
)
from simrt.utils.schedulability_analyzer import SchedulabilityAnalyzer
from simrt.utils.schedulability_test import ExactTest, TestFactory
from simrt.utils.schedulability_test_executor import (
    ParallelStrategy,
    PersistenceStrategy,
    SchedulabilityTestExecutor,
    SerialStrategy,
    SqlitePersistence,
)
from simrt.utils.task_storage import TaskStorage
from tqdm import tqdm, trange


class NonePersistenceStrategy(PersistenceStrategy):

    def __init__(
        self,
    ) -> None:
        pass

    def connect(self) -> None:
        pass

    def save_task(self, task: TaskInfo) -> None:
        pass

    def save_taskset(self, taskset: Taskset, **kwargs) -> None:
        pass

    def close(self):
        pass


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
    #     TestFactory.create_test("SimulationTest", show_progress=True)
    # )
    # analyzer.set_sufficient_test(TestFactory.create_test("GlobalEDFTest"))
    # analyzer.set_exact_test(
    #     GFPTest(Path("/home/polyarc/Development/schedulability-prediction/gfp_test_p1"))
    # )
    # analyzer.set_sufficient_test(TestFactory.create_test("GlobalFPTest"))
    # analyzer.set_exact_test(TestFactory.create_test("SimulationTest"))
    # analyzer.set_sufficient_test(TestFactory.create_test("GlobalEDFTest"))

    # 执行策略
    # execution = ParallelStrategy(num_process=20, chunksize=1, show_progress=True)
    execution = SerialStrategy(show_progress=True)

    # 数据持久化策略
    persistence = NonePersistenceStrategy()

    # 可调度性测试执行器
    executor = SchedulabilityTestExecutor(
        test_analyzer=analyzer,
        execution_strategy=execution,
        persistence_strategy=persistence,
    )

    # 数据集生成器
    platform = PlatformInfo([1, 1])
    period_bound = (10, 40)
    num_task = 2000
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

    # tasksets: list[Taskset] = []
    # for i in range(100):
    #     taskset = generator.generate_taskset(num_task=9)
    #     tasksets.append(taskset)
    # tasksets.sort(key=lambda taskset: sum(task.utilization for task in taskset))

    # running_time = []
    # for l, r in zip(range(0, 10), range(1, 11)):
    #     l, r = l * 0.1, r * 0.1
    #     left_idx = bisect.bisect_left(
    #         a=tasksets,
    #         x=l,
    #         key=lambda taskset: sum(task.utilization for task in taskset),
    #     )
    #     right_idx = bisect.bisect_right(
    #         a=tasksets,
    #         x=r,
    #         key=lambda taskset: sum(task.utilization for task in taskset),
    #     )

    #     start_time = time.time()
    #     executor.execute(tasksets[left_idx:right_idx], platform)
    #     end_time = time.time()
    #     running_time.append(end_time - start_time)

    # print(f"running_time: {running_time}")

    # # 测试不同任务基数的任务集的平均超周期
    # taskset_num = 10000
    # hyper_periods = []
    # for num_task in range(5, 15):
    #     hyper_period_average = 0
    #     for i in range(taskset_num):
    #         taskset = generator.generate_taskset(num_task=num_task)
    #         hyper_period = np.lcm.reduce([task.period for task in taskset])
    #         hyper_period_average += hyper_period.item() / taskset_num
    #     hyper_periods.append(hyper_period_average)

    # print(f"hyper_periods: {hyper_periods}")

    # 测试不同任务基数的任务集的平均超周期
    taskset_num = 10000
    hyper_periods = []
    representative_tasksets = []
    for num_task in range(5, 9):
        tasksets = []
        hyper_period_average = 0
        for i in range(taskset_num):
            taskset = generator.generate_taskset(num_task=num_task)
            tasksets.append(taskset)
            hyper_period = np.lcm.reduce([task.period for task in taskset]).item()
            hyper_period_average += hyper_period / taskset_num
        hyper_periods.append(hyper_period_average)

        min_diff = float("inf")
        for taskset in tasksets:
            hyper_period = np.lcm.reduce([task.period for task in taskset]).item()
            if abs(hyper_period_average - hyper_period) < min_diff:
                min_diff = abs(hyper_period_average - hyper_period)
                representative = taskset
                hypa = hyper_period_average
        representative_tasksets.append(representative)

        [
            np.lcm.reduce([task.period for task in representative]).item()
            for representative in representative_tasksets
        ]

    print(f"hyper_period_averages: {hyper_periods}")
    print(
        f"actual_hyper_periods: {[
            np.lcm.reduce([task.period for task in representative]).item()
            for representative in representative_tasksets
        ]}"
    )

    # test = TestFactory.create_test("SimulationTest", show_progress=True)
    # test = GFPTest(
    #     Path("/home/polyarc/Development/schedulability-prediction/gfp_test_p1")
    # )
    test = TestFactory.create_test("GlobalFPTest")
    # test = TestFactory.create_test("GlobalEDFTest")
    running_time = []
    test_result = []
    for taskset in representative_tasksets:
        start_time = time.time()
        res = test.test(taskset, platform)
        test_result.append(res)
        end_time = time.time()
        running_time.append(end_time - start_time)

    print(f"running_time: {running_time}")
    print(f"test_result: {test_result}")
