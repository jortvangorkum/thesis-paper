import os
import pathlib
from typing import Dict, List
from generate_graphs.time import parse_time_data_file
from generate_graphs.memory import parse_memory_data_file
import generate_graphs.utils as utils
import pandas as pd

class Plotter:
    images_path: str
    images_mem_path: str  = 'memory'
    images_time_path: str = 'time'

    df_mem_path: str
    df_time_path: str

    runs: List[str]
    run_mem_dict: Dict[str, pd.DataFrame]  = {}
    run_time_dict: Dict[str, pd.DataFrame] = {}

    def __init__(
        self,
        images_path: str,
        df_mem_path: str,
        df_time_path: str,
        runs: List[str]
    ) -> None:
        self.images_path  = images_path
        self.df_mem_path  = df_mem_path
        self.df_time_path = df_time_path
        self.runs         = runs

        print("Parsing Data Files...")
        for run in runs:
            df_mem_run  = parse_memory_data_file(self.df_mem_path, run)
            df_time_run = parse_time_data_file(self.df_time_path, run)

            self.run_mem_dict[run]  = df_mem_run
            self.run_time_dict[run] = df_time_run

    def __file_name_converter(self, name: str) -> str:
        return name.replace(' ', '_').lower()

    def __save_memory_benchmark(self, run: str, benchmark_case: str, benchmark_iteration: str, benchmark_name: str):
        benchmark_filename = self.__file_name_converter(benchmark_name)
        file_path = f"{self.images_path}/{run}/{self.images_mem_path}/{benchmark_case}/{benchmark_iteration}/{benchmark_filename}"
        folder_path = pathlib.PurePath(file_path).parent
        os.makedirs(folder_path, exist_ok=True)
        utils.save_benchmark(file_path)

    def __save_time_benchmark(self, run: str, benchmark_case: str, benchmark_iteration: str, benchmark_name: str):
        benchmark_filename = self.__file_name_converter(benchmark_name)
        file_path = f"{self.images_path}/{run}/{self.images_time_path}/{benchmark_case}/{benchmark_iteration}/{benchmark_filename}"
        folder_path = pathlib.PurePath(file_path).parent
        os.makedirs(folder_path, exist_ok=True)
        utils.save_benchmark(file_path)

    # Single Benchmark Plot
    def plot_linear_memory_benchmark(self, df_mem_run: pd.DataFrame, benchmark_case: str, benchmark_iteration: str, benchmark_name: str) -> None:
        utils.plot_linear_benchmark(df_mem_run, benchmark_case, benchmark_iteration, benchmark_name, 'Amount Nodes', 'Max Bytes Used')
    
    def plot_linear_time_benchmark(self, df_time_run: pd.DataFrame, benchmark_case: str, benchmark_iteration: str, benchmark_name: str) -> None:
        utils.plot_linear_benchmark(df_time_run, benchmark_case, benchmark_iteration, benchmark_name, 'Amount Nodes', 'Execution Time')

    def plot_log_memory_benchmark(self, df_mem_run: pd.DataFrame, benchmark_case: str, benchmark_iteration: str, benchmark_name: str) -> None:
        utils.plot_log_benchmark(df_mem_run, benchmark_case, benchmark_iteration, benchmark_name, 'Amount Nodes', 'Max Bytes Used')

    def plot_log_time_benchmark(self, df_time_run: pd.DataFrame, benchmark_case: str, benchmark_iteration: str, benchmark_name: str) -> None:
        utils.plot_log_benchmark(df_time_run, benchmark_case, benchmark_iteration, benchmark_name, 'Amount Nodes', 'Execution Time')

    # Multiple Benchmarks Plot
    def plot_memory_benchmarks(self, df_mem_run: pd.DataFrame, benchmark_case: str, benchmark_iteration: str,  benchmark_names: List[str]) -> None:
        df_mem_benchmarks = df_mem_run[df_mem_run['Benchmark'].isin(benchmark_names)]
        utils.plot_all_benchmarks(df_mem_benchmarks, benchmark_case, benchmark_iteration, 'Amount Nodes', 'Max Bytes Used')

    def plot_time_benchmarks(self, df_time_run: pd.DataFrame, benchmark_case: str, benchmark_iteration: str,  benchmark_names: List[str]) -> None:
        df_mem_benchmarks = df_time_run[df_time_run['Benchmark'].isin(benchmark_names)]
        utils.plot_all_benchmarks(df_mem_benchmarks, benchmark_case, benchmark_iteration, 'Amount Nodes', 'Execution Time')

    # Plot all benchmarks for a single run
    def plot_run_benchmarks(self, run: str) -> None:
        df_mem_run  = self.run_mem_dict[run]
        df_time_run = self.run_time_dict[run]

        for benchmark_case in df_mem_run['Case'].unique():
            print(f"Plotting memory benchmarks for case {benchmark_case}")
            for benchmark_iteration in df_mem_run['Iterations'].unique():
                print(f"Plotting memory benchmarks for iteration {benchmark_iteration}")
                df_benchmark_data: pd.DataFrame = df_mem_run[(df_mem_run['Case'] == benchmark_case) & (df_mem_run['Iterations'] == benchmark_iteration)] # type: ignore

                print(f"Plotting memory benchmarks for {run}")
                for benchmark_name in df_benchmark_data['Benchmark'].unique():
                    self.plot_linear_memory_benchmark(df_benchmark_data, benchmark_case, benchmark_iteration, benchmark_name)
                    self.__save_memory_benchmark(run, benchmark_case, benchmark_iteration, f"linear/{benchmark_name}")
                    self.plot_log_memory_benchmark(df_benchmark_data, benchmark_case, benchmark_iteration, benchmark_name)
                    self.__save_memory_benchmark(run, benchmark_case, benchmark_iteration, f"log/{benchmark_name}")

                print(f'Plotting combined memory benchmarks for {run}')
                self.plot_memory_benchmarks(df_benchmark_data, benchmark_case, benchmark_iteration, list(df_benchmark_data['Benchmark']))
                self.__save_memory_benchmark(run, benchmark_case, benchmark_iteration, 'All Benchmarks')

        for benchmark_case in df_time_run['Case'].unique():
            print(f"Plotting time benchmarks for case {benchmark_case}")
            for benchmark_iteration in df_time_run['Iterations'].unique():
                print(f"Plotting time benchmarks for iteration {benchmark_iteration}")
                df_benchmark_data: pd.DataFrame = df_time_run[(df_time_run['Case'] == benchmark_case) & (df_time_run['Iterations'] == benchmark_iteration)] # type: ignore

                print(f"Plotting time benchmarks for {run}")
                for benchmark_name in df_benchmark_data['Benchmark'].unique():
                    self.plot_linear_time_benchmark(df_benchmark_data, benchmark_case, benchmark_iteration, benchmark_name)
                    self.__save_time_benchmark(run, benchmark_case, benchmark_iteration, f"linear/{benchmark_name}")
                    self.plot_log_time_benchmark(df_benchmark_data, benchmark_case, benchmark_iteration, benchmark_name)
                    self.__save_time_benchmark(run, benchmark_case, benchmark_iteration, f"log/{benchmark_name}")

                print(f'Plotting combined time benchmarks for {run}')
                self.plot_time_benchmarks(df_benchmark_data, benchmark_case, benchmark_iteration, list(df_benchmark_data['Benchmark']))
                self.__save_time_benchmark(run, benchmark_case, benchmark_iteration, 'All Benchmarks')

    def plot_comparison_runs(self, runs: List[str]) -> None:
        runs_images_path = '_'.join(runs)
        os.makedirs(f'{self.images_path}/{runs_images_path}', exist_ok=True)
        print(f'Plotting comparison benchmarks for runs {runs}')
        utils.plot_comparison_runs(self.run_mem_dict, self.run_time_dict, runs)
        utils.save_benchmark(f'{self.images_path}/{runs_images_path}/comparison_benchmark')