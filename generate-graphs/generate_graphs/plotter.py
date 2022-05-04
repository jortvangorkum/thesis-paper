import os
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

        self.__create_images_directories()

    def __create_images_directories(self) -> None:
        print("Creating Images Directories...")
        for run in self.runs:
            os.makedirs(f'{self.images_path}/{run}/{self.images_mem_path}', exist_ok=True)
            os.makedirs(f'{self.images_path}/{run}/{self.images_time_path}', exist_ok=True)

    def __file_name_converter(self, name: str) -> str:
        return name.replace(' ', '_').lower()

    def __save_memory_benchmark(self, run: str, benchmark_name: str):
        benchmark_filename = self.__file_name_converter(benchmark_name)
        utils.save_benchmark(f'{self.images_path}/{run}/{self.images_mem_path}/{benchmark_filename}')

    def __save_time_benchmark(self, run: str, benchmark_name: str):
        benchmark_filename = self.__file_name_converter(benchmark_name)
        utils.save_benchmark(f'{self.images_path}/{run}/{self.images_time_path}/{benchmark_filename}')

    # Single Benchmark Plot
    def plot_memory_benchmark(self, df_mem_run: pd.DataFrame, benchmark_name: str) -> None:
        utils.plot_linear_benchmark(df_mem_run, benchmark_name, 'Amount Nodes', 'Memory Usage')
    
    def plot_time_benchmark(self, df_time_run: pd.DataFrame, benchmark_name: str) -> None:
        utils.plot_linear_benchmark(df_time_run, benchmark_name, 'Amount Nodes', 'Execution Time')

    # Multiple Benchmarks Plot
    def plot_memory_benchmarks(self, df_mem_run: pd.DataFrame, benchmark_names: List[str]) -> None:
        df_mem_benchmarks = df_mem_run[df_mem_run['Benchmark'].isin(benchmark_names)]
        utils.plot_all_benchmarks(df_mem_benchmarks, 'Amount Nodes', 'Memory Usage')

    def plot_time_benchmarks(self, df_time_run: pd.DataFrame, benchmark_names: List[str]) -> None:
        df_mem_benchmarks = df_time_run[df_time_run['Benchmark'].isin(benchmark_names)]
        utils.plot_all_benchmarks(df_mem_benchmarks, 'Amount Nodes', 'Execution Time')

    # Plot all benchmarks for a single run
    def plot_run_benchmarks(self, run: str) -> None:
        df_mem_run  = self.run_mem_dict[run]
        df_time_run = self.run_time_dict[run]

        print(f"Plotting memory benchmarks for {run}")
        for benchmark_name in df_mem_run['Benchmark']:
            self.plot_memory_benchmark(df_mem_run, benchmark_name)
            self.__save_memory_benchmark(run, benchmark_name)

        print(f"Plotting time benchmarks for {run}")
        for benchmark_name in df_time_run['Benchmark']:
            self.plot_time_benchmark(df_time_run, benchmark_name)
            self.__save_time_benchmark(run, benchmark_name)

        print(f'Plotting combined memory benchmarks for {run}')
        self.plot_memory_benchmarks(df_mem_run, list(df_mem_run['Benchmark']))
        self.__save_memory_benchmark(run, 'All Benchmarks')

        print(f'Plotting combined time benchmarks for {run}')
        self.plot_time_benchmarks(df_time_run, list(df_time_run['Benchmark']))
        self.__save_time_benchmark(run, 'All Benchmarks')

    def plot_comparison_runs(self, runs: List[str]) -> None:
        runs_images_path = '_'.join(runs)
        os.makedirs(f'{self.images_path}/{runs_images_path}', exist_ok=True)
        print(f'Plotting comparison benchmarks for runs {runs}')
        utils.plot_comparison_runs(self.run_mem_dict, self.run_time_dict, runs)
        utils.save_benchmark(f'{self.images_path}/{runs_images_path}/comparison_benchmark')