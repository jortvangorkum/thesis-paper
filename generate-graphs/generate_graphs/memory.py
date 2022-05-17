import ast
from glob import glob
import os
import pathlib
import pandas as pd

def parse_memory_data_file(data_memory_path: str, folder_name: str) -> pd.DataFrame:
    df_data_memory = pd.DataFrame(columns=['Benchmark Type', 'Benchmark', 'Amount Nodes', 'Bytes Allocated', 'Average Bytes Used', 'Max Bytes Used'])

    benchmark_type_paths = glob(f"{data_memory_path}/{folder_name}/**")
    benchmark_type_names = [pathlib.PurePath(folder).name for folder in benchmark_type_paths]

    for (benchmark_type_path, benchmark_type_name) in zip(benchmark_type_paths, benchmark_type_names):
        benchmark_paths = glob(f"{benchmark_type_path}/**")
        benchmark_names = [pathlib.PurePath(folder).name for folder in benchmark_paths]

        for (benchmark_path, benchmark_name) in zip(benchmark_paths, benchmark_names):
            amount_nodes_file_paths = glob(f"{benchmark_path}/**")
            list_amount_nodes = [os.path.splitext(pathlib.PurePath(file).name)[0] for file in amount_nodes_file_paths]

            for (amount_nodes_file_path, amount_nodes) in zip(amount_nodes_file_paths, list_amount_nodes):
                with open(amount_nodes_file_path, 'r') as data_memory_file:
                    data_memory_contents = data_memory_file.read()

                    dict_memory_values = dict(ast.literal_eval(data_memory_contents))
                    average_bytes_used = dict_memory_values["average_bytes_used"]
                    max_bytes_used     = dict_memory_values["max_bytes_used"]
                    bytes_allocated    = dict_memory_values["bytes allocated"]

                    df_data_memory.loc[len(df_data_memory.index)] = [benchmark_type_name, benchmark_name, int(amount_nodes), int(bytes_allocated), int(average_bytes_used), int(max_bytes_used)]  # type: ignore

    df_data_memory_single_iteration = df_data_memory[df_data_memory['Benchmark Type'] == 'Single Iteration']
    df_data_memory_single_iteration = df_data_memory_single_iteration.drop('Benchmark Type', axis=1)
    print(df_data_memory_single_iteration)

    return df_data_memory_single_iteration
