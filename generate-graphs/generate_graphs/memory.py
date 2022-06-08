import ast
from glob import glob
import os
import pathlib
import pandas as pd

def parse_folder(path):
    benchmark_type_paths = glob(f"{path}/**")
    benchmark_type_names = [pathlib.PurePath(folder).name for folder in benchmark_type_paths]
    return benchmark_type_paths,benchmark_type_names

def parse_memory_data_file(data_memory_path: str, folder_name: str) -> pd.DataFrame:
    df_data_memory = pd.DataFrame(columns=["Name", "Case", "Iterations", "Benchmark", "Amount Nodes", 'Bytes Allocated', 'Average Bytes Used', 'Max Bytes Used'])

    (benchmark_type_paths, benchmark_type_names) = parse_folder(f"{data_memory_path}/{folder_name}")

    for (benchmark_type_path, benchmark_type_name) in zip(benchmark_type_paths, benchmark_type_names):
        (benchmark_case_paths, benchmark_cases) = parse_folder(benchmark_type_path)

        for (benchmark_case_path, benchmark_case) in zip(benchmark_case_paths, benchmark_cases):
            (benchmark_iteration_paths, benchmark_iterations) = parse_folder(benchmark_case_path)

            for (benchmark_iteration_path, benchmark_iteration) in zip(benchmark_iteration_paths, benchmark_iterations):
                (benchmark_name_paths, benchmark_names) = parse_folder(benchmark_iteration_path)

                for (benchmark_name_path, benchmark_name) in zip(benchmark_name_paths, benchmark_names):
                    (benchmark_amount_nodes_paths, benchmark_amount_nodes) = parse_folder(benchmark_name_path)
                    benchmark_amount_nodes = list(map(lambda x: os.path.splitext(x)[0], benchmark_amount_nodes))

                    for (benchmark_amount_nodes_path, benchmark_amount_node) in zip(benchmark_amount_nodes_paths, benchmark_amount_nodes):
                        with open(benchmark_amount_nodes_path, 'r') as data_memory_file:
                            data_memory_contents = data_memory_file.read()

                            dict_memory_values = dict(ast.literal_eval(data_memory_contents))
                            average_bytes_used = dict_memory_values["average_bytes_used"]
                            max_bytes_used     = dict_memory_values["max_bytes_used"]
                            bytes_allocated    = dict_memory_values["bytes allocated"]

                            df_data_memory.loc[len(df_data_memory.index)] = [  # type: ignore
                                benchmark_type_name,
                                benchmark_case,
                                benchmark_iteration,
                                benchmark_name,
                                int(benchmark_amount_node),
                                int(bytes_allocated),
                                int(average_bytes_used),
                                int(max_bytes_used)
                            ] 

    df_data_memory.sort_values('Benchmark', inplace=True)

    print(df_data_memory)

    return df_data_memory

