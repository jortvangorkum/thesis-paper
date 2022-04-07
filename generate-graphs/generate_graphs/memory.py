import pandas as pd
import re

regex_memory_benchmark = r"benchmarking (.*?)\/(\d*)(?:(?:.|\n)*?)iters(?:(?:\s)*)(\d+\.?\d+e?\d*)"

def parse_memory_data_file(data_memory_path: str, file_name: str) -> pd.DataFrame:
    with open(f"{data_memory_path}/{file_name}.txt", 'r') as data_memory_file:
        data_memory_contents = data_memory_file.read()
        matches = re.finditer(regex_memory_benchmark, data_memory_contents)
        df_data_memory = pd.DataFrame(columns=['Benchmark', 'Amount Nodes', 'Memory Usage'])

        for match in matches:
            benchmark_name  = match.group(1)
            amount_nodes    = match.group(2)
            allocated_bytes = match.group(3)

            if (benchmark_name is None or amount_nodes is None or allocated_bytes is None):
                raise Exception("Cannot correctly convert data memory benchmark to values")

            df_data_memory.loc[len(df_data_memory.index)] = [benchmark_name, int(amount_nodes), float(allocated_bytes)]  # type: ignore

        return df_data_memory
