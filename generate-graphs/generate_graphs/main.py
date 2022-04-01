import pandas as pd
import re

regex = r"benchmarking (.*?)\/(\d*)(?:(?:.|\n)*?)iters(?:(?:\s)*)(\d+\.?\d+e?\d*)"

DATA_MEMORY_PATH = './data/memory'

def parse_memory_data_file(file_name: str):
    with open(f"{DATA_MEMORY_PATH}/{file_name}.txt", 'r') as data_memory_file:
        data_memory_contents = data_memory_file.read()
        matches = re.finditer(regex, data_memory_contents)
        df_data_memory = pd.DataFrame(columns=['Benchmark', 'Amount Nodes', 'Allocated Bytes'])

        for match in matches:
            benchmark_name  = match.group(1)
            amount_nodes    = match.group(2)
            allocated_bytes = match.group(3)
            print(f"benchmark {benchmark_name}/{amount_nodes}\n\tallocated: {allocated_bytes} bytes\n")

            df_data_memory.loc[len(df_data_memory.index)] = [benchmark_name, int(amount_nodes), float(allocated_bytes)]

        print(df_data_memory)
        print(df_data_memory.dtypes)

if __name__ == "__main__":
    parse_memory_data_file('run-0')