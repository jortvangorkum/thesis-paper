import os
import matplotlib.pyplot as plt
import seaborn as sns

from generate_graphs.utils import *
from generate_graphs.time import parse_time_data_file
from generate_graphs.memory import (parse_memory_data_file)

DATA_TIME_PATH = './data/time'
DATA_MEMORY_PATH = './data/memory'
IMAGES_PATH = '../images'
RUN = 'run-1'
RUN_IMAGES_PATH = f'plots/{RUN}'
TIME_IMAGES_PATH = f'{RUN_IMAGES_PATH}/time'
MEMORY_IMAGES_PATH = f'{RUN_IMAGES_PATH}/memory'

def save_benchmark(file_name: str) -> None:
    plt.savefig(f'{IMAGES_PATH}/{file_name}.pdf')
    plt.clf()

if __name__ == "__main__":
    sns.set_palette("pastel")

    if not os.path.exists(f'{IMAGES_PATH}/{RUN_IMAGES_PATH}'):
        os.makedirs(f'{IMAGES_PATH}/{TIME_IMAGES_PATH}')
        os.makedirs(f'{IMAGES_PATH}/{MEMORY_IMAGES_PATH}')

    df_time = parse_time_data_file(DATA_TIME_PATH, RUN)

    plot_linear_benchmark(df_time, 'Cata Sum', 'Amount Nodes', 'Execution Time')
    save_benchmark(f'{TIME_IMAGES_PATH}/benchmark_cata_sum')

    plot_linear_benchmark(df_time, 'Generic Cata Sum', 'Amount Nodes', 'Execution Time')
    save_benchmark(f'{TIME_IMAGES_PATH}/benchmark_generic_cata_sum')

    plot_log_benchmark(df_time, 'Incremental Compute Map', 'Amount Nodes', 'Execution Time')
    save_benchmark(f'{TIME_IMAGES_PATH}/benchmark_incremental_cata_sum')

    plot_all_benchmarks(df_time, "Amount Nodes", "Execution Time")
    save_benchmark(f'{TIME_IMAGES_PATH}/all_benchmarks')

    df_data_memory = parse_memory_data_file(DATA_MEMORY_PATH, RUN)

    plot_linear_benchmark(df_data_memory, 'Cata Sum Memory', 'Amount Nodes', 'Allocated Bytes')
    save_benchmark(f'{MEMORY_IMAGES_PATH}/benchmark_cata_sum')

    plot_linear_benchmark(df_data_memory, 'Generic Cata Sum', 'Amount Nodes', 'Allocated Bytes')
    save_benchmark(f'{MEMORY_IMAGES_PATH}/benchmark_generic_cata_sum')

    plot_log_benchmark(df_data_memory, 'Incremental Compute Map', 'Amount Nodes', 'Allocated Bytes')
    save_benchmark(f'{MEMORY_IMAGES_PATH}/benchmark_incremental_cata_sum')

    plot_all_benchmarks(df_data_memory, 'Amount Nodes', 'Allocated Bytes')
    save_benchmark(f'{MEMORY_IMAGES_PATH}/all_benchmarks')
