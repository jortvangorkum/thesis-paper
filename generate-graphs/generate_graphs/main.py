from generate_graphs.utils import FuncType
import matplotlib.pyplot as plt
import seaborn as sns

from generate_graphs.utils import *
from generate_graphs.time import parse_time_data_file
from generate_graphs.memory import (parse_memory_data_file)

DATA_TIME_PATH = './data/time'
DATA_MEMORY_PATH = './data/memory'
IMAGES_PATH = '../images'

def save_benchmark(file_name: str) -> None:
    plt.savefig(f'{IMAGES_PATH}/{file_name}.pdf')
    plt.clf()

if __name__ == "__main__":
    sns.set_palette("pastel")

    df_time = parse_time_data_file(DATA_TIME_PATH, 'run-0')

    plot_linear_benchmark(df_time, 'Cata Sum', 'Amount Nodes', 'Execution Time')
    save_benchmark('plots/time/benchmark_cata_sum')

    plot_linear_benchmark(df_time, 'Generic Cata Sum', 'Amount Nodes', 'Execution Time')
    save_benchmark('plots/time/benchmark_generic_cata_sum')

    plot_log_benchmark(df_time, 'Incremental Compute Map', 'Amount Nodes', 'Execution Time')
    save_benchmark('plots/time/benchmark_incremental_cata_sum')

    plot_all_benchmarks(df_time, "Amount Nodes", "Execution Time")
    save_benchmark('plots/time/all_benchmarks')

    df_data_memory = parse_memory_data_file(DATA_MEMORY_PATH, 'run-1')

    plot_linear_benchmark(df_data_memory, 'Cata Sum Memory', 'Amount Nodes', 'Allocated Bytes')
    save_benchmark('plots/memory/benchmark_cata_sum')

    plot_linear_benchmark(df_data_memory, 'Generic Cata Sum', 'Amount Nodes', 'Allocated Bytes')
    save_benchmark('plots/memory/benchmark_generic_cata_sum')

    plot_log_benchmark(df_data_memory, 'Incremental Compute Map', 'Amount Nodes', 'Allocated Bytes')
    save_benchmark('plots/memory/benchmark_incremental_cata_sum')

    plot_all_benchmarks(df_data_memory, 'Amount Nodes', 'Allocated Bytes')
    save_benchmark('plots/memory/all_benchmarks')
