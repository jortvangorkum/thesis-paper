import os
import matplotlib.pyplot as plt
import seaborn as sns

from generate_graphs.utils import *
from generate_graphs.time import parse_time_data_file
from generate_graphs.memory import (parse_memory_data_file)

DATA_TIME_PATH = './data/time'
DATA_MEMORY_PATH = './data/memory'
IMAGES_PATH = '../images'
RUN = 'run-5'
RUN_2 = 'run-4'
RUN_IMAGES_PATH = f'plots/{RUN}'
TIME_IMAGES_PATH = f'{RUN_IMAGES_PATH}/time'
MEMORY_IMAGES_PATH = f'{RUN_IMAGES_PATH}/memory'
COMPARE_RUNS_IMAGES_PATH = f'plots/{RUN}_{RUN_2}'

def save_benchmark(file_name: str) -> None:
    plt.savefig(f'{IMAGES_PATH}/{file_name}.pdf')
    plt.clf()

def plot_time(df_time: pd.DataFrame, time_images_path: str) -> None:
    if not os.path.exists(f'{IMAGES_PATH}/{RUN_IMAGES_PATH}'):
        os.makedirs(f'{IMAGES_PATH}/{time_images_path}')

    # plot_linear_benchmark(df_time, 'Cata Sum', 'Amount Nodes', 'Execution Time')
    # save_benchmark(f'{time_images_path}/benchmark_cata_sum')

    # plot_linear_benchmark(df_time, 'Generic Cata Sum', 'Amount Nodes', 'Execution Time')
    # save_benchmark(f'{time_images_path}/benchmark_generic_cata_sum')

    # plot_log_benchmark(df_time, 'Incremental Compute Map', 'Amount Nodes', 'Execution Time')
    # save_benchmark(f'{time_images_path}/benchmark_incremental_cata_sum')

    plot_all_benchmarks(df_time, "Amount Nodes", "Execution Time")
    save_benchmark(f'{time_images_path}/all_benchmarks')

def plot_memory(df_data_memory: pd.DataFrame, mem_images_path: str) -> None:
    if not os.path.exists(f'{IMAGES_PATH}/{RUN_IMAGES_PATH}/{mem_images_path}'):
        os.makedirs(f'{IMAGES_PATH}/{mem_images_path}')

    # plot_linear_benchmark(df_data_memory, 'Cata Sum', 'Amount Nodes', 'Memory Usage')
    # save_benchmark(f'{mem_images_path}/benchmark_cata_sum')

    # plot_linear_benchmark(df_data_memory, 'Generic Cata Sum', 'Amount Nodes', 'Memory Usage')
    # save_benchmark(f'{mem_images_path}/benchmark_generic_cata_sum')

    # plot_linear_benchmark(df_data_memory, 'Incremental Compute Map', 'Amount Nodes', 'Memory Usage')
    # save_benchmark(f'{mem_images_path}/benchmark_incremental_cata_sum')

    plot_all_benchmarks(df_data_memory, 'Amount Nodes', 'Memory Usage')
    save_benchmark(f'{mem_images_path}/all_benchmarks')

if __name__ == "__main__":
    sns.set_palette("pastel")

    df_time_1 = parse_time_data_file(DATA_TIME_PATH, RUN)
    df_data_memory_1 = parse_memory_data_file(DATA_MEMORY_PATH, RUN)

    (df_time_non_iter_1, df_time_iters_1) = split_benchmark_df_with_iterations(df_time_1)
    (df_mem_non_iter_1, df_mem_iters_1) = split_benchmark_df_with_iterations(df_data_memory_1)

    plot_time(df_time_iters_1, TIME_IMAGES_PATH)

    plot_memory(df_mem_iters_1, MEMORY_IMAGES_PATH)

    df_time_2 = parse_time_data_file(DATA_TIME_PATH, RUN_2)
    df_data_memory_2 = parse_memory_data_file(DATA_MEMORY_PATH, RUN_2)

    if not os.path.exists(f'{IMAGES_PATH}/{COMPARE_RUNS_IMAGES_PATH}'):
        os.makedirs(f'{IMAGES_PATH}/{COMPARE_RUNS_IMAGES_PATH}')

    plot_compare_runs(df_time_1, df_time_2, df_data_memory_1, df_data_memory_2)
    save_benchmark(f'{COMPARE_RUNS_IMAGES_PATH}/comparison_benchmarks')