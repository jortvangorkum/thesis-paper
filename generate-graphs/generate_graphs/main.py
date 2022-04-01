from typing import List
import numpy as np
import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from scipy import stats
import statsmodels.api as sm

regex_memory_benchmark = r"benchmarking (.*?)\/(\d*)(?:(?:.|\n)*?)iters(?:(?:\s)*)(\d+\.?\d+e?\d*)"

DATA_MEMORY_PATH = './data/memory'

def parse_memory_data_file(file_name: str) -> pd.DataFrame:
    with open(f"{DATA_MEMORY_PATH}/{file_name}.txt", 'r') as data_memory_file:
        data_memory_contents = data_memory_file.read()
        matches = re.finditer(regex_memory_benchmark, data_memory_contents)
        df_data_memory = pd.DataFrame(columns=['Benchmark', 'Amount Nodes', 'Allocated Bytes'])

        for match in matches:
            benchmark_name  = match.group(1)
            amount_nodes    = match.group(2)
            allocated_bytes = match.group(3)

            if (benchmark_name is None or amount_nodes is None or allocated_bytes is None):
                raise Exception("Cannot correctly convert data memory benchmark to values")

            df_data_memory.loc[len(df_data_memory.index)] = [benchmark_name, int(amount_nodes), float(allocated_bytes)]

        return df_data_memory

def plot_linear_benchmark(df: pd.DataFrame, benchmark_name: str, logx = True, logy = True) -> None:
    data = df[df['Benchmark'] == benchmark_name]
    x = data["Amount Nodes"]
    y = data["Allocated Bytes"]

    model = sm.OLS(y, x)
    r = model.fit()
    slope = r.params[0]
    rsquared = r.rsquared 
    
    ax1 = sns.lineplot(x=x, y=slope * x)
    ax2 = sns.scatterplot(x=x, y=y)

    if logx:
        plt.xscale('log')
    if logy:
        plt.yscale('log')
    
    ax1.legend(title="Equation", loc='upper left', labels=["y={0:.1f}x, R²={1:.2f}".format(slope, rsquared)])


if __name__ == "__main__":
    sns.set_palette("pastel")

    df_data_memory_1 = parse_memory_data_file('run-1')
    # print(df_data_memory_1)

    plot_linear_benchmark(df_data_memory_1, 'Cata Sum Memory')
    plt.show()
    plt.clf()

    plot_linear_benchmark(df_data_memory_1, 'Generic Cata Sum')
    plt.show()
    plt.clf()

    plot_linear_benchmark(df_data_memory_1, 'Incremental Compute Map')
    plt.show()
    plt.clf()