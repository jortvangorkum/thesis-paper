from enum import Enum, auto
from typing import List, Tuple
import numpy as np
import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from scipy.optimize import curve_fit
import statsmodels.api as sm

regex_memory_benchmark = r"benchmarking (.*?)\/(\d*)(?:(?:.|\n)*?)iters(?:(?:\s)*)(\d+\.?\d+e?\d*)"

DATA_MEMORY_PATH = './data/memory'
IMAGES_PATH = '../images'

class FuncType(Enum):
    Linear = auto()
    Log = auto()

def calculate_rsquared(x: pd.Series, y: pd.Series, f, popt) -> float:
    ss_res   = np.dot((y - f(x, *popt)), (y - f(x, *popt)))
    ymean    = np.mean(y)
    ss_tot   = np.dot((y-ymean), (y-ymean))
    rsquared = 1 - ss_res/ss_tot

    return rsquared

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

def plot_linear_line(x: pd.Series, y: pd.Series) -> None:
    model = sm.OLS(y, x)
    r = model.fit()
    slope = r.params[0]
    rsquared = r.rsquared 

    ax1 = sns.lineplot(x=x, y=slope * x)
    
    ax1.legend(title="Equation", loc='upper left', labels=["y={0:.1f}x, R²={1:.2f}".format(slope, rsquared)])

def plot_linear_benchmark(df: pd.DataFrame, benchmark_name: str) -> None:
    data = df[df['Benchmark'] == benchmark_name]
    x = data["Amount Nodes"]
    y = data["Allocated Bytes"]

    plot_linear_line(x, y)
    ax2 = sns.scatterplot(x=x, y=y)

    plt.xscale('log')
    plt.yscale('log')

def plot_log_line(x: pd.Series, y: pd.Series) -> None:
    def log_func(x, a, b): 
        return a + b * np.log(x)

    (popt, _) = curve_fit(log_func, x, y)
    rsquared = calculate_rsquared(x, y, log_func, popt)

    ax1 = sns.lineplot(x=x, y=x.apply(lambda i: log_func(i, *popt)))
    
    ax1.legend(title="Equation", loc='upper left', labels=["y={0:.1f} * ln(x) + {1:.1f}, R²={2:.2f}".format(*popt, rsquared)])

def plot_log_benchmark(df: pd.DataFrame, benchmark_name: str) -> None:
    data = df[df['Benchmark'] == benchmark_name]
    x = data["Amount Nodes"]
    y = data["Allocated Bytes"]

    plot_log_line(x, y)
    ax2 = sns.scatterplot(x=x, y=y)

    plt.xscale('log')
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0,3), useMathText=True)

def plot_all_benchmarks(df: pd.DataFrame, benchmarks: List[Tuple[str, FuncType]]) -> None:
    benchmark_names = []
    for (benchmark_name, func_type) in benchmarks:
        data = df[df['Benchmark'] == benchmark_name]
        x = data["Amount Nodes"]
        y = data["Allocated Bytes"]

        match func_type:
            case FuncType.Linear:
                plot_linear_line(x, y)
            case FuncType.Log:
                plot_log_line(x,y)

        ax = sns.scatterplot(x=x, y=y, label="_nolegend_")
        benchmark_names.append(benchmark_name)

    ax.legend(title="Benchmarks", loc="upper left", labels=benchmark_names)

    plt.xscale('log')
    plt.yscale('log')

def save_benchmark(file_name: str) -> None:
    plt.savefig(f'{IMAGES_PATH}/{file_name}.pdf')
    plt.clf()

if __name__ == "__main__":
    sns.set_palette("pastel")

    df_data_memory_1 = parse_memory_data_file('run-1')

    plot_linear_benchmark(df_data_memory_1, 'Cata Sum Memory')
    save_benchmark('plots/memory/benchmark_cata_sum')

    plot_linear_benchmark(df_data_memory_1, 'Generic Cata Sum')
    save_benchmark('plots/memory/benchmark_generic_cata_sum')

    plot_log_benchmark(df_data_memory_1, 'Incremental Compute Map')
    save_benchmark('plots/memory/benchmark_incremental_cata_sum')

    plot_all_benchmarks(df_data_memory_1, [('Cata Sum Memory', FuncType.Linear), ('Generic Cata Sum', FuncType.Linear), ('Incremental Compute Map', FuncType.Log)])
    save_benchmark('plots/memory/all_benchmarks')