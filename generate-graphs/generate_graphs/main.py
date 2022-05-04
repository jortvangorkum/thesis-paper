from generate_graphs.plotter import Plotter
import seaborn as sns


if __name__ == "__main__":
    sns.set_palette("pastel")

    plotter = Plotter(
        images_path='../images',
        df_mem_path='./data/memory',
        df_time_path='./data/time',
        runs=['run-4', 'run-5']
    )

    plotter.plot_run_benchmarks('run-5')
    plotter.plot_comparison_runs(['run-4', 'run-5'])