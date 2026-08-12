from concurrent.futures import ProcessPoolExecutor
import os
import matplotlib.pyplot as plt
import numpy as np
from percolation import HoshenKopelman, generate_grid
import time

def run_single_simulation(args: tuple[int, float]) -> bool: 
    L, p = args
    grid = generate_grid(L,p)
    hk = HoshenKopelman(grid)
    hk.run()
    has_perc, _ = hk.check_percolation()
    return has_perc


def mc_estimate_p_parallel(L: int, p: float, num_trials: int, executor: ProcessPoolExecutor) -> float:
    task_args = [(L, p) for _ in range(num_trials)]
    results = list(executor.map(run_single_simulation, task_args))
    return sum(results) / num_trials

def extrapolate_pc(L_values: list[int], pc_L_values: list[float]) -> float:
    inv_L = [1.0 / L for L in L_values]
    k, b = np.polyfit(inv_L, pc_L_values, 1)  
    return b





def main():
    L_values = [20, 40, 60, 80, 100]
    p_values = np.linspace(0.40, 0.80, 41)
    num_trials = 1000

    FORCE_RECALCULATE = False

    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(data_dir, exist_ok=True)

    plt.figure(figsize=(9, 6))
    pc_L_list = []

    count_cores = os.cpu_count() or 4

    with ProcessPoolExecutor(max_workers=count_cores) as executor:

        start_time = time.perf_counter()

        for L in L_values:
            csv_filename = f"percolation_L{L}_N{num_trials}.csv"
            csv_path = os.path.join(data_dir, csv_filename)

            if os.path.exists(csv_path) and not FORCE_RECALCULATE:
                print(f"Загрузка готовых данных из файла: {csv_filename}")
                data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
                curr_p = data[:, 0]
                curr_P = data[:, 1]
            else:
                print(
                    f"Запуск новой симуляции: L = {L}, N = {num_trials} (все ядра)..."
                )
                curr_p = p_values
                curr_P = []

                for p in curr_p:
                    prob = mc_estimate_p_parallel(L, p, num_trials, executor)
                    curr_P.append(prob)

                curr_P = np.array(curr_P)

                data_to_save = np.column_stack((curr_p, curr_P))
                np.savetxt(
                    csv_path,
                    data_to_save,
                    delimiter=",",
                    header="p,P_percolation",
                    comments="",
                )
                print(f"Результаты успешно сохранены в: {csv_filename}")

            pc_L = np.interp(0.5, curr_P, curr_p)
            pc_L_list.append(pc_L)

            plt.plot(
                curr_p,
                curr_P,
                marker="o",
                markersize=3,
                label=f"L = {L} ($p_c \\approx {pc_L:.3f}$)",
            )

        end_time = time.perf_counter()

    elapsed_seconds = end_time - start_time
    minutes = int(elapsed_seconds // 60)
    seconds = elapsed_seconds % 60

    pc_infinity = extrapolate_pc(L_values, pc_L_list)

    plt.axvline(
        x=0.5927,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label="Теория: $p_c(\\infty) \\approx 0.5927$",
    )
    plt.axvline(
        x=pc_infinity,
        color="green",
        linestyle=":",
        linewidth=1.5,
        label=f"Эксперимент: $p_c(\\infty) \\approx {pc_infinity:.4f}$",
    )

    plt.xlabel("Вероятность закраски p", fontsize=12)
    plt.ylabel("Вероятность перколяции P(p)", fontsize=12)
    plt.title(
        f"Зависимость P(p) от L (Монте-Карло, N = {num_trials})", fontsize=14
    )
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.legend(fontsize=10)

    print("\n" + "=" * 50)
    print("РЕЗУЛЬТАТЫ:")
    print(f"Размеры решёток L:          {L_values}")
    print(f"Количество симуляций N:     {num_trials}")
    print(f"Экспериментальное pc(∞):   {pc_infinity:.4f}")
    print(f"Теоретическое значение:      0.5927")
    error_pct = abs(pc_infinity - 0.5927) / 0.5927 * 100
    print(f"Относительная погрешность:  {error_pct:.2f}%")
    print(f"Время выполнения: {minutes} мин {seconds:.2f} сек")
    print("=" * 50 + "\n")

    plt.show()


if __name__ == "__main__":
    main()