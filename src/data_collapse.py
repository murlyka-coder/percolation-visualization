import os
import sys
import matplotlib.pyplot as plt
import numpy as np

def main():

    p_c = 0.5927
    nu = 4/3
    L_values = [20, 40, 60, 80, 100]    
    num_trials = 1000

    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")

    plt.figure(figsize=(9, 6))

    for L in L_values:
        csv_filename = f"percolation_L{L}_N{num_trials}.csv"
        csv_path = os.path.join(data_dir, csv_filename)

        if not os.path.exists(csv_path):
            print(f"Ошибка: Файл {csv_filename} не найден в папке data/!")
            print("Сначала запустите calculate.py для проведения расчётов.")
            return

        data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
        p = data[:, 0]
        P = data[:, 1]

        x_scaled = (p - p_c) * (L ** (1 / nu))    #масштабирование

        plt.plot(
            x_scaled,
            P,
            marker="o",
            markersize=3,
            alpha=0.8,
            label=f"L = {L}",
        )

    plt.axvline(
        x=0,
        color="black",
        linestyle="--",
        alpha=0.6,
        label="Центр перехода ($x = 0$)",
    )

    plt.xlabel(
        "Масштабированная переменная $x = (p - p_c) L^{1/\\nu}$", fontsize=12
    )
    plt.ylabel("Вероятность перколяции P(p)", fontsize=12)
    plt.title(
        "Коллапс данных (Data Collapse) для 2D перколяции ($\\nu = 4/3$)",
        fontsize=14,
    )
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.legend(fontsize=11)

    print("График Коллапса Данных успешно построен из CSV!")
    plt.show()


if __name__ == "__main__":
    main()