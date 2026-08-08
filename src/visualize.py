import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.widgets import Slider
import numpy as np
from percolation import HoshenKopelman, generate_grid


def main():
    L = 30
    p = 0.5927

    # 1. Подготовка случайной палитры цветов
    num_colors = 1000
    colors = np.random.rand(num_colors, 3)
    colors[0] = [1.0, 1.0, 1.0]  # Индекс 0 — пустые клетки (Белый)
    RED_INDEX = 1
    colors[RED_INDEX] = [1.0, 0.0, 0.0]  # Индекс 1 зарезервирован под КРАСНЫЙ
    custom_cmap = ListedColormap(colors)

    # 2. Создание окна и элементов управления
    fig, ax = plt.subplots(figsize=(6, 7))
    plt.subplots_adjust(bottom=0.25)

    ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
    slider_p = Slider(
        ax=ax_slider,
        label="Вероятность p ",
        valmin=0.0,
        valmax=1.0,
        valinit=p,
        valstep=0.01,
    )

    # Вспомогательная функция для выделения главного кластера красным
    def prepare_display_grid(hk_instance):
        labeled_grid = hk_instance.run()
        has_perc, perc_id = hk_instance.check_percolation()

        display_grid = labeled_grid.copy()
        if has_perc and perc_id is not None:
            # Все клетки перколяционного кластера красим в зарезервированный RED_INDEX (1)
            display_grid[labeled_grid == perc_id] = RED_INDEX

        return display_grid, has_perc, perc_id

    # 3. Первичная отрисовка
    raw_grid = generate_grid(L, p)
    hk = HoshenKopelman(raw_grid)
    display_grid, has_perc, perc_id = prepare_display_grid(hk)

    img = ax.imshow(display_grid, cmap=custom_cmap, origin="upper")

    # Устанавливаем заголовок
    if has_perc:
        ax.set_title(
            f"p = {p:.2f} | Перколяция: ЕСТЬ (Кластер #{perc_id})",
            color="red",
            fontweight="bold",
        )
    else:
        ax.set_title(f"p = {p:.2f} | Перколяции НЕТ")

    # 4. Функция обновления при движении ползунка
    def update(val):
        new_p = slider_p.val
        new_raw_grid = generate_grid(L, new_p)
        new_hk = HoshenKopelman(new_raw_grid)

        new_display_grid, has_perc, perc_id = prepare_display_grid(new_hk)

        img.set_data(new_display_grid)
        max_val = np.max(new_display_grid)
        img.set_clim(vmin=0, vmax=max_val if max_val > 0 else 1)

        if has_perc:
            ax.set_title(
                f"p = {new_p:.2f} | Перколяция: ЕСТЬ (Кластер #{perc_id})",
                color="red",
                fontweight="bold",
            )
        else:
            ax.set_title(f"p = {new_p:.2f} | Перколяции НЕТ", color="black")

        fig.canvas.draw_idle()

    slider_p.on_changed(update)
    plt.show()


if __name__ == "__main__":
    main()