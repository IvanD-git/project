import matplotlib.pyplot as plt
import numpy as np

class Plot:
    def __init__(self):
        self.figure = None
        self.ax = None

    def clear_plot(self):
        """Очищает текущий график."""
        if self.figure is not None:
            plt.close(self.figure)
        self.figure = plt.figure(figsize=(10, 6))
        self.ax = self.figure.add_subplot(111)
        self.ax.grid(True)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_title('Аппроксимация')

    def plot(self, x, y, f_approx, label="Аппроксимация", x_approx=None, is_trend=False):
        """
        Построение графика. Если is_trend=True, рисуется линия тренда, иначе — исходные данные.
        Если x_approx передан, используется для линии тренда.
        """
        if self.figure is None or self.ax is None:
            self.clear_plot()

        if not is_trend:
            # Рисуем исходные данные как точки
            self.ax.scatter(x, y, s=20, c='blue', label='Исходные данные')
        else:
            # Рисуем линию тренда
            if x_approx is None:
                x_approx = x
            self.ax.plot(x_approx, f_approx, c='red', label=label)

        self.ax.legend()
        plt.draw()
        plt.show(block=False)
        plt.pause(0.001)
