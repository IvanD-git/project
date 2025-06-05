import numpy as np
import matplotlib.pyplot as plt
import derivative as dv

# есть массив данных y
x = np.arange(0, 100, 1)
y = np.sin(x) + 0.5 * x


def gradient_descent(arr_y, N = 20, min_step = 100, arr_x = None, plot = True, x0 = None):
    ''' Находит минимум функции, заданной массивами, с помощью градиентного спуска.

    Parameters:
        arr_y (array-like): Значения функции
        arr_x (array-like, optional): Аргументы функции
        N (int): Число итераций
        x0 (float, optional): Начальная точка. Если None, берётся среднее arr_x или 0.
        mn (int): Максимальное значение шага
        plot (bool): Строить ли график функции

    Returns:
        float: Найденная точка минимума

    '''

    # определение массива

    y_plt = arr_y
    if arr_x is None:
        x_plt = np.arange(len(y_plt))
    else:
        x_plt = arr_x


    # график функции.

    if plot is True:
        plt.scatter(x_plt, y_plt)
        plt.title("График функции")
        plt.grid(True)
        plt.show()

    # выбор диапазона
    if x0 is None:
        x0 = x_plt[len(x_plt) // 2]


    # Градиентный спуск

    for i in range(N):
        lmd = 1/(min(i + 1, min_step))
        grad = dv.derivative_at_point(y_plt, x_plt, x0,)
        x0 = x0 - lmd * grad
        x0 = np.clip(x0, x_plt[0], x_plt[-1])

    return x0

gradient_descent(y)




