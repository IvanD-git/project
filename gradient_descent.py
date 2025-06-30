import numpy as np
import matplotlib.pyplot as plt
import derivative as dv

# есть только массив данных y
x = np.arange(0, 100, 1)

y = x*x + np.random.normal(10, 100, size = x.shape)


def gradient_descent(arr_y, arr_x = None, N = 20, min_step = 100, plot = True, x0 = None):
    ''' Находит минимум функции, заданной массивами, с помощью градиентного спуска.

    Parameters:
        arr_y (array-like): Значения функции
        arr_x (array-like, optional): Аргументы функции
        N (int): Число итераций
        x0 (float, optional): Начальная точка. Если None, берётся среднее arr_x или 0.
        min_step (int): Максимальное значение шага
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
        learning_rate = 1/(min(i + 1, min_step))
        grad = dv.derivative_at_point(y_plt, x_plt, x0,)
        x0 = x0 - learning_rate * grad
        x0 = np.clip(x0, x_plt[0], x_plt[-1])

    return x0


def gradient_descent_2d(A, B, C, N=100, learning_rate=0.1, x0=None, plot=True):
    """
    Находит минимум функции f(x, y) = A*x^2 + B*y^2 + C методом градиентного спуска.

    Parameters:
        A (float): Коэффициент при x^2
        B (float): Коэффициент при y^2
        C (float): Константа
        N (int): Число итераций
        learning_rate (float): Начальный шаг (скорость обучения)
        x0 (array-like, optional): Начальная точка [x, y]. Если None, используется [10, 10]
        plot (bool): Строить ли график траектории спуска

    Returns:
        array: Точка минимума [x_min, y_min]
    """

    # Начальная точка
    if x0 is None:
        x0 = np.array([10.0, 10.0])
    else:
        x0 = np.array(x0, dtype=float)

    path = [x0.copy()]

    # Градиентный спуск
    for i in range(N):
        df_dx = 2 * A * x0[0]
        df_dy = 2 * B * x0[1]
        grad = np.array([df_dx, df_dy])

        x0 = x0 - learning_rate * grad
        path.append(x0.copy())

    print(f"Найденная точка минимума: x = {x0[0]:.4f}, y = {x0[1]:.4f}")

    # Визуализация
    if plot:
        path = np.array(path)

        # Создаём сетку для графика
        x_vals = np.linspace(-15, 15, 400)
        y_vals = np.linspace(-15, 15, 400)
        X, Y = np.meshgrid(x_vals, y_vals)
        Z = A * X**2 + B * Y**2 + C

        # 3D график
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(projection='3d')
        ax.plot_surface(X, Y, Z)
        ax.plot(path[:, 0], path[:, 1], [A*x**2 + B*y**2 + C for x, y in path], color='red', linewidth=2)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('f(x, y)')
        ax.set_title("Градиентный спуск на графике функции")
        plt.show()

    return x0

gradient_descent_2d(2,3,4)
