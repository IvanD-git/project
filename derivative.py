import numpy as np
import matplotlib.pyplot as plt


def derivative_arr(y, x=None, order=1):
    """
    Вычисляет численную производную dy/dx.
    """

    if x is None:
        x = np.arange(len(y))

    for _ in range(order):
        y = np.diff(y)/ np.diff(x)

    return y


def derivative_at_point(y_values, x_values = None, x0 = None):
    """
        Оценивает первую производную в заданной точке по массивам y (и x).

        Параметры:
        - y_values : обязательный, массив значений функции
        - x_values : необязательный, массив аргументов. Если None, используется равномерная сетка (шаг = 1)
        - x0       : точка, в которой нужно найти производную.
                     Если x_values == None, то x0 — индекс точки в массиве [0 ... len(y)-1]
        - method   : метод ('forward', 'backward', 'central')

        Возвращает:
        - Значение производной в точке x0
        """


    # Если x_values не заданы, то создаём их как индексы y_values
    if x_values is None:
        x_values = np.arange(len(y_values))
        using_index = True
    else:
        using_index = False


    # Если x0  не задан, то используем среднюю точку
    if x0 is None:
        x0 = x_values[len(y_values) // 2]


    #Ближайший индекс к x0
    if using_index:
        idx = int(round(x0))
        if idx < 0 or idx >= len(y_values):
            raise ValueError('Значение x0 вне диапазона')
    else:
        idx = np.abs(x_values - x0).argmin()


     # Задаём метод вычисления производной
    if idx == 0:
        method = 'forward'
    elif idx == len(y_values) - 1:
        method = 'backward'
    else:
        method = 'central'


    # Вычисление производной в зависимости от метода
    if method == 'central':
        # Центральная разность: f'(x) ≈ (f(x+h) - f(x-h)) / (2h)
        if idx == 0 or idx == len(x_values) - 1:
            raise ValueError("Нельзя использовать центральную разность на границе")
        h = (x_values[idx + 1] - x_values[idx - 1]) / 2
        return (y_values[idx + 1] - y_values[idx - 1]) / (2 * h)

    elif method == 'forward':
        if idx == len(y_values) - 1:
            raise ValueError("Нельзя использовать forward на последнем элементе.")
        h = x_values[idx + 1] - x_values[idx]
        return (y_values[idx + 1] - y_values[idx]) / h

    elif method == 'backward':
        if idx == 0:
            raise ValueError("Нельзя использовать backward на первом элементе.")
        h = x_values[idx] - x_values[idx - 1]
        return (y_values[idx] - y_values[idx - 1]) / h

    else:
        raise ValueError("Неизвестный метод. Допустимые значения: 'central', 'forward', 'backward'")





