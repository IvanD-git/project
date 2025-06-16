import numpy as np
import matplotlib.pyplot as plt

class Approximation:
    def mnk_straight(self, y_arr, x_arr=None, N=None, plot=True, k=True, b=True):
        """
        Выполняет линейную аппроксимацию данных методом наименьших квадратов (МНК).
        Аппроксимирует зависимость Y от X прямой вида: y = k*x + b.
        """
        if len(y_arr) == 0:
            raise ValueError("Входной массив 'y_arr' пуст.")

        if x_arr is None:
            x_arr = np.arange(len(y_arr))
        else:
            x_arr = np.asarray(x_arr)

        if N is None:
            N = len(y_arr)

        x = x_arr[:N]
        y = y_arr[:N]

        if len(x) != len(y):
            raise ValueError("Длины массивов x_arr и y_arr должны совпадать.")

        math_expectX = x.sum() / N
        math_expectY = y.sum() / N

        a1 = np.dot(x, y) / N
        a2 = np.dot(x, x) / N

        if abs(a2 - math_expectX ** 2) < 1e-10:
            raise ValueError("Данные слишком близки к одной точке, невозможно вычислить наклон.")

        if k is True:
            k_approx = (a1 - math_expectX * math_expectY) / (a2 - math_expectX ** 2)
        else:
            k_approx = float(k)

        if b is True:
            b_approx = math_expectY - k_approx * math_expectX
        else:
            b_approx = float(b)

        f_approx = k_approx * x + b_approx

        if plot:
            from plotter import Plot
            Plot().plot(x, y, f_approx, label=f'Аппроксимация: y = {k_approx:.4f}x + {b_approx:.4f}')

        return k_approx, b_approx, f_approx

    def mnk_exponent(self, y_arr, x_arr=None, N=None, plot=True, k=True, a=True):
        """
        Выполняет экспоненциальную аппроксимацию данных методом наименьших квадратов (МНК).
        Аппроксимирует зависимость Y от X функцией вида: y = a * exp(k * x).
        """
        if len(y_arr) == 0:
            raise ValueError("Входной массив 'y_arr' пуст.")

        if any(val <= 0 for val in y_arr):
            raise ValueError("Все значения в y_arr должны быть положительными для логарифмирования.")

        if x_arr is None:
            x_arr = np.arange(len(y_arr))
        else:
            x_arr = np.asarray(x_arr)

        if N is None:
            N = len(y_arr)

        x = x_arr[:N]
        y = y_arr[:N]

        if len(x) != len(y):
            raise ValueError("Длины массивов x_arr и y_arr должны совпадать.")

        log_y = np.log(y)

        # Вычисляем средние значения независимо от фиксирования k
        math_expectX = x.sum() / N
        math_expectY = log_y.sum() / N

        a1 = np.dot(x, log_y) / N
        a2 = np.dot(x, x) / N

        if abs(a2 - math_expectX ** 2) < 1e-10 and k is True:
            raise ValueError("Данные слишком близки к одной точке, невозможно вычислить экспоненту.")

        if k is True:
            k_approx = (a1 - math_expectX * math_expectY) / (a2 - math_expectX ** 2)
        else:
            k_approx = float(k)

        if a is True:
            log_a_approx = math_expectY - k_approx * math_expectX
            a_approx = np.exp(log_a_approx)
        else:
            a_approx = float(a)

        f_approx = a_approx * np.exp(k_approx * x)

        if plot:
            from plotter import Plot
            Plot().plot(x, y, f_approx, label=f'Аппроксимация: y = {a_approx:.4f}·exp({k_approx:.4f}x)')

        return k_approx, a_approx, f_approx

    def mnk_logarithmic(self, y_arr, x_arr=None, N=None, plot=True, a=True, b=True):
        """
        Выполняет аппроксимацию данных методом наименьших квадратов (МНК)
        по логарифмической функции вида: y = a + b * ln(x).
        """
        if len(y_arr) == 0:
            raise ValueError("Входной массив 'y_arr' пуст.")

        if x_arr is None:
            x_arr = np.arange(1, len(y_arr) + 1)
        else:
            x_arr = np.asarray(x_arr)
            if any(x <= 0 for x in x_arr):
                raise ValueError("Все значения x должны быть > 0 для вычисления ln(x).")

        if N is None:
            N = len(y_arr)

        x = x_arr[:N]
        y = y_arr[:N]

        if len(x) != len(y):
            raise ValueError("Длины массивов x_arr и y_arr должны совпадать.")

        ln_x = np.log(x)

        if b is True and a is True:
            sum_y = y.sum()
            sum_lnx = ln_x.sum()
            sum_lnx2 = np.dot(ln_x, ln_x)
            sum_ylx = np.dot(y, ln_x)

            denominator = N * sum_lnx2 - sum_lnx ** 2
            if abs(denominator) < 1e-10:
                raise ValueError("Данные слишком близки к одной точке, невозможно вычислить коэффициенты.")

            b_approx = (N * sum_ylx - sum_y * sum_lnx) / denominator
            a_approx = (sum_y - b_approx * sum_lnx) / N
        elif b is True and a is not True:
            a_approx = float(a)
            b_approx = np.dot(y - a_approx, ln_x) / np.dot(ln_x, ln_x)
        elif a is True and b is not True:
            b_approx = float(b)
            a_approx = (y - b_approx * ln_x).mean()
        elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
            a_approx = float(a)
            b_approx = float(b)
        else:
            raise ValueError("Параметры 'a' и 'b' должны быть либо True, либо числами.")

        f_approx = a_approx + b_approx * ln_x

        if plot:
            from plotter import Plot
            Plot().plot(x, y, f_approx, label=f'Аппроксимация: y = {a_approx:.4f} + {b_approx:.4f}·ln(x)')

        return a_approx, b_approx, f_approx

    def approx_moving_average(self, y_arr, x_arr=None, window=5, plot=True):
        """
        Выполняет аппроксимацию данных методом скользящего среднего.
        """
        if len(y_arr) == 0:
            raise ValueError("Входной массив 'y_arr' пуст.")

        if not isinstance(window, int) or window <= 0:
            raise ValueError("Параметр 'window' должен быть положительным целым числом.")

        if window > len(y_arr):
            raise ValueError("Размер окна не может превышать длину входного массива.")

        if x_arr is None:
            x_arr = np.arange(len(y_arr))
        else:
            x_arr = np.asarray(x_arr)

        if len(x_arr) != len(y_arr):
            raise ValueError("Длины массивов x_arr и y_arr должны совпадать.")

        y = np.asarray(y_arr, dtype=np.float64)
        cumsum = np.cumsum(np.insert(y, 0, 0))
        y_smooth = (cumsum[window:] - cumsum[:-window]) / float(window)
        x_smooth = x_arr[window - 1:]

        if plot:
            from plotter import Plot
            Plot().plot(x_arr, y_arr, y_smooth, label=f'Скользящее среднее (окно={window})', x_approx=x_smooth)

        return x_smooth, y_smooth

    def mnk_polynomial(self, y_arr, x_arr=None, N=None, degree=3, plot=True):
        """
        Выполняет полиномиальную аппроксимацию данных методом наименьших квадратов (МНК).
        Аппроксимирует зависимость Y от X полиномом n-й степени.
        """
        if len(y_arr) == 0:
            raise ValueError("Входной массив 'y_arr' пуст.")

        if x_arr is None:
            x_arr = np.arange(len(y_arr))
        else:
            x_arr = np.asarray(x_arr)

        if N is None:
            N = len(y_arr)

        x = x_arr[:N]
        y = y_arr[:N]

        if len(x) != len(y):
            raise ValueError("Длины массивов x_arr и y_arr должны совпадать.")

        A = np.vander(x, degree + 1, increasing=True)
        ATA = A.T @ A
        ATy = A.T @ y
        coefficients = np.linalg.solve(ATA, ATy)
        f_approx = np.polyval(np.flip(coefficients), x)

        # Формирование строки полинома
        poly_str = "y = "
        for i, coef in enumerate(coefficients):
            """i  (степень переменной x) и coef (значение коэффициента) """
            coef = round(coef, 4)
            if abs(coef) < 1e-10:  # Пропускаем коэффициенты, близкие к нулю
                continue
            sign = "+" if coef >= 0 and i > 0 else "" if i == 0 else "-"
            if i == 0:
                term = f"{sign}{abs(coef):.4f}"
            elif i == 1:
                term = f"{sign}{abs(coef):.4f}x"
            else:
                term = f"{sign}{abs(coef):.4f}x^{i}"
            poly_str += term + " "
        poly_str = poly_str.strip() or "y = 0"

        if plot:
            from plotter import Plot
            Plot().plot(x, y, f_approx, label=f'Полином: {poly_str}')

        return coefficients, f_approx, poly_str
