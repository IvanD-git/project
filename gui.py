import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
import os
import re
from data_processor import Approximation
from plotter import Plot


class GUI:
    """Класс для создания графического интерфейса и управления аппроксимацией."""

    def __init__(self):
        """Инициализация GUI."""
        self.approx = Approximation()
        self.plotter = Plot()
        self.root = tk.Tk()
        self.root.title("Обработка данных")
        self.x_data = None
        self.y_data = None
        self.approx_window = None
        self.setup_initial_window()

    def setup_initial_window(self):
        """Настройка начального окна с полями ввода и кнопками."""

        self.root.grid_rowconfigure(1,weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)

        self.entry_x = tk.Text(self.root, height=10, width=30)
        self.entry_y = tk.Text(self.root, height=10, width=30)

        tk.Label(self.root, text="Введите значения X:").grid(row=0, column=0, padx=5, pady=5)
        self.entry_x.grid(row=1, column=0, padx=5, pady=5, sticky='nsew')

        tk.Label(self.root, text="Введите значения Y:").grid(row=0, column=1, padx=5, pady=5)
        self.entry_y.grid(row=1, column=1, padx=5, pady=5, sticky='nsew')

        btn_frame = tk.Frame(self.root)
        btn_frame.grid(row=2, column=0, columnspan=2, pady=10)

        tk.Button(btn_frame, text="Загрузить данные", command=self.load_file).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Построить график", command=self.show_plot_and_approximation).pack(side=tk.LEFT,padx=5)
        tk.Button(btn_frame, text="Стереть данные в X", command=self.clear_entry_x).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Стереть данные в Y", command=self.clear_entry_y).pack(side=tk.LEFT, padx=5)

    def clear_entry_x(self):
        self.entry_x.delete("1.0", tk.END)

    def clear_entry_y(self):
        self.entry_y.delete("1.0", tk.END)

    def get_data(self):
        """Получение данных из полей ввода."""
        x_input = self.entry_x.get("1.0", tk.END).strip()
        y_input = self.entry_y.get("1.0", tk.END).strip()

        if not x_input or not y_input:
            messagebox.showerror("Ошибка", "Заполните поля X и Y !")
            return None, None

        # Обработка X
        x_lines = x_input.splitlines()
        x_list = []
        for line in x_lines:
            clean_line = line.strip()
            if clean_line:
                parts = re.split(r'[,\s]+', clean_line)
                x_list.extend(parts)

        # Обработка Y
        y_lines = y_input.splitlines()
        y_list = []
        for line in y_lines:
            clean_line = line.strip()
            if clean_line:
                parts = re.split(r'[,\s]+', clean_line)
                y_list.extend(parts)

        if len(x_list) != len(y_list):
            messagebox.showerror("Ошибка", "Количество знчений в X b Y должно совпадать")
            return None, None

        try:
            x = np.array([float(val) for val in x_list])
            y = np.array([float(val) for val in y_list])

            if len(x) == 0 or len(y) == 0:
                messagebox.showerror("Ошибка", "После обработки данных массивы X или Y пусты.")
                return None, None
            if len(x) != len(y):
                messagebox.showerror("Ошибка", "Количество значений X и Y не совпадает после обработки.")
                return None, None
            if np.all(x == x[0]) or np.all(y == y[0]):
                messagebox.showerror("Ошибка", "Все значения X и/или Y одинаковы. Аппроксимация невозможна.")
                return None, None
            return x, y
        except ValueError as e:
            messagebox.showerror("Ошибка", f"Все значения должны быть числовыми. Ошибка: {str(e)}")
            return None, None

    def load_file(self):
        """Загрузка данных из файла."""
        file_path = filedialog.askopenfilename(
            title="Выберите файл",
            filetypes=[("Все файлы", "*.*"),
                       ("CSV файлы", "*.csv"),
                       ("Excel файлы", "*.xlsx *.xls"),
                       ("Текстовые файлы", "*.txt")]
        )
        if not file_path:
            return

        ext = os.path.splitext(file_path)[1].lower()

        try:
            if ext == ".csv":
                df = pd.read_csv(file_path)
            elif ext in [".xlsx", ".xls"]:
                df = pd.read_excel(file_path)
            elif ext == ".txt":
                df = pd.read_csv(file_path, sep=r'\s+', header=None)
                df.columns = ['x', 'y']
            else:
                messagebox.showerror("Ошибка", "Неподдерживаемый формат файла.")
                return

            if len(df.columns) < 2:
                messagebox.showerror("Ошибка", "Файл должен содержать как минимум два столбца (X и Y).")
                return

            if df.iloc[:, 0].isna().any() or df.iloc[:, 1].isna().any():
                messagebox.showerror("Ошибка", "Файл содержит пропущенные значения.")
                return

            x_list = df.iloc[:, 0].astype(str).tolist()
            y_list = df.iloc[:, 1].astype(str).tolist()

            self.entry_x.delete("1.0", tk.END)
            self.entry_y.delete("1.0", tk.END)
            self.entry_x.insert(tk.END, "\n".join(x_list))
            self.entry_y.insert(tk.END, "\n".join(y_list))

        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить файл:\n{str(e)}")

    def show_plot_and_approximation(self):
        """Открытие окна с кнопками и графика одновременно."""
        x, y = self.get_data()
        if x is None or y is None or len(x) == 0 or len(y) == 0:
            return

        self.x_data = x
        self.y_data = y

        # Открываем окно с кнопками аппроксимации
        if self.approx_window is None or not self.approx_window.winfo_exists():
            self.approx_window = tk.Toplevel(self.root, )
            self.approx_window.title("Выбор метода аппроксимации")
            self.approx_window.geometry("300x300")

            btn_frame = tk.Frame(self.approx_window)
            btn_frame.pack(pady=10)

            tk.Button(btn_frame, text="Линейная аппроксимация",
                      command=lambda: self.show_coefficient_dialog("linear")).pack(fill=tk.X, padx=5, pady=5)
            tk.Button(btn_frame, text="Экспоненциальная аппроксимация",
                      command=lambda: self.show_coefficient_dialog("exponential")).pack(fill=tk.X, padx=5, pady=5)
            tk.Button(btn_frame, text="Логарифмическая аппроксимация",
                      command=lambda: self.show_coefficient_dialog("logarithmic")).pack(fill=tk.X, padx=5, pady=5)
            tk.Button(btn_frame, text="Скользящее среднее",
                      command=lambda: self.show_coefficient_dialog("moving_average")).pack(fill=tk.X, padx=5, pady=5)
            tk.Button(btn_frame, text="Полиномиальная аппроксимация",
                      command=lambda: self.show_coefficient_dialog("polynomial")).pack(fill=tk.X, padx=5, pady=5)

        # Показываем исходный график
        self.plotter.clear_plot()
        self.plotter.plot(self.x_data, self.y_data, self.y_data, label="Исходные данные", is_trend=False)

    def show_coefficient_dialog(self, approx_type):
        """Открытие диалогового окна для ввода фиксированных параметров."""
        dialog = tk.Toplevel(self.root)
        dialog.title(f"Настройка {approx_type} аппроксимации")
        dialog.geometry("300x250")

        frame = tk.Frame(dialog)
        frame.pack(padx=10, pady=10)

        # Определяем формулу и параметры в зависимости от типа аппроксимации
        if approx_type == "linear":
            formula = "y = kx + b"
            labels = ["k", "b"]
            defaults = [True, True]
        elif approx_type == "exponential":
            formula = "y = a · exp(kx)"
            labels = ["k", "a"]
            defaults = [True, True]
        elif approx_type == "logarithmic":
            formula = "y = a + b · ln(x)"
            labels = ["a", "b"]
            defaults = [True, True]
        elif approx_type == "moving_average":
            formula = "y = Среднее(y_i) по окну размера n"
            labels = ["Размер окна (n)"]
            defaults = [min(5, len(self.x_data) if self.x_data is not None else 5)]
        elif approx_type == "polynomial":
            formula = "y = a₀ + a₁x + a₂x² + ... + aₙxⁿ"
            labels = ["Степень полинома"]
            defaults = [3]

        tk.Label(frame, text=f"Метод: {formula}", font=("Arial", 10, "bold")).grid(row=0, column=0, columnspan=3,
                                                                                   padx=5, pady=5)

        entries = []
        checkboxes = []

        for i, label in enumerate(labels):
            tk.Label(frame, text=label).grid(row=i + 1, column=0, padx=5, pady=5, sticky="w")
            if approx_type not in ["polynomial", "moving_average"]:
                var = tk.BooleanVar(value=False)
                tk.Checkbutton(frame, text="Фиксировать", variable=var).grid(row=i + 1, column=1, padx=5, pady=5)
                checkboxes.append(var)
            entry = tk.Entry(frame)
            entry.grid(row=i + 1, column=2, padx=5, pady=5)
            entries.append(entry)
            if approx_type == "moving_average":
                entry.insert(0, str(defaults[0]))  # Устанавливаем значение по умолчанию для окна

        # Передаём все нужные данные в отдельный метод
        tk.Button(frame, text="Применить", command=lambda: self.on_apply(
            approx_type, entries, checkboxes, defaults, dialog)).grid(
            row=len(labels) + 1, column=0, columnspan=3, pady=10)

    def on_apply(self, approx_type, entries, checkboxes, defaults, dialog):
        """обработке пользовательского ввода и применении параметров аппроксимации"""
        try:
            params = {}
            if approx_type == "linear":
                k = float(entries[0].get()) if checkboxes[0].get() and entries[0].get().strip() else True
                b = float(entries[1].get()) if checkboxes[1].get() and entries[1].get().strip() else True
                params = {"k": k, "b": b}
            elif approx_type == "exponential":
                k = float(entries[0].get()) if checkboxes[0].get() and entries[0].get().strip() else True
                a = float(entries[1].get()) if checkboxes[1].get() and entries[1].get().strip() else True
                params = {"k": k, "a": a}
            elif approx_type == "logarithmic":
                a = float(entries[0].get()) if checkboxes[0].get() and entries[0].get().strip() else True
                b = float(entries[1].get()) if checkboxes[1].get() and entries[1].get().strip() else True
                params = {"a": a, "b": b}
            elif approx_type == "moving_average":
                window = int(entries[0].get()) if entries[0].get().strip() else defaults[0]
                if window <= 0:
                    raise ValueError("Размер окна должен быть положительным целым числом.")
                if self.x_data is not None and window > len(self.x_data):
                    raise ValueError("Размер окна не может превышать количество точек данных.")
                params = {"window": window}
            elif approx_type == "polynomial":
                degree = int(entries[0].get()) if entries[0].get().strip() else 3
                params = {"degree": degree}

            dialog.destroy()
            self.apply_approximation(approx_type, params)
        except ValueError as e:
            messagebox.showerror("Ошибка", f"Неверный ввод: {str(e)}")

    def apply_approximation(self, approx_type, params):
        """Применение аппроксимации с заданными параметрами."""
        if self.x_data is None or self.y_data is None or len(self.x_data) == 0 or len(self.y_data) == 0:
            messagebox.showerror("Ошибка", "Данные не загружены или пусты.")
            return
        try:
            self.plotter.clear_plot()
            self.plotter.plot(self.x_data, self.y_data, self.y_data, label="Исходные данные", is_trend=False)

            if approx_type == "linear":
                k_approx, b_approx, f_approx = self.approx.mnk_straight(self.y_data, self.x_data, plot=False, **params)
                self.plotter.plot(self.x_data, self.y_data, f_approx,
                                  label=f'Аппроксимация: y = {k_approx:.4f}x + {b_approx:.4f}', is_trend=True)
            elif approx_type == "exponential":
                k_approx, a_approx, f_approx = self.approx.mnk_exponent(self.y_data, self.x_data, plot=False, **params)
                self.plotter.plot(self.x_data, self.y_data, f_approx,
                                  label=f'Аппроксимация: y = {a_approx:.4f}·exp({k_approx:.4f}x)', is_trend=True)
            elif approx_type == "logarithmic":
                a_approx, b_approx, f_approx = self.approx.mnk_logarithmic(self.y_data, self.x_data, plot=False,
                                                                           **params)
                self.plotter.plot(self.x_data, self.y_data, f_approx,
                                  label=f'Аппроксимация: y = {a_approx:.4f} + {b_approx:.4f}·ln(x)', is_trend=True)
            elif approx_type == "moving_average":
                x_smooth, y_smooth = self.approx.approx_moving_average(self.y_data, self.x_data, plot=False, **params)
                self.plotter.plot(self.x_data, self.y_data, y_smooth,
                                  label=f'Скользящее среднее (окно={params["window"]})', x_approx=x_smooth,
                                  is_trend=True)
            elif approx_type == "polynomial":
                coefficients, f_approx, poly_str = self.approx.mnk_polynomial(self.y_data, self.x_data, plot=False,
                                                                              **params)
                self.plotter.plot(self.x_data, self.y_data, f_approx, label=f'Полином: {poly_str}', is_trend=True)
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при выполнении аппроксимации:\n{str(e)}")

    def run(self):
        """Запуск приложения."""
        self.root.mainloop()


if __name__ == "__main__":
    app = GUI()
    app.run()
