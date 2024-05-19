import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# Método de la Secante
def metodo_secante(f, x0, x1, tol, max_iter):
    for _ in range(max_iter):
        if abs(f(x1)) < tol:
            break
        try:
            x_temp = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))
        except ZeroDivisionError:
            messagebox.showerror("Error", "División por cero en el método de la secante.")
            return None
        x0, x1 = x1, x_temp
    return x1


# Interpolación de Lagrange
def interpolacion_lagrange(x, y, x_val):
    n = len(x)

    def L(k, x_val):
        result = 1
        for i in range(n):
            if i != k:
                result *= (x_val - x[i]) / (x[k] - x[i])
        return result

    def P(x_val):
        result = 0
        for k in range(n):
            result += y[k] * L(k, x_val)
        return result

    return P(x_val)


# Regla Trapecial
def regla_trapecial(f, a, b, n):
    h = (b - a) / n
    result = 0.5 * (f(a) + f(b))
    for i in range(1, n):
        result += f(a + i * h)
    result *= h
    return result


# Graficar método de la Secante
def plot_secante(f, x0, x1, tol, max_iter):
    x_vals = np.linspace(x0, x1, 400)
    y_vals = f(x_vals)

    fig, ax = plt.subplots()
    ax.plot(x_vals, y_vals, label="f(x)")
    ax.axhline(0, color='red', linewidth=0.5)
    ax.axvline(0, color='red', linewidth=0.5)

    root = metodo_secante(f, x0, x1, tol, max_iter)
    if root is not None:
        ax.plot(root, f(root), 'go', label=f'Root: {root:.5f}')

    ax.legend()
    ax.grid(True)

    return fig


# Graficar interpolación de Lagrange
def plot_interpolacion(x, y, x_val, y_val):
    fig, ax = plt.subplots()
    ax.plot(x, y, 'ro', label="Puntos")
    x_dense = np.linspace(min(x), max(x), 400)
    y_dense = [interpolacion_lagrange(x, y, xi) for xi in x_dense]
    ax.plot(x_dense, y_dense, label="Interpolación de Lagrange")
    ax.plot(x_val, y_val, 'go', label=f'Interpolación: y({x_val}) = {y_val:.5f}')

    ax.legend()
    ax.grid(True)

    return fig


# Graficar regla trapecial
def plot_trapecial(f, a, b, n, result):
    x_vals = np.linspace(a, b, 400)
    y_vals = f(x_vals)

    fig, ax = plt.subplots()
    ax.plot(x_vals, y_vals, label="f(x)")
    x_trap = np.linspace(a, b, n + 1)
    y_trap = f(x_trap)
    ax.plot(x_trap, y_trap, 'bo', label="Puntos de integración")
    for i in range(n):
        ax.add_patch(
            plt.Polygon([(x_trap[i], 0), (x_trap[i], y_trap[i]), (x_trap[i + 1], y_trap[i + 1]), (x_trap[i + 1], 0)],
                        closed=True, fill=None, edgecolor='gray'))
    ax.axhline(0, color='red', linewidth=0.5)
    ax.axvline(0, color='red', linewidth=0.5)

    ax.legend()
    ax.grid(True)

    return fig


# Clase principal de la aplicación
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Métodos Numéricos con Interfaz Gráfica")
        self.geometry("800x600")

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(pady=10, expand=True)

        self.frames = {}
        for F in (SecanteFrame, InterpolacionFrame, TrapecialFrame):
            page_name = F.__name__
            frame = F(parent=self.notebook, controller=self)
            self.frames[page_name] = frame
            self.notebook.add(frame, text=page_name.replace("Frame", ""))

    def show_frame(self, page_name):
        frame = self.frames[page_name]
        frame.tkraise()


# Frame para el método de la Secante
class SecanteFrame(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        self.create_widgets()

    def create_widgets(self):
        tk.Label(self, text="Método de la Secante").pack()

        self.f_entry = tk.Entry(self)
        self.f_entry.pack()
        self.f_entry.insert(0, "x**3 - x - 2")  # Ejemplo de función

        self.x0_entry = tk.Entry(self)
        self.x0_entry.pack()
        self.x0_entry.insert(0, "1")

        self.x1_entry = tk.Entry(self)
        self.x1_entry.pack()
        self.x1_entry.insert(0, "2")

        self.tol_entry = tk.Entry(self)
        self.tol_entry.pack()
        self.tol_entry.insert(0, "1e-5")

        self.max_iter_entry = tk.Entry(self)
        self.max_iter_entry.pack()
        self.max_iter_entry.insert(0, "100")

        self.plot_button = tk.Button(self, text="Calcular y Graficar", command=self.plot)
        self.plot_button.pack()

        self.canvas = None

    def plot(self):
        f = lambda x: eval(self.f_entry.get())
        x0 = float(self.x0_entry.get())
        x1 = float(self.x1_entry.get())
        tol = float(self.tol_entry.get())
        max_iter = int(self.max_iter_entry.get())

        fig = plot_secante(f, x0, x1, tol, max_iter)

        if self.canvas:
            self.canvas.get_tk_widget().pack_forget()
        self.canvas = FigureCanvasTkAgg(fig, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack()


# Frame para la interpolación de Lagrange
class InterpolacionFrame(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        self.create_widgets()

    def create_widgets(self):
        tk.Label(self, text="Interpolación de Lagrange").pack()

        self.n_entry = tk.Entry(self)
        self.n_entry.pack()
        self.n_entry.insert(0, "4")

        self.points_entry = tk.Text(self, height=4)
        self.points_entry.pack()
        self.points_entry.insert(tk.END, "0 1\n1 2\n2 0\n3 3")  # Ejemplo de puntos

        self.x_val_entry = tk.Entry(self)
        self.x_val_entry.pack()
        self.x_val_entry.insert(0, "1.5")

        self.plot_button = tk.Button(self, text="Calcular y Graficar", command=self.plot)
        self.plot_button.pack()

        self.canvas = None

    def plot(self):
        n = int(self.n_entry.get())
        points = self.points_entry.get("1.0", tk.END).strip().split("\n")
        x = []
        y = []
        for point in points:
            xi, yi = map(float, point.split())
            x.append(xi)
            y.append(yi)
        x_val = float(self.x_val_entry.get())

        y_val = interpolacion_lagrange(x, y, x_val)
        fig = plot_interpolacion(x, y, x_val, y_val)

        if self.canvas:
            self.canvas.get_tk_widget().pack_forget()
        self.canvas = FigureCanvasTkAgg(fig, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack()


# Frame para la regla trapecial
class TrapecialFrame(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        self.create_widgets()

    def create_widgets(self):
        tk.Label(self, text="Regla Trapecial").pack()

        self.f_entry = tk.Entry(self)
        self.f_entry.pack()
        self.f_entry.insert(0, "x**2")  # Ejemplo de función

        self.a_entry = tk.Entry(self)
        self.a_entry.pack()
        self.a_entry.insert(0, "0")

        self.b_entry = tk.Entry(self)
        self.b_entry.pack()
        self.b_entry.insert(0, "1")

        self.n_entry = tk.Entry(self)
        self.n_entry.pack()
        self.n_entry.insert(0, "10")

        self.plot_button = tk.Button(self, text="Calcular y Graficar", command=self.plot)
        self.plot_button.pack()

        self.canvas = None

    def plot(self):
        f = lambda x: eval(self.f_entry.get())
        a = float(self.a_entry.get())
        b = float(self.b_entry.get())
        n = int(self.n_entry.get())

        result = regla_trapecial(f, a, b, n)
        fig = plot_trapecial(f, a, b, n, result)

        if self.canvas:
            self.canvas.get_tk_widget().pack_forget()
        self.canvas = FigureCanvasTkAgg(fig, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack()


# Ejecutar la aplicación
if __name__ == "__main__":
    app = App()
    app.mainloop()
