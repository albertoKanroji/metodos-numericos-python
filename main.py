import numpy as np
import matplotlib.pyplot as plt


def menu_principal():
    print("\nPrograma de Métodos Numéricos")
    print("Seleccione una opción:")
    print("1. Introducción a los métodos numéricos")
    print("2. Solución de ecuaciones no lineales de una variable")
    print("3. Interpolación")
    print("4. Integración numérica")
    print("5. Solución de sistemas de ecuaciones lineales")
    print("0. Salir")

    opcion = int(input("Ingrese la opción deseada: "))
    return opcion


def menu_ecuaciones_no_lineales():
    print("\nSolución de Ecuaciones No Lineales de una Variable")
    print("1. Búsqueda de valores iniciales (Tabulación y graficación)")
    print("2. Método de la Bisección")
    print("3. Método de la Regla Falsa")
    print("4. Método de Newton")
    print("5. Método de la Secante")
    print("0. Volver al menú principal")

    opcion = int(input("Ingrese la opción deseada: "))
    return opcion


def menu_interpolacion():
    print("\nInterpolación")
    print("1. Interpolación de Lagrange")
    print("2. Interpolación de Newton")
    print("0. Volver al menú principal")

    opcion = int(input("Ingrese la opción deseada: "))
    return opcion


def menu_integracion_numerica():
    print("\nIntegración Numérica")
    print("1. Regla Trapecial")
    print("0. Volver al menú principal")

    opcion = int(input("Ingrese la opción deseada: "))
    return opcion


def menu_sistemas_ecuaciones():
    print("\nSolución de Sistemas de Ecuaciones Lineales")
    print("1. Eliminación Gaussiana")
    print("2. Método de Gauss-Jordan")
    print("3. Método de Gauss-Seidel")
    print("0. Volver al menú principal")

    opcion = int(input("Ingrese la opción deseada: "))
    return opcion


def introduccion_metodos_numericos():
    print("\nIntroducción a los Métodos Numéricos")
    print("Errores absoluto y relativo mediante series de Taylor")

    def taylor_series_expansion(x, n):
        """Calcula la serie de Taylor de e^x en torno a 0 hasta el n-ésimo término."""
        return sum((x ** i) / np.math.factorial(i) for i in range(n))

    x = float(input("Ingrese el valor de x: "))
    n = int(input("Ingrese el número de términos de la serie de Taylor: "))

    taylor_approx = taylor_series_expansion(x, n)
    exact_value = np.exp(x)
    absolute_error = abs(exact_value - taylor_approx)
    relative_error = absolute_error / abs(exact_value)

    print(f"Valor exacto: {exact_value}")
    print(f"Aproximación de Taylor: {taylor_approx}")
    print(f"Error absoluto: {absolute_error}")
    print(f"Error relativo: {relative_error}")


def tabulacion_y_graficacion():
    print("\nTabulación y Graficación de Función")

    def f(x):
        return x ** 3 - x - 2  # Ejemplo de función, se puede cambiar

    a = float(input("Ingrese el extremo inferior del intervalo (a): "))
    b = float(input("Ingrese el extremo superior del intervalo (b): "))
    n = int(input("Ingrese el número de puntos: "))

    x = np.linspace(a, b, n)
    y = f(x)

    plt.plot(x, y, label='f(x)')
    plt.axhline(0, color='red', linewidth=0.5)
    plt.axvline(0, color='red', linewidth=0.5)
    plt.title('Tabulación y Graficación de f(x)')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.grid(True)
    plt.show()


def metodo_biseccion():
    print("\nMétodo de la Bisección")

    def f(x):
        return x ** 3 - x - 2  # Ejemplo de función, se puede cambiar

    a = float(input("Ingrese el extremo inferior del intervalo (a): "))
    b = float(input("Ingrese el extremo superior del intervalo (b): "))
    tol = float(input("Ingrese la tolerancia (tol): "))

    if f(a) * f(b) >= 0:
        print("El método de la bisección no es aplicable en este intervalo.")
        return

    while (b - a) / 2.0 > tol:
        c = (a + b) / 2.0
        if f(c) == 0:
            break
        elif f(a) * f(c) < 0:
            b = c
        else:
            a = c

    print(f"La raíz aproximada es: {c}")


def metodo_regla_falsa():
    print("\nMétodo de la Regla Falsa")

    def f(x):
        return x ** 3 - x - 2  # Ejemplo de función, se puede cambiar

    a = float(input("Ingrese el extremo inferior del intervalo (a): "))
    b = float(input("Ingrese el extremo superior del intervalo (b): "))
    tol = float(input("Ingrese la tolerancia (tol): "))

    if f(a) * f(b) >= 0:
        print("El método de la regla falsa no es aplicable en este intervalo.")
        return

    c = a
    while abs(f(c)) > tol:
        c = b - (f(b) * (b - a)) / (f(b) - f(a))
        if f(a) * f(c) < 0:
            b = c
        else:
            a = c

    print(f"La raíz aproximada es: {c}")


def metodo_newton():
    print("\nMétodo de Newton-Raphson")

    def f(x):
        return x ** 3 - x - 2  # Ejemplo de función, se puede cambiar

    def df(x):
        return 3 * x ** 2 - 1  # Derivada de la función

    x0 = float(input("Ingrese el valor inicial (x0): "))
    tol = float(input("Ingrese la tolerancia (tol): "))

    x = x0
    while abs(f(x)) > tol:
        x = x - f(x) / df(x)

    print(f"La raíz aproximada es: {x}")


def metodo_secante():
    print("\nMétodo de la Secante")

    def f(x):
        return x ** 3 - x - 2  # Ejemplo de función, se puede cambiar

    x0 = float(input("Ingrese el primer valor inicial (x0): "))
    x1 = float(input("Ingrese el segundo valor inicial (x1): "))
    tol = float(input("Ingrese la tolerancia (tol): "))

    while abs(f(x1)) > tol:
        x_temp = x1
        x1 = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))
        x0 = x_temp

    print(f"La raíz aproximada es: {x1}")


def interpolacion_lagrange():
    print("\nInterpolación de Lagrange")

    n = int(input("Ingrese el número de puntos: "))
    x = np.zeros(n)
    y = np.zeros(n)

    for i in range(n):
        x[i] = float(input(f"Ingrese x[{i}]: "))
        y[i] = float(input(f"Ingrese y[{i}]: "))

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

    x_val = float(input("Ingrese el valor de x para interpolar: "))
    y_val = P(x_val)
    print(f"El valor interpolado en x={x_val} es y={y_val}")


def interpolacion_newton():
    print("\nInterpolación de Newton")

    n = int(input("Ingrese el número de puntos: "))
    x = np.zeros(n)
    y = np.zeros(n)

    for i in range(n):
        x[i] = float(input(f"Ingrese x[{i}]: "))
        y[i] = float(input(f"Ingrese y[{i}]: "))

    coef = np.copy(y)
    for j in range(1, n):
        for i in range(n - 1, j - 1, -1):
            coef[i] = (coef[i] - coef[i - 1]) / (x[i] - x[i - j])

    def P(x_val):
        result = coef[-1]
        for i in range(n - 2, -1, -1):
            result = result * (x_val - x[i]) + coef[i]
        return result

    x_val = float(input("Ingrese el valor de x para interpolar: "))
    y_val = P(x_val)
    print(f"El valor interpolado en x={x_val} es y={y_val}")


def regla_trapecial():
    print("\nRegla Trapecial")

    def f(x):
        return x ** 2  # Ejemplo de función, se puede cambiar

    a = float(input("Ingrese el límite inferior del intervalo (a): "))
    b = float(input("Ingrese el límite superior del intervalo (b): "))
    n = int(input("Ingrese el número de subintervalos (n): "))

    h = (b - a) / n
    result = 0.5 * (f(a) + f(b))

    for i in range(1, n):
        result += f(a + i * h)

    result *= h
    print(f"El valor aproximado de la integral es: {result}")


def eliminacion_gaussiana():
    print("\nEliminación Gaussiana")

    n = int(input("Ingrese el número de ecuaciones: "))
    A = np.zeros((n, n))
    b = np.zeros(n)

    print("Ingrese la matriz de coeficientes:")
    for i in range(n):
        for j in range(n):
            A[i, j] = float(input(f"A[{i}][{j}] = "))

    print("Ingrese el vector de términos independientes:")
    for i in range(n):
        b[i] = float(input(f"b[{i}] = "))

    for i in range(n):
        for k in range(i + 1, n):
            factor = A[k, i] / A[i, i]
            for j in range(i, n):
                A[k, j] -= factor * A[i, j]
            b[k] -= factor * b[i]

    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i + 1:], x[i + 1:])) / A[i, i]

    print(f"La solución es: {x}")


def metodo_gauss_jordan():
    print("\nMétodo de Gauss-Jordan")

    n = int(input("Ingrese el número de ecuaciones: "))
    A = np.zeros((n, n + 1))

    print("Ingrese la matriz aumentada:")
    for i in range(n):
        for j in range(n + 1):
            A[i, j] = float(input(f"A[{i}][{j}] = "))

    for i in range(n):
        A[i] = A[i] / A[i, i]
        for j in range(n):
            if i != j:
                A[j] = A[j] - A[j, i] * A[i]

    x = A[:, -1]
    print(f"La solución es: {x}")


def metodo_gauss_seidel():
    print("\nMétodo de Gauss-Seidel")

    n = int(input("Ingrese el número de ecuaciones: "))
    A = np.zeros((n, n))
    b = np.zeros(n)
    x = np.zeros(n)

    print("Ingrese la matriz de coeficientes:")
    for i in range(n):
        for j in range(n):
            A[i, j] = float(input(f"A[{i}][{j}] = "))

    print("Ingrese el vector de términos independientes:")
    for i in range(n):
        b[i] = float(input(f"b[{i}] = "))

    tol = float(input("Ingrese la tolerancia (tol): "))
    max_iter = int(input("Ingrese el número máximo de iteraciones: "))

    for iteration in range(max_iter):
        x_new = np.copy(x)
        for i in range(n):
            sum = b[i]
            for j in range(n):
                if i != j:
                    sum -= A[i, j] * x_new[j]
            x_new[i] = sum / A[i, i]
        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new

    print(f"La solución aproximada es: {x}")


def main():
    while True:
        opcion = menu_principal()

        if opcion == 1:
            introduccion_metodos_numericos()
        elif opcion == 2:
            while True:
                opcion_submenu = menu_ecuaciones_no_lineales()
                if opcion_submenu == 1:
                    tabulacion_y_graficacion()
                elif opcion_submenu == 2:
                    metodo_biseccion()
                elif opcion_submenu == 3:
                    metodo_regla_falsa()
                elif opcion_submenu == 4:
                    metodo_newton()
                elif opcion_submenu == 5:
                    metodo_secante()
                elif opcion_submenu == 0:
                    break
                else:
                    print("Opción no válida, intente nuevamente.")
        elif opcion == 3:
            while True:
                opcion_submenu = menu_interpolacion()
                if opcion_submenu == 1:
                    interpolacion_lagrange()
                elif opcion_submenu == 2:
                    interpolacion_newton()
                elif opcion_submenu == 0:
                    break
                else:
                    print("Opción no válida, intente nuevamente.")
        elif opcion == 4:
            while True:
                opcion_submenu = menu_integracion_numerica()
                if opcion_submenu == 1:
                    regla_trapecial()
                elif opcion_submenu == 0:
                    break
                else:
                    print("Opción no válida, intente nuevamente.")
        elif opcion == 5:
            while True:
                opcion_submenu = menu_sistemas_ecuaciones()
                if opcion_submenu == 1:
                    eliminacion_gaussiana()
                elif opcion_submenu == 2:
                    metodo_gauss_jordan()
                elif opcion_submenu == 3:
                    metodo_gauss_seidel()
                elif opcion_submenu == 0:
                    break
                else:
                    print("Opción no válida, intente nuevamente.")
        elif opcion == 0:
            print("Saliendo del programa...")
            break
        else:
            print("Opción no válida, intente nuevamente.")


if __name__ == "__main__":
    main()
