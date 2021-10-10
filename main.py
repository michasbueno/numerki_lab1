import math
import numpy as np

def cylinder_area(r: float, h: float) -> float:
    """Obliczenie pola powierzchni walca.
    Szczegółowy opis w zadaniu 1.

    Parameters:
    r (float): promień podstawy walca
    h (float): wysokosć walca

    Returns:
    float: pole powierzchni walca
    """
    if r > 0 and h > 0:
        return math.pi * r * r * 2 + 2 * math.pi * r * h
    else:
        return math.nan


def fib(n: int) -> np.ndarray:
    """Obliczenie pierwszych n wyrazów ciągu Fibonnaciego.
    Szczegółowy opis w zadaniu 3.

    Parameters:
    n (int): liczba określająca ilość wyrazów ciągu do obliczenia

    Returns:
    np.ndarray: wektor n pierwszych wyrazów ciągu Fibonnaciego.
    """
    output = np.ndarray(shape=(1, 2), dtype=int)
    output[0, 0], output[0, 1] = 0, 1
    if n > 2:
        output.resize((1, n))
        output[0, 0], output[0, 1] = 0, 1
        for i in range(2, n):
            output[0, i] = output[0, i-1] + output[0, i-2]
    return output[0, :n]


def matrix_calculations(a: float):
    """Funkcja zwraca wartości obliczeń na macierzy stworzonej
    na podstawie parametru a.
    Szczegółowy opis w zadaniu 4.

    Parameters:
    a (float): wartość liczbowa

    Returns:
    touple: krotka zawierająca wyniki obliczeń
    (Minv, Mt, Mdet) - opis parametrów w zadaniu 4.
    """
    return None


def custom_matrix(m: int, n: int):
    """Funkcja zwraca macierz o wymiarze mxn zgodnie
    z opisem zadania 7.

    Parameters:
    m (int): ilość wierszy macierzy
    n (int): ilość kolumn macierzy

    Returns:
    np.ndarray: macierz zgodna z opisem z zadania 7.
    """
    return None

print(fib(50))