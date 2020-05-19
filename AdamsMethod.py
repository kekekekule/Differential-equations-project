import sys
sys.path.append(".")

from typing import Callable, Tuple, List, NewType
import pandas as pd
import matplotlib.pyplot as plt
import math
import DEMethod

Float = NewType('step', float)
Frame = NewType('frame', pd.DataFrame)

class AdamsMethodII(DEMethod.DESolveMethod):
    def __init__(self):
        pass

    @staticmethod
    def n_iterations(segment: List[float], step: Float) -> int:
        return int((segment[1] - segment[0]) / step)

    @staticmethod
    def add_new_entry(i, table,
                      f: Callable[[float, float], float],
                      x: Float,
                      y: Float):

        table.append(
            {
                'i': i,
                'x': x,
                'y': y,
                'func': f(x, y)
            }
        )

    @staticmethod
    def runge_kutta(f: Callable[[float, float], float],
                    x: Float,
                    y: Float,
                    step: Float) -> Float:
        k_1 = f(x, y)
        k_2 = f(x + step/2, y + step/2 * k_1)
        k_3 = f(x + step/2, y + step/2 * k_2)
        k_4 = f(x + step, y + step * k_3)
        return y + step/6 * (k_1 + 2 * k_2 + 2 * k_3 + k_4)

    # Using Runge-Kutta method for initial points
    def precount(self, table, f: Callable[[float, float], float],
                 initial_dot: Tuple[float, float],
                 step: Float):

        x_0, y_0 = initial_dot[0], initial_dot[1]
        self.add_new_entry(0, table, f, x_0, y_0)

        x_1 = x_0 + step
        y_1 = self.runge_kutta(f, x_0, y_0, step)
        self.add_new_entry(1, table, f, x_1, y_1)

    @DEMethod.support_lambda
    def solve(self, f: Callable[[float, float], float],
              initial_dot: Tuple[float, float],
              segment: List[float],
              step: Float) -> pd.DataFrame:

        if len(segment) != 2:
            raise Exception('Segment must be a list of size = 2.')

        table = []
        self.precount(table, f, initial_dot, step)

        for i in range(2, self.n_iterations(segment, step) + 1):
            x_new = table[i - 1]['x'] + step
            y_new = table[i - 1]['y'] + step * (3/2 * table[i - 1]['func'] - 1/2 * table[i - 2]['func'])
            self.add_new_entry(i, table, f, x_new, y_new)

        df = pd.DataFrame(table)
        return df

    def __str__(self):
        return 'Adams-Bashforth method of solving differential equations.'


a = AdamsMethodII()
result = a.solve(lambda x, y: x ** 2 - 2 * y, (0, 1), [0, 1], 0.1)
x_axis, y_axis = result['x'], result['y']
fig = plt.figure()

g = lambda x: 3/4 * math.e ** (-2 * x) + 1/2 * x ** 2 - 1/2 * x + 1/4
true_func = [g(x) for x in x_axis]
should_be, = plt.plot(x_axis, true_func, color = 'red', linewidth = 2, label='true')
got_adams, = plt.plot(x_axis, y_axis, color = 'black', linewidth = 2, label='Adams-Bashforth')
plt.legend(handles=[should_be, got_adams])
plt.show()
