import sys
sys.path.append(".")

from typing import Callable, Tuple, List, NewType
import pandas as pd
import matplotlib.pyplot as plt
import math
import DEMethod


Float = NewType('step', float)
Frame = NewType('frame', pd.DataFrame)

class IRK2Method(DEMethod.DESolveMethod):
    def __init__(self):
        pass

    @staticmethod
    def n_iterations(segment: List[float], step: Float) -> int:
        return int((segment[1] - segment[0]) / step)

    @staticmethod
    def update_y(f: Callable[[float, float], float],
                x: Float,
                y: Float,
                step: Float) -> int:

        temporary = y + step * f(x, y)
        y_new = y + step * (f(x, y) + f(x + step, temporary)) / 2

        return y_new

    def solve(self, f: Callable[[float, float], float],
              initial_dot: Tuple[float, float],
              segment: List[float],
              step: Float) -> pd.DataFrame:

        if len(segment) != 2:
            raise Exception('Segment must be a list of size = 2.')

        table = []
        x_0, y_0 = initial_dot[0], initial_dot[1]
        table.append(
            {
                'i': 0,
                'x': x_0,
                'y': y_0,
                'func': f(x_0, y_0)
            }
        )

        for i in range(1, self.n_iterations(segment, step) + 1):
            x_last, y_last = table[i - 1]['x'], table[i - 1]['y']
            x_new = x_last + step
            y_new = self.update_y(f, x_last, y_last, step)
            table.append(
                {
                    'i': i,
                    'x': x_new,
                    'y': y_new,
                    'func': f(x_new, y_new)
                }
            )

        df = pd.DataFrame(table)
        return df

    def __str__(self):
        return 'Euler method of solving differential equations.'


a = IRK2Method()
result = a.solve(lambda x, y: 100 * (y - math.cos(x)), (0, 1), [0, 1], 0.1)
x_axis, y_axis = result['x'], result['y']
fig = plt.figure()

# g = lambda x: 3/4 * math.e ** (-2 * x) + 1/2 * x ** 2 - 1/2 * x + 1/4
# true_func = [g(x) for x in x_axis]
# should_be, = plt.plot(x_axis, true_func, color = 'red', linewidth = 2, label='true')
got_euler, = plt.plot(x_axis, y_axis, color = 'black', linewidth = 2, label='Euler')
# plt.legend(handles=[should_be, got_euler])
plt.show()