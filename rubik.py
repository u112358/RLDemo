# MIT License
#
# Copyright (c) 2017 BingZhang Hu
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
from itertools import product, combinations

front = 0
right = 1
top = 2
back = 3
left = 4
bottom = 5

left_top = 0
right_top = 1
right_bottom = 2
left_bottom = 3


class Rubik():
    def __init__(self):
        # for the whole rubik, surface are indexed by
        self.action_space = ['t1','t2','t3','r1','r2','r3','f1','f2','f3']
        self.n_actions = len(self.action_space)
        self.count=0
        self.surface = [[0 for x in range(4)] for y in range(6)]
        # self.surface[front] = ['r', 'r', 'b', 'w']
        # self.surface[right] = ['w', 'g', 'm', 'm']
        # self.surface[top] = ['b', 'y', 'g', 'y']
        # self.surface[back] = ['m', 'm', 'r', 'w']
        # self.surface[left] = ['w', 'r', 'r', 'b']
        # self.surface[bottom] = ['b', 'y', 'g', 'y']
        # self.state = np.squeeze(np.reshape(self.surface,24))
        self.reset()
        # plt.ion()
        # self.fig = plt.figure()
        #
        # r = [-1, 0, 1]
        # points = np.array(list(product(r, r, r)))
        # self.points = np.delete(points, 13, 0)
        # self.ax = Axes3D(self.fig)
        self.update_rubik()

    def reset(self):
        self.surface[front] = ['w', 'g', 'g', 'b']
        self.surface[right] = ['r', 'y', 'm', 'r']
        self.surface[top] = ['g', 'r', 'w', 'r']
        self.surface[back] = ['b', 'm', 'b', 'w']
        self.surface[left] = ['y', 'b', 'y', 'w']
        self.surface[bottom] = ['m', 'y', 'g', 'm']
        self.state = np.squeeze(np.reshape(self.surface,24))
        self.count=0
        #self.update_rubik()
    def take_action(self, action):

        if action == 0:   # top 1
            self.twist_top()
            print 't1 ',
        elif action == 1:   # top 2
            self.twist_top_n(2)
            print 't2 ',
        elif action == 2:   # top 3
            self.twist_top_n(3)
            print 't3 ',
        elif action == 3:  # right 1
            self.twist_right()
            print 'r1 ',
        elif action == 4:  # right 2
            self.twist_right_n(2)
            print 'r2 ',
        elif action == 5:  # right 3
            self.twist_right_n(3)
            print 'r3 ',
        elif action == 6:  # front 1
            self.twist_front()
            print 'f1 ',
        elif action == 7:  # front 2
            self.twist_front_n(2)
            print 'f2 ',
        elif action == 8:  # front 3
            self.twist_front_n(3)
            print 'f3 ',
        self.count+=1
        s_ = self.state  # next state

        # reward function
        if check(s_):
            reward = 1
            done = True
        else:
            reward = 0
            done = False

        return s_, reward, done

    def twist_right(self):
        """Performs operation blah."""
        previous = np.copy(self.surface)

        self.surface[front][right_top] = previous[bottom][right_top]
        self.surface[front][right_bottom] = previous[bottom][right_bottom]

        self.surface[top][right_top] = previous[front][right_top]
        self.surface[top][right_bottom] = previous[front][right_bottom]

        self.surface[back][left_bottom] = previous[top][right_top]
        self.surface[back][left_top] = previous[top][right_bottom]

        self.surface[bottom][right_top] = previous[back][left_bottom]
        self.surface[bottom][right_bottom] = previous[back][left_top]

        self.surface[right][left_top] = previous[right][left_bottom]
        self.surface[right][right_top] = previous[right][left_top]
        self.surface[right][right_bottom] = previous[right][right_top]
        self.surface[right][left_bottom] = previous[right][right_bottom]

        del previous
        self.state = np.squeeze(np.reshape(self.surface,24))
        self.update_rubik()

    def twist_right_n(self, k):
        for _ in range(k):
            self.twist_right()

    def twist_top(self):
        previous = np.copy(self.surface)

        self.surface[front][left_top] = previous[right][left_top]
        self.surface[front][right_top] = previous[right][right_top]

        self.surface[left][left_top] = previous[front][left_top]
        self.surface[left][right_top] = previous[front][right_top]

        self.surface[back][left_top] = previous[left][left_top]
        self.surface[back][right_top] = previous[left][right_top]

        self.surface[right][left_top] = previous[back][left_top]
        self.surface[right][right_top] = previous[back][right_top]

        self.surface[top][left_top] = previous[top][left_bottom]
        self.surface[top][right_top] = previous[top][left_top]
        self.surface[top][right_bottom] = previous[top][right_top]
        self.surface[top][left_bottom] = previous[top][right_bottom]

        del previous

        self.state = np.squeeze(np.reshape(self.surface,24))
        self.update_rubik()

    def twist_top_n(self, k):
        for _ in range(k):
            self.twist_top()

    def twist_front(self):
        previous = np.copy(self.surface)

        self.surface[right][left_top] = previous[top][left_bottom]
        self.surface[right][left_bottom] = previous[top][right_bottom]

        self.surface[top][left_bottom] = previous[left][right_bottom]
        self.surface[top][right_bottom] = previous[left][right_top]

        self.surface[left][right_top] = previous[bottom][left_top]
        self.surface[left][right_bottom] = previous[bottom][right_top]

        self.surface[bottom][left_top] = previous[right][left_bottom]
        self.surface[bottom][right_top] = previous[right][left_top]

        self.surface[front][left_top] = previous[front][left_bottom]
        self.surface[front][right_top] = previous[front][left_top]
        self.surface[front][right_bottom] = previous[front][right_top]
        self.surface[front][left_bottom] = previous[front][right_bottom]

        del previous

        self.state = np.squeeze(np.reshape(self.surface,24))
        self.update_rubik()

    def twist_front_n(self, k):
        for _ in range(k):
            self.twist_front()

    def update_rubik(self):

        # for s, e in combinations(self.points, 2):
        #     if np.sum(np.abs(s - e)) == 1:
        #         self.ax.plot3D(*zip(s, e), color="k")
        # # front
        # x = [-0, -0, -1, -1]
        # y = [-1, -1, -1, -1]
        # z = [0, 1, 1, 0]
        # verts = [zip(x, y, z)]
        # rect = Poly3DCollection(verts)
        # rect.set_color(self.surface[front][left_top])
        # self.ax.add_collection3d(rect)
        #
        # x = [0, 0, -1, -1]
        # y = [-1, -1, -1, -1]
        # z = [0, -1, -1, 0]
        # verts = [zip(x, y, z)]
        # rect = Poly3DCollection(verts)
        # rect.set_color(self.surface[front][left_bottom])
        # self.ax.add_collection3d(rect)
        #
        # x = [0, 0, 1, 1]
        # y = [-1, -1, -1, -1]
        # z = [0, 1, 1, 0]
        # verts = [zip(x, y, z)]
        # rect = Poly3DCollection(verts)
        # rect.set_color(self.surface[front][right_top])
        # self.ax.add_collection3d(rect)
        #
        # x = [0, 0, 1, 1]
        # y = [-1, -1, -1, -1]
        # z = [0, -1, -1, 0]
        # verts = [zip(x, y, z)]
        # rect = Poly3DCollection(verts)
        # rect.set_color(self.surface[front][right_bottom])
        # self.ax.add_collection3d(rect)
        #
        # # back
        #
        # x = [0, 0, -1, -1]
        # y = [1, 1, 1, 1]
        # z = [0, 1, 1, 0]
        # verts = [zip(x, y, z)]
        # rect = Poly3DCollection(verts)
        # rect.set_color(self.surface[back][right_top])
        # self.ax.add_collection3d(rect)
        #
        # x = [0, 0, -1, -1]
        # y = [1, 1, 1, 1]
        # z = [0, -1, -1, 0]
        # verts = [zip(x, y, z)]
        # rect = Poly3DCollection(verts)
        # rect.set_color(self.surface[back][right_bottom])
        # self.ax.add_collection3d(rect)
        #
        # x = [0, 0, 1, 1]
        # y = [1, 1, 1, 1]
        # z = [0, 1, 1, 0]
        # verts = [zip(x, y, z)]
        # rect = Poly3DCollection(verts)
        # rect.set_color(self.surface[back][left_top])
        # self.ax.add_collection3d(rect)
        #
        # x = [0, 0, 1, 1]
        # y = [1, 1, 1, 1]
        # z = [0, -1, -1, 0]
        # verts = [zip(x, y, z)]
        # rect = Poly3DCollection(verts)
        # rect.set_color(self.surface[back][left_bottom])
        # self.ax.add_collection3d(rect)
        #
        # # right
        # x = [1, 1, 1, 1]
        # y = [0, 0, -1, -1]
        # z = [0, 1, 1, 0]
        # verts = [zip(x, y, z)]
        # rect = Poly3DCollection(verts)
        # rect.set_color(self.surface[right][left_top])
        # self.ax.add_collection3d(rect)
        #
        # x = [1, 1, 1, 1]
        # y = [0, 0, 1, 1]
        # z = [0, 1, 1, 0]
        # verts = [zip(x, y, z)]
        # rect = Poly3DCollection(verts)
        # rect.set_color(self.surface[right][right_top])
        # self.ax.add_collection3d(rect)
        #
        # x = [1, 1, 1, 1]
        # y = [0, 0, -1, -1]
        # z = [0, -1, -1, 0]
        # verts = [zip(x, y, z)]
        # rect = Poly3DCollection(verts)
        # rect.set_color(self.surface[right][left_bottom])
        # self.ax.add_collection3d(rect)
        #
        # x = [1, 1, 1, 1]
        # y = [0, 0, 1, 1]
        # z = [0, -1, -1, 0]
        # verts = [zip(x, y, z)]
        # rect = Poly3DCollection(verts)
        # rect.set_color(self.surface[right][right_bottom])
        # self.ax.add_collection3d(rect)
        #
        # # left
        # x = [-1, -1, -1, -1]
        # y = [0, 0, -1, -1]
        # z = [0, 1, 1, 0]
        # verts = [zip(x, y, z)]
        # rect = Poly3DCollection(verts)
        # rect.set_color(self.surface[left][right_top])
        # self.ax.add_collection3d(rect)
        #
        # x = [-1, -1, -1, -1]
        # y = [0, 0, 1, 1]
        # z = [0, 1, 1, 0]
        # verts = [zip(x, y, z)]
        # rect = Poly3DCollection(verts)
        # rect.set_color(self.surface[left][left_top])
        # self.ax.add_collection3d(rect)
        #
        # x = [-1, -1, -1, -1]
        # y = [0, 0, -1, -1]
        # z = [0, -1, -1, 0]
        # verts = [zip(x, y, z)]
        # rect = Poly3DCollection(verts)
        # rect.set_color(self.surface[left][right_bottom])
        # self.ax.add_collection3d(rect)
        #
        # x = [-1, -1, -1, -1]
        # y = [0, 0, 1, 1]
        # z = [0, -1, -1, 0]
        # verts = [zip(x, y, z)]
        # rect = Poly3DCollection(verts)
        # rect.set_color(self.surface[left][left_bottom])
        # self.ax.add_collection3d(rect)
        #
        # # top
        # x = [0, -1, -1, 0]
        # y = [0, 0, 1, 1]
        # z = [1, 1, 1, 1]
        # verts = [zip(x, y, z)]
        # rect = Poly3DCollection(verts)
        # rect.set_color(self.surface[top][left_top])
        # self.ax.add_collection3d(rect)
        #
        # x = [0, 1, 1, 0]
        # y = [0, 0, 1, 1]
        # z = [1, 1, 1, 1]
        # verts = [zip(x, y, z)]
        # rect = Poly3DCollection(verts)
        # rect.set_color(self.surface[top][right_top])
        # self.ax.add_collection3d(rect)
        #
        # x = [0, 1, 1, 0]
        # y = [0, 0, -1, -1]
        # z = [1, 1, 1, 1]
        # verts = [zip(x, y, z)]
        # rect = Poly3DCollection(verts)
        # rect.set_color(self.surface[top][right_bottom])
        # self.ax.add_collection3d(rect)
        #
        # x = [0, -1, -1, 0]
        # y = [0, 0, -1, -1]
        # z = [1, 1, 1, 1]
        # verts = [zip(x, y, z)]
        # rect = Poly3DCollection(verts)
        # rect.set_color(self.surface[top][left_bottom])
        # self.ax.add_collection3d(rect)
        #
        # # bottom
        # x = [0, -1, -1, 0]
        # y = [0, 0, 1, 1]
        # z = [-1, -1, -1, -1]
        # verts = [zip(x, y, z)]
        # rect = Poly3DCollection(verts)
        # rect.set_color(self.surface[bottom][left_bottom])
        # self.ax.add_collection3d(rect)
        #
        # x = [0, 1, 1, 0]
        # y = [0, 0, 1, 1]
        # z = [-1, -1, -1, -1]
        # verts = [zip(x, y, z)]
        # rect = Poly3DCollection(verts)
        # rect.set_color(self.surface[bottom][right_bottom])
        # self.ax.add_collection3d(rect)
        #
        # x = [0, 1, 1, 0]
        # y = [0, 0, -1, -1]
        # z = [-1, -1, -1, -1]
        # verts = [zip(x, y, z)]
        # rect = Poly3DCollection(verts)
        # rect.set_color(self.surface[bottom][right_top])
        # self.ax.add_collection3d(rect)
        #
        # x = [0, -1, -1, 0]
        # y = [0, 0, -1, -1]
        # z = [-1, -1, -1, -1]
        # verts = [zip(x, y, z)]
        # rect = Poly3DCollection(verts)
        # rect.set_color(self.surface[bottom][left_top])
        # self.ax.add_collection3d(rect)
        #
        # self.fig.canvas.draw()
        return
def check(s):
    s = np.reshape(s,[6,4])
    for i in range(6):
        u = np.unique(s[i])
        if np.size(u) == 1:
            return True
    return False