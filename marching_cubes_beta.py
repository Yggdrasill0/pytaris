import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import collections
from collections import defaultdict
sys.setrecursionlimit(int(1E8))
import os
import pyvista as pv

tri_table =[
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [9, 2, 10, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [2, 8, 3, 2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1, -1, -1],
            [3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [0, 11, 2, 8, 11, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [1, 9, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [1, 11, 2, 1, 9, 11, 9, 8, 11, -1, -1, -1, -1, -1, -1, -1],
            [3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [0, 10, 1, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1],
            [3, 9, 0, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1, -1, -1, -1],
            [9, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [4, 3, 0, 7, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [0, 1, 9, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [4, 1, 9, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1, -1],
            [1, 2, 10, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [3, 4, 7, 3, 0, 4, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1],
            [9, 2, 10, 9, 0, 2, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1],
            [2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1, -1, -1, -1],
            [8, 4, 7, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [11, 4, 7, 11, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1, -1],
            [9, 0, 1, 8, 4, 7, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1],
            [4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1, -1, -1, -1],
            [3, 10, 1, 3, 11, 10, 7, 8, 4, -1, -1, -1, -1, -1, -1, -1],
            [1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1, -1, -1, -1],
            [4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1, -1, -1, -1],
            [4, 7, 11, 4, 11, 9, 9, 11, 10, -1, -1, -1, -1, -1, -1, -1],
            [9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [9, 5, 4, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [0, 5, 4, 1, 5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [8, 5, 4, 8, 3, 5, 3, 1, 5, -1, -1, -1, -1, -1, -1, -1],
            [1, 2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [3, 0, 8, 1, 2, 10, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1],
            [5, 2, 10, 5, 4, 2, 4, 0, 2, -1, -1, -1, -1, -1, -1, -1],
            [2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1, -1, -1, -1],
            [9, 5, 4, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [0, 11, 2, 0, 8, 11, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1],
            [0, 5, 4, 0, 1, 5, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1],
            [2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1, -1, -1, -1],
            [10, 3, 11, 10, 1, 3, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1],
            [4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1, -1, -1, -1],
            [5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1, -1, -1, -1],
            [5, 4, 8, 5, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1],
            [9, 7, 8, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [9, 3, 0, 9, 5, 3, 5, 7, 3, -1, -1, -1, -1, -1, -1, -1],
            [0, 7, 8, 0, 1, 7, 1, 5, 7, -1, -1, -1, -1, -1, -1, -1],
            [1, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [9, 7, 8, 9, 5, 7, 10, 1, 2, -1, -1, -1, -1, -1, -1, -1],
            [10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1, -1, -1, -1],
            [8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, -1, -1, -1, -1],
            [2, 10, 5, 2, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1],
            [7, 9, 5, 7, 8, 9, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1],
            [9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1, -1, -1, -1],
            [2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1, -1, -1, -1],
            [11, 2, 1, 11, 1, 7, 7, 1, 5, -1, -1, -1, -1, -1, -1, -1],
            [9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1, -1, -1, -1],
            [5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, -1],
            [11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0, -1],
            [11, 10, 5, 7, 11, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [0, 8, 3, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [9, 0, 1, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [1, 8, 3, 1, 9, 8, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1],
            [1, 6, 5, 2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [1, 6, 5, 1, 2, 6, 3, 0, 8, -1, -1, -1, -1, -1, -1, -1],
            [9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1, -1],
            [5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1, -1, -1, -1],
            [2, 3, 11, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [11, 0, 8, 11, 2, 0, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1],
            [0, 1, 9, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1],
            [5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1, -1, -1, -1],
            [6, 3, 11, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1, -1],
            [0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1, -1, -1, -1],
            [3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1, -1, -1, -1],
            [6, 5, 9, 6, 9, 11, 11, 9, 8, -1, -1, -1, -1, -1, -1, -1],
            [5, 10, 6, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [4, 3, 0, 4, 7, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1],
            [1, 9, 0, 5, 10, 6, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1],
            [10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1, -1, -1, -1],
            [6, 1, 2, 6, 5, 1, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1],
            [1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1, -1, -1, -1],
            [8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1, -1, -1, -1],
            [7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, -1],
            [3, 11, 2, 7, 8, 4, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1],
            [5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1, -1, -1, -1],
            [0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1],
            [9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6, -1],
            [8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1, -1, -1, -1],
            [5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11, -1],
            [0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7, -1],
            [6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1, -1, -1, -1],
            [10, 4, 9, 6, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [4, 10, 6, 4, 9, 10, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1],
            [10, 0, 1, 10, 6, 0, 6, 4, 0, -1, -1, -1, -1, -1, -1, -1],
            [8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, -1, -1, -1, -1],
            [1, 4, 9, 1, 2, 4, 2, 6, 4, -1, -1, -1, -1, -1, -1, -1],
            [3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1, -1, -1, -1],
            [0, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [8, 3, 2, 8, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1],
            [10, 4, 9, 10, 6, 4, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1],
            [0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1, -1, -1, -1],
            [3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1, -1, -1, -1],
            [6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1, -1],
            [9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1, -1, -1, -1],
            [8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1, -1],
            [3, 11, 6, 3, 6, 0, 0, 6, 4, -1, -1, -1, -1, -1, -1, -1],
            [6, 4, 8, 11, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [7, 10, 6, 7, 8, 10, 8, 9, 10, -1, -1, -1, -1, -1, -1, -1],
            [0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, -1, -1, -1, -1],
            [10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, -1, -1, -1, -1],
            [10, 6, 7, 10, 7, 1, 1, 7, 3, -1, -1, -1, -1, -1, -1, -1],
            [1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1, -1, -1, -1],
            [2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9, -1],
            [7, 8, 0, 7, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1, -1, -1],
            [7, 3, 2, 6, 7, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1, -1, -1, -1],
            [2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7, -1],
            [1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, -1],
            [11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1, -1, -1, -1],
            [8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6, -1],
            [0, 9, 1, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, -1, -1, -1, -1],
            [7, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [3, 0, 8, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [0, 1, 9, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [8, 1, 9, 8, 3, 1, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1],
            [10, 1, 2, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [1, 2, 10, 3, 0, 8, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1],
            [2, 9, 0, 2, 10, 9, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1],
            [6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, -1, -1, -1, -1],
            [7, 2, 3, 6, 2, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [7, 0, 8, 7, 6, 0, 6, 2, 0, -1, -1, -1, -1, -1, -1, -1],
            [2, 7, 6, 2, 3, 7, 0, 1, 9, -1, -1, -1, -1, -1, -1, -1],
            [1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1, -1, -1, -1],
            [10, 7, 6, 10, 1, 7, 1, 3, 7, -1, -1, -1, -1, -1, -1, -1],
            [10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, -1, -1, -1, -1],
            [0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, -1, -1, -1, -1],
            [7, 6, 10, 7, 10, 8, 8, 10, 9, -1, -1, -1, -1, -1, -1, -1],
            [6, 8, 4, 11, 8, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [3, 6, 11, 3, 0, 6, 0, 4, 6, -1, -1, -1, -1, -1, -1, -1],
            [8, 6, 11, 8, 4, 6, 9, 0, 1, -1, -1, -1, -1, -1, -1, -1],
            [9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1, -1, -1, -1],
            [6, 8, 4, 6, 11, 8, 2, 10, 1, -1, -1, -1, -1, -1, -1, -1],
            [1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1, -1, -1, -1],
            [4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1, -1, -1, -1],
            [10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, -1],
            [8, 2, 3, 8, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1],
            [0, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, -1, -1, -1, -1],
            [1, 9, 4, 1, 4, 2, 2, 4, 6, -1, -1, -1, -1, -1, -1, -1],
            [8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, -1, -1, -1, -1],
            [10, 1, 0, 10, 0, 6, 6, 0, 4, -1, -1, -1, -1, -1, -1, -1],
            [4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3, -1],
            [10, 9, 4, 6, 10, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [4, 9, 5, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [0, 8, 3, 4, 9, 5, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1],
            [5, 0, 1, 5, 4, 0, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1],
            [11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, -1, -1, -1, -1],
            [9, 5, 4, 10, 1, 2, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1],
            [6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, -1, -1, -1, -1],
            [7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1, -1, -1, -1],
            [3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6, -1],
            [7, 2, 3, 7, 6, 2, 5, 4, 9, -1, -1, -1, -1, -1, -1, -1],
            [9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, -1, -1, -1, -1],
            [3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1, -1, -1, -1],
            [6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8, -1],
            [9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, -1, -1, -1, -1],
            [1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4, -1],
            [4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10, -1],
            [7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1, -1, -1, -1],
            [6, 9, 5, 6, 11, 9, 11, 8, 9, -1, -1, -1, -1, -1, -1, -1],
            [3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1, -1, -1, -1],
            [0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, -1, -1, -1, -1],
            [6, 11, 3, 6, 3, 5, 5, 3, 1, -1, -1, -1, -1, -1, -1, -1],
            [1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1, -1, -1, -1],
            [0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10, -1],
            [11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, -1],
            [6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, -1, -1, -1, -1],
            [5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1, -1, -1, -1],
            [9, 5, 6, 9, 6, 0, 0, 6, 2, -1, -1, -1, -1, -1, -1, -1],
            [1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8, -1],
            [1, 5, 6, 2, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, -1],
            [10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, -1, -1, -1, -1],
            [0, 3, 8, 5, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [10, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [11, 5, 10, 7, 5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [11, 5, 10, 11, 7, 5, 8, 3, 0, -1, -1, -1, -1, -1, -1, -1],
            [5, 11, 7, 5, 10, 11, 1, 9, 0, -1, -1, -1, -1, -1, -1, -1],
            [10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, -1, -1, -1, -1],
            [11, 1, 2, 11, 7, 1, 7, 5, 1, -1, -1, -1, -1, -1, -1, -1],
            [0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, -1, -1, -1, -1],
            [9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, -1, -1, -1, -1],
            [7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2, -1],
            [2, 5, 10, 2, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1],
            [8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, -1, -1, -1, -1],
            [9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, -1, -1, -1, -1],
            [9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2, -1],
            [1, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [0, 8, 7, 0, 7, 1, 1, 7, 5, -1, -1, -1, -1, -1, -1, -1],
            [9, 0, 3, 9, 3, 5, 5, 3, 7, -1, -1, -1, -1, -1, -1, -1],
            [9, 8, 7, 5, 9, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [5, 8, 4, 5, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1],
            [5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, -1, -1, -1, -1],
            [0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, -1, -1, -1, -1],
            [10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4, -1],
            [2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, -1, -1, -1, -1],
            [0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11, -1],
            [0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5, -1],
            [9, 4, 5, 2, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1, -1, -1, -1],
            [5, 10, 2, 5, 2, 4, 4, 2, 0, -1, -1, -1, -1, -1, -1, -1],
            [3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9, -1],
            [5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, -1, -1, -1, -1],
            [8, 4, 5, 8, 5, 3, 3, 5, 1, -1, -1, -1, -1, -1, -1, -1],
            [0, 4, 5, 1, 0, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, -1, -1, -1, -1],
            [9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [4, 11, 7, 4, 9, 11, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1],
            [0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, -1, -1, -1, -1],
            [1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, -1, -1, -1, -1],
            [3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4, -1],
            [4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, -1, -1, -1, -1],
            [9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3, -1],
            [11, 7, 4, 11, 4, 2, 2, 4, 0, -1, -1, -1, -1, -1, -1, -1],
            [11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, -1, -1, -1, -1],
            [2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1, -1, -1, -1],
            [9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7, -1],
            [3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10, -1],
            [1, 10, 2, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [4, 9, 1, 4, 1, 7, 7, 1, 3, -1, -1, -1, -1, -1, -1, -1],
            [4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, -1, -1, -1, -1],
            [4, 0, 3, 7, 4, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [9, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [3, 0, 9, 3, 9, 11, 11, 9, 10, -1, -1, -1, -1, -1, -1, -1],
            [0, 1, 10, 0, 10, 8, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1],
            [3, 1, 10, 11, 3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [1, 2, 11, 1, 11, 9, 9, 11, 8, -1, -1, -1, -1, -1, -1, -1],
            [3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, -1, -1, -1, -1],
            [0, 2, 11, 8, 0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [3, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [2, 3, 8, 2, 8, 10, 10, 8, 9, -1, -1, -1, -1, -1, -1, -1],
            [9, 10, 2, 0, 9, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, -1, -1, -1, -1],
            [1, 10, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [1, 3, 8, 9, 1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]]

def interpolation_alpha_value(v1, v2, t):
  """Calculates the alpha value for the linear interpolation, this is given
  by the equation a_r = (t - v1)/(v2 -v1)"""
  ###Agregar aca esa notacion chevere del docstring que he visto por ahi

  if v1 == v2 and t == v1:
    return 0
  elif t > v1 and t > v2:
    return None
  elif t < v1 and t < v2:
    return None
  #Case when we have a bipolar grid edge
  else:
    return (v1 - t) / (v1 - v2)

cache={}
def linear_interpolation(edge,cells,top,left,depth,thres):
  """Hacer un mejor docstring"""


  tval = 0
  point = None

  #Edge 0 case
  if (edge == 0):
    #If the point is already in the cache set, return it
    if (((left,top,depth),(left+1,top,depth)) in cache):
      point = cache[((left,top,depth),(left+1,top,depth))]

    #If it's not in cache, calculate it using linear interpolation
    else:
      tval = interpolation_alpha_value(cells[left,top,depth],cells[left+1,top,depth],thres)
      if (tval is None):
        return None
      point = (left+tval,top,depth)
      cache[((left,top,depth),(left+1,top,depth))] = point

    return point

  #Edge 1 case
  if (edge == 1):
    if (((left+1,top,depth),(left+1,top+1,depth)) in cache):
      point = cache[((left+1,top,depth),(left+1,top+1,depth))]

    else:
      tval = interpolation_alpha_value(cells[left+1,top,depth],cells[left+1,top+1,depth],thres)
      if (tval is None):
        return None
      point = (left+1,top+tval,depth)
      cache[((left+1,top,depth),(left+1,top+1,depth))] = point

    return point

  #Edge 2 case
  if (edge == 2):
    if (((left,top+1,depth),(left+1,top+1,depth)) in cache):
      point = cache[((left,top+1,depth),(left+1,top+1,depth))]

    else:
      tval = interpolation_alpha_value(cells[left,top+1,depth],cells[left+1,top+1,depth],thres)
      if (tval is None):
        return None
      point = (left+tval,top+1,depth)
      cache[((left,top+1,depth),(left+1,top+1,depth))] = point

    return point

  #Edge 3 case
  if (edge == 3):
    if (((left,top,depth),(left,top+1,depth)) in cache):
      point = cache[((left,top,depth),(left,top+1,depth))]

    else:
      tval = interpolation_alpha_value(cells[left,top,depth],cells[left,top+1,depth],thres)
      if (tval is None):
        return None
      point = (left,top+tval,depth)
      cache[((left,top,depth),(left,top+1,depth))] = point
    return point

  #Edge 4 case
  if (edge == 4):
    if (((left,top,depth+1),(left+1,top,depth+1)) in cache):
      point = cache[((left,top,depth+1),(left+1,top,depth+1))]

    else:
      tval = interpolation_alpha_value(cells[left,top,depth+1],cells[left+1,top,depth+1],thres)
      if (tval is None):
        return None
      point = (left+tval,top,depth+1)
      cache[((left,top,depth+1),(left+1,top,depth+1))] = point

    return point

  #Edge 5 case
  if (edge == 5):
    if (((left+1,top,depth+1),(left+1,top+1,depth+1)) in cache):
      point = cache[((left+1,top,depth+1),(left+1,top+1,depth+1))]

    else:
      tval = interpolation_alpha_value(cells[left+1,top,depth+1],cells[left+1,top+1,depth+1],thres)
      if (tval is None):
        return None
      point = (left+1,top+tval,depth+1)
      cache[((left+1,top,depth+1),(left+1,top+1,depth+1))] = point

    return point

  #Edge 6 case
  if (edge == 6):
    if (((left,top+1,depth+1),(left+1,top+1,depth+1)) in cache):
      point = cache[((left,top+1,depth+1),(left+1,top+1,depth+1))]

    else:
      tval = interpolation_alpha_value(cells[left,top+1,depth+1],cells[left+1,top+1,depth+1],thres)
      if (tval is None):
        return None
      point = (left+tval,top+1,depth+1)
      cache[((left,top+1,depth+1),(left+1,top+1,depth+1))] = point

    return point

  #Edge 7 case
  if (edge == 7):
    if (((left,top,depth+1),(left,top+1,depth+1)) in cache):
      point = cache[((left,top,depth+1),(left,top+1,depth+1))]

    else:
      tval = interpolation_alpha_value(cells[left,top,depth+1],cells[left,top+1,depth+1],thres)
      if (tval is None):
        return None
      point = (left,top+tval,depth+1)
      cache[((left,top,depth+1),(left,top+1,depth+1))] = point

    return point

  #Edge 8 case
  if (edge == 8):
    if (((left,top,depth),(left,top,depth+1)) in cache):
      point = cache[((left,top,depth),(left,top,depth+1))]

    else:
      tval = interpolation_alpha_value(cells[left,top,depth],cells[left,top,depth+1],thres)
      if (tval is None):
        return None
      point = (left,top,depth+tval)
      cache[((left,top,depth),(left,top,depth+1))] = point

    return point

  #Edge 9 case
  if (edge == 9):
    if (((left+1,top,depth),(left+1,top,depth+1)) in cache):
      point = cache[((left+1,top,depth),(left+1,top,depth+1))]

    else:
      tval = interpolation_alpha_value(cells[left+1,top,depth],cells[left+1,top,depth+1],thres)
      if (tval is None):
        return None
      point = (left+1,top,depth+tval)
      cache[((left+1,top,depth),(left+1,top,depth+1))] = point

    return point

  #Edge 10 case
  if (edge == 10):
    if (((left+1,top+1,depth),(left+1,top+1,depth+1)) in cache):
      point = cache[((left+1,top+1,depth),(left+1,top+1,depth+1))]

    else:
      tval = interpolation_alpha_value(cells[left+1,top+1,depth],cells[left+1,top+1,depth+1],thres)
      if (tval is None):
        return None
      point = (left+1,top+1,depth+tval)
      cache[((left+1,top+1,depth),(left+1,top+1,depth+1))] = point

    return point

  #Edge 11 case
  if (edge == 11):
    if (((left,top+1,depth),(left,top+1,depth+1)) in cache):
      point = cache[((left,top+1,depth),(left,top+1,depth+1))]
    else:
      tval = interpolation_alpha_value(cells[left,top+1,depth],cells[left,top+1,depth+1],thres)
      if (tval is None):
        return None
      point = (left,top+1,depth+tval)
      cache[((left,top+1,depth),(left,top+1,depth+1))] = point

    return point

def getContourCase(top,left,depth, thres,cells):
  """Returns the decimal number of the eight bit number representing
  the values of each grid vertex compared to the threshold value"""

  x = 0
  if (thres < cells[left,top+1,depth+1]):
    x = 128
  if (thres < cells[left+1,top+1,depth+1]):
    x = x + 64
  if (thres < cells[left+1,top,depth+1]):
    x = x + 32
  if (thres < cells[left,top,depth+1]):
    x = x + 16
  if (thres < cells[left,top+1,depth]):
    x = x + 8
  if (thres < cells[left+1,top+1,depth]):
    x = x + 4
  if (thres < cells[left+1,top,depth]):
    x = x + 2
  if (thres < cells[left,top,depth]):
    x = x + 1
  case_value = tri_table[x]

  return case_value

def getContourSegments(thres,cells):
  """Hacer un mejor docstring"""

  rows = cells.shape[0]
  cols = cells.shape[1]
  zcols  = cells.shape[2]
  vertex_counter = 0
  vertex_array = collections.OrderedDict()
  face_array = []
  t1 = time.time()

  for left in range(0, rows-1):
    for top in range(0, cols-1):
      for depth in range(0, zcols-1):
        case_val = getContourCase(top,left,depth,thres,cells)
        k = 0 #que es k
        while (case_val[k] != -1):
          v1 = linear_interpolation(case_val[k],cells,top,left,depth,thres)

          if v1 is None:
            k += 3
            continue
          v2  = linear_interpolation(case_val [k+1],cells,top,left,depth,thres)

          if v2 is None:
            k  += 3
            continue
          v3 =  linear_interpolation(case_val [k+2],cells,top,left,depth,thres)

          if v3 is None:
            k += 3
            continue

          k += 3
          tmp = [3, 0, 0, 0]
          if v1 not in vertex_array:
            vertex_array[v1] = [vertex_counter, v1[0], v1[1], v1[2]]
            tmp[1] = vertex_counter
            vertex_counter += 1

          else:
            tmp[1] = vertex_array[v1][0]
            if v2 not in vertex_array:
              vertex_array[v2] = [vertex_counter, v2[0], v2[1], v2[2]]
              tmp[2] = vertex_counter
              vertex_counter += 1

            else:
              tmp[2] = vertex_array[v2][0]
              if v3 not in vertex_array:
                vertex_array[v3] = [vertex_counter, v3[0], v3[1], v3[2]]
                tmp[3] = vertex_counter
                vertex_counter += 1
              else:
                tmp[3] = vertex_array[v3][0]
                face_array.append(tmp)

  t2 = time.time()
  print("\nTime taken by algorithm\n"+'-'*40+"\n{} s".format(t2-t1))
  vertex_array = np.array(list(vertex_array.values()))

  return vertex_array[:,1:], np.array(face_array)

def obtain_coordinates(dir, neg):
  with open(dir, "r") as cubo_potencial:
    lineas_cubo_potencial = cubo_potencial.readlines()

  valores_iniciales_coordenadas = list(map(float, lineas_cubo_potencial[2].split()))
  num_atomos, x_inicial, y_inicial, z_inicial = map(float, valores_iniciales_coordenadas[:1] + valores_iniciales_coordenadas[1:4])

  coordenada_x = list(map(float, lineas_cubo_potencial[3].split()))
  puntos_x, incremento_x = map(float, coordenada_x[:1] + coordenada_x[1:2])

  coordenada_y = list(map(float, lineas_cubo_potencial[4].split()))
  puntos_y, incremento_y = map(float, coordenada_y[:1] + coordenada_y[2:3])

  coordenada_z = list(map(float, lineas_cubo_potencial[5].split()))
  puntos_z, incremento_z = map(float, coordenada_z[:1] + coordenada_z[3:4])

  valores_pem = lineas_cubo_potencial[6+int(num_atomos):]
  valores_pem_f = np.array([float(valor) for conj_valores in valores_pem for valor in conj_valores.split()])

  min_value = np.min(valores_pem_f)
  valores_pem_f = valores_pem_f.reshape((int(puntos_x), int(puntos_y), int(puntos_z)))

  if neg:
    return valores_pem_f, incremento_x , min_value
  else:
    return valores_pem_f, incremento_x

# Represent the faces and vertices as a graph
def generate_graph(faces):
  graph = defaultdict(set)
  for face in faces[:,1:]:
    for i in range(3):
      graph[face[i]].add(face[(i+1)%3])
      graph[face[i]].add(face[(i+2)%3])

  return graph

#Implement the DFS algorithm
def dfs(vertex, visited, component, graph):
  visited.add(vertex)
  component.append(vertex)
  for neighbor in graph[vertex]:
    if neighbor not in visited:
      dfs(neighbor, visited, component, graph)

#Find connected components
def find_connected_components(graph):
  visited = set()
  components = []
  for vertex in graph:
    if vertex not in visited:
      component = []
      dfs(vertex, visited, component, graph)
      components.append(component)

  return components



def triangles_in_component(vertices, faces, component):

  if component.size ==0:
    pass
  else:
    # Create a boolean mask for faces containing any element from the component
    mask = np.all(np.isin(faces[:, 1:], list(component)), axis=1)
    filtered_faces = faces[mask][:,1:]
    triangles = vertices[filtered_faces]

    return triangles

def isosurface_area(triangles, spacing):

  return (np.sum(0.5*abs(np.linalg.det(triangles[:]))))*(spacing**2)

def isosurface_barycenter(vertices, component):
  vert_triangles = vertices[component]
  baricenter = np.array([np.mean(vert_triangles[:,0]), np.mean(vert_triangles[:,1]), np.mean(vert_triangles[:,2])])

  return baricenter

def project_xy(triangles, barycenter_old):
  if ((np.max(triangles[:,:,-1]) > barycenter_old[-1]) and (np.min(triangles[:,:,-1]) < barycenter_old[-1])):
    barycenters_current_surface = np.mean(triangles, axis=1)
    masks = (barycenters_current_surface[:,-1] > barycenter_old[-1])
    filtered_triangles = triangles[masks]
    points_2d = filtered_triangles[:, :, :2].reshape(-1, 2)
    unique_points_2d = np.round(np.unique(points_2d, axis=0), 5)
    points_3d = np.hstack((unique_points_2d, np.zeros((unique_points_2d.shape[0], 1))))
    xy_point_cloud = pv.PolyData(points_3d)
    xy_point_surf = xy_point_cloud.delaunay_2d()
    xy_point_surf = xy_point_surf.clean(point_merging=False)
    points = xy_point_surf.points[:,:-1]
    faces = xy_point_surf.faces.reshape(-1, 4)[:, 1:]

    return points[faces]
  else:
    return np.array([])


def triangle_area(triangle):
  return np.round(0.5 * np.abs(
      triangle[1, 0] * triangle[2, 1] - triangle[2, 0] * triangle[1, 1] -
      triangle[0, 0] * triangle[2, 1] + triangle[2, 0] * triangle[0, 1] +
      triangle[0, 0] * triangle[1, 1] - triangle[1, 0] * triangle[0, 1]),5)

def check_containment(projected_triangles, barycenter_old):
  if projected_triangles.size == 0:
    return False
  else:
    barycenter_old_trimmed = barycenter_old[:-1]

    triangles_1 = projected_triangles.copy()
    triangles_1[:, 0] = barycenter_old_trimmed

    triangles_2 = projected_triangles.copy()
    triangles_2[:, 1] = barycenter_old_trimmed

    triangles_3 = projected_triangles.copy()
    triangles_3[:, 2] = barycenter_old_trimmed

    area_original = np.array([triangle_area(triangle) for triangle in projected_triangles])
    area_1 = np.array([triangle_area(triangle) for triangle in triangles_1])
    area_2 = np.array([triangle_area(triangle) for triangle in triangles_2])
    area_3 = np.array([triangle_area(triangle) for triangle in triangles_3])
    area_barycenter = area_1 + area_2 + area_3

    is_contained = np.any(np.abs(area_original - area_barycenter) <= 2E-5)

    return is_contained

def obtain_values_neg(min_value, step= 0.002):
  P_i = np.ceil(min_value/step) * step
  P_f = -0.009
  return np.round(np.arange(P_i, P_f, step), 3)

def obtain_values_pos(step = -0.05):
  P_i = 1.500
  P_f = 0.009
  return np.round(np.arange(P_i, P_f, step), 3)

def Pytaris(values_isopotential, valores_pem_f, incremento_x, tol=4):
  dicc_pytaris = {}
  barycenters_old = np.empty((0, 3))
  if len(values_isopotential) > 0: #esto lo cambie
    conec =np.array([values_isopotential[0], -1, -1])
  else:
    conec = np.array(())

  for value_isopotential in values_isopotential:
    print(f"El valor de potencial es {value_isopotential}")
    barycenters_new = np.empty((0, 3))
    try:
      verts, faces = getContourSegments(value_isopotential, valores_pem_f)
    except IndexError:
      continue

    dicc_pytaris[value_isopotential] = np.array(())
    graph = generate_graph(faces)
    connected_components_new = find_connected_components(graph)

    filtered_components = []

    for component_new in connected_components_new:
      component_new = np.array((component_new))
      triangles_new_component = triangles_in_component(verts, faces, component_new)
      area = isosurface_area(triangles_new_component, incremento_x)

      if area >= tol:
        filtered_components.append((component_new, area))


    for index_new, (component_new, area) in enumerate(filtered_components):
      component_new = np.array((component_new))
      triangles_new_component = triangles_in_component(verts, faces, component_new)
      barycenter = isosurface_barycenter(verts, component_new)
      barycenters_new = np.append(barycenters_new, [barycenter], axis=0)
      barycenter = np.append([value_isopotential, area], barycenter)
      dicc_pytaris[value_isopotential] = np.append(dicc_pytaris[value_isopotential], barycenter)

      if barycenters_old.size != 0:
        for index_old, barycenter_old in enumerate(barycenters_old):
          projected_points  = project_xy(triangles_new_component, barycenter_old)
          containment = check_containment(projected_points, barycenter_old)
          if containment:
            conec = np.append(conec, [value_isopotential, index_old, index_new])

    barycenters_old = barycenters_new

  return conec, dicc_pytaris

def preprocess_neg(dicc_pytaris):
  llaves_completas = np.array(sorted(dicc_pytaris.keys()))
  llaves_completas = np.append(llaves_completas, np.array((0)))
  print(llaves_completas)
  info_nodos = np.array(())
  for info_nodo in dicc_pytaris.values():
    info_nodos = np.append(info_nodos, info_nodo)

  info_nodos = info_nodos.reshape(-1,5)

  info_nodos = np.vstack((info_nodos, np.array([0, 0, 0, 0, 0 ])))
  num_nodos = info_nodos.shape[0]

  return llaves_completas, info_nodos, num_nodos

def preprocess_pos(dicc_pytaris):
  llaves_completas = np.array(sorted(dicc_pytaris.keys(),reverse=True))
  llaves_completas = np.append(llaves_completas, np.array((0)))
  print(llaves_completas)
  info_nodos = np.array(())
  for info_nodo in dicc_pytaris.values():
    info_nodos = np.append(info_nodos, info_nodo)

  info_nodos = info_nodos.reshape(-1,5)

  info_nodos = np.vstack((info_nodos, np.array([0, 0, 0, 0, 0 ])))
  num_nodos = info_nodos.shape[0]

  return llaves_completas, info_nodos, num_nodos

def etiquetado_nodo_izquierda(nodos_peso, llaves_seleccionadas):
  """Función que me permite etiquetar los nodos de que comparten el mismo potencial
  de izquierda a derecha"""


  pesos_etiquetados = np.array(())

  for i in range(len(llaves_seleccionadas)):
    nodos_seleccionados = np.array(([nodo_seleccionado for nodo_seleccionado in nodos_peso if nodo_seleccionado[0] == llaves_seleccionadas[i]]))

    if len(nodos_seleccionados) != 0:
      indices_minimos = np.lexsort((nodos_seleccionados[:, 2], nodos_seleccionados[:, 3], nodos_seleccionados[:, 4]))
      nodos_orden_izquierdo = np.hstack((nodos_seleccionados, np.arange(len(nodos_seleccionados)).reshape(-1, 1)))

      for i in indices_minimos:
        nodos_orden_izquierdo[indices_minimos[i], -1] = i

      for i in nodos_orden_izquierdo:
        pesos_etiquetados = np.append(pesos_etiquetados, i)

  return pesos_etiquetados.reshape(-1,6)

def obtain_pre_edges(info_nodos, conec, dicc_pytaris, llaves_completas, num_nodos):
  info_nodos = info_nodos.tolist()
  vertices = conec.reshape(-1,3)[1:]
  lista_aristas = []
  llave = np.array(vertices[:,0], dtype="float64")
  superficies = np.array(vertices[:,1:], dtype="int")
  for i in range(len(llave)):
    try:
      previous_potential = llaves_completas[(list(llaves_completas).index(llave[i]))-1]
      actual_potential = llaves_completas[(list(llaves_completas).index(llave[i]))]
      a = list(dicc_pytaris[previous_potential][5*superficies[i][0]: (5*(superficies[i][0] +1))])
      b = list(dicc_pytaris[actual_potential][5*superficies[i][1]: (5*(superficies[i][1] +1))])

      if a and b:
        a_int = (info_nodos).index(a)
        b_int = (info_nodos).index(b)
        lista_aristas.append(tuple((a_int, b_int)))
    except:
      pass
  if len(dicc_pytaris) > 0:
    for j in range(len(dicc_pytaris[llaves_completas[-2]])//5):
      c = tuple((num_nodos-2-j, num_nodos-1))
      lista_aristas.append(c)

  return lista_aristas


def obtain_edges(lista_aristas, info_nodos_1):
  aristas_numpy = np.array(lista_aristas)
  unique_values, counts = np.unique(aristas_numpy[:,0], return_counts=True)
  if unique_values[counts > 1].size != 0:
    try:
      values_more_than_once = unique_values[counts > 1].reshape(2,1)
    except:
      values_more_than_once = unique_values[counts > 1]
    for node in values_more_than_once:
      more_than_one = aristas_numpy[aristas_numpy[:, 0] == node]
      try:
        values_xyz_reference = info_nodos_1[node][:,2:5]
      except:
        values_xyz_reference = info_nodos_1[node][2:5]
      xyz_difference = info_nodos_1[more_than_one[:,1]][:,2:5] - values_xyz_reference
      xyz_distances = np.sqrt(np.sum(xyz_difference**2, axis=1))#

      min_index = np.argmin(xyz_distances)
      row_to_eliminate = more_than_one[~min_index]
      aristas_numpy = aristas_numpy[~np.all(aristas_numpy == row_to_eliminate , axis=1)]

  lista_aristas = aristas_numpy.tolist()
  return lista_aristas

def create_file(nom_arch, num_nodos, info_nodos_1, lista_aristas):
  with open(nom_arch, "a") as fo:
    fo.write("graph [ \n")
    fo.write("directed 1 \n")
    for i in range(num_nodos):
      fo.write("node [ \n")
      fo.write(f"id {i} \n")
      fo.write(f'label "{i}" \n')
      fo.write("weight [ \n")
      fo.write(f"externalId {i} \n")
      fo.write(f"potentialValue {info_nodos_1[i][0]} \n")
      fo.write(f"areaValue {info_nodos_1[i][1]} \n")
      fo.write(f"xValue {info_nodos_1[i][2]} \n")
      fo.write(f"yValue {info_nodos_1[i][3]} \n")
      fo.write(f"zValue {info_nodos_1[i][4]} \n")
      fo.write(f"izquierda {int(info_nodos_1[i][5])} \n")
      fo.write("] \n")
      fo.write("\t ] \n")

    for arista in lista_aristas:
      fo.write("edge [ \n")
      fo.write(f"source {arista[0]} \n")
      fo.write(f"target {arista[1]} \n")
      fo.write("\t ] \n")
    fo.write("tree 1 \n")
    fo.write("] \n")

master_directory = r"/home/ricardomr/Desktop/GQT_Database/" #cambiar esto
fold_names = os.listdir(master_directory)
fold_names_sorted = list(map(str,sorted(list(map(int, fold_names)))))
file_count = 0

for file in fold_names_sorted:
  print(file_count)

    
  current_file = os.path.join(master_directory, file)
  molecule_files = os.listdir(current_file)
  file_count += 1
  cube_file = [molecule_file for molecule_file in molecule_files if molecule_file.lower().endswith('.cube')]
  gml_files = [molecule_file for molecule_file in molecule_files if molecule_file.lower().endswith('.gml')]

  
  if len(cube_file) == 1 and len(gml_files) == 0:
    print(f"HACIENDO CUBO {cube_file[0]}")
    
    Neg_cube = True
    dir = os.path.join(current_file,cube_file[0])   
    print(dir)             
    
    valores_pem, incremento_x, min_value = obtain_coordinates(dir, Neg_cube)
    values_isopotential = obtain_values_neg(min_value)
    conec, dicc_pytaris = Pytaris(values_isopotential, valores_pem, incremento_x)
    llaves_completas, info_nodos, num_nodos = preprocess_neg(dicc_pytaris)
    info_nodos_1 = etiquetado_nodo_izquierda(info_nodos, llaves_completas)
    pre_edges = obtain_pre_edges(info_nodos, conec, dicc_pytaris, llaves_completas, num_nodos)
    aristas = obtain_edges(pre_edges, info_nodos_1)
    NOM_ARCHIVO = cube_file[0][:-5] + "_neg.gml"
    NOM_ARCHIVO = os.path.join(current_file,NOM_ARCHIVO)
    create_file(NOM_ARCHIVO, num_nodos, info_nodos_1,aristas)
            
    Neg_cube = False
            
    valores_pem, incremento_x = obtain_coordinates(dir, Neg_cube)
    values_isopotential = obtain_values_pos()
    conec, dicc_pytaris = Pytaris(values_isopotential, valores_pem, incremento_x, tol=15)
    llaves_completas, info_nodos, num_nodos = preprocess_pos(dicc_pytaris)
    info_nodos_1 = etiquetado_nodo_izquierda(info_nodos, llaves_completas)
    pre_edges = obtain_pre_edges(info_nodos, conec, dicc_pytaris, llaves_completas, num_nodos)
    print(pre_edges)
    aristas = obtain_edges(pre_edges, info_nodos_1)
    NOM_ARCHIVO = basename + "_pos.gml"
    create_file(NOM_ARCHIVO, num_nodos, info_nodos_1, aristas)
