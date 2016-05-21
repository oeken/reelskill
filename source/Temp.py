# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as st
import Model as md

class Point:
    counter = 101
    def __init__(self):
        self.id = Point.counter
        self.x=5
        self.y=5
        Point.counter += 1

l = []
for i in range(5):l.append(Point())
l[0].x = 10
print "selam"
