# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import Model as md

a = 5
def fn1():
    print a
    print 'selam tirrih'

def fn2():
    global a
    a += 1

fn2()
fn2()
print a
