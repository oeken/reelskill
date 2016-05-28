# -*- coding: utf-8 -*-
import numpy as np
import nodes as nd
import Model as md
import DataFactory as df
import builder
import trueskill

np.random.seed(12345)

p1 = md.Player(reel_skill=40)
p2 = md.Player(reel_skill=5)

p1.ts_mu = 25.0
p1.ts_sigma = 25.0/3

p2.ts_mu = 30.0
p2.ts_sigma = 25.0/3

p = [p1,p2]

t1 = md.Team([p1])
t2 = md.Team([p2])
t = [t1,t2]

md.draw_factor = 0.0
m = df.generateSyntheticMatchesFullTimes(t,1)



fg = builder.build_factor_graph(m[0])
out = builder.execute_order(fg)