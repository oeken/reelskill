# -*- coding: utf-8 -*-
import numpy as np
import source.ep as nd
import source.model as md
import source.factory as df
import source.builder
import source.reader

np.random.seed(12345)
### TEST #1
### =======
# p1 = md.Player(reel_skill=5)
# p2 = md.Player(reel_skill=15)
# p3 = md.Player(reel_skill=25)
# p4 = md.Player(reel_skill=35)
# p5 = md.Player(reel_skill=45)
# p = [p1,p2,p3,p4,p5]
#
# t1 = md.Team([p1])
# t2 = md.Team([p2])
# t3 = md.Team([p3])
# t4 = md.Team([p4])
# t5 = md.Team([p5])
# t = [t1,t2,t3,t4,t5]
# md.draw_factor = 0.0
# m = df.generateSyntheticMatchesFullTimes(t,10)

### TEST #3
### =======
data_training = source.reader.read_data('../data/tennis/ausopen.csv', '../data/tennis/rg.csv')
# data_test = Reader.read_data('../data/tennis/wimbledon.csv','../data/tennis/usopen.csv')
p,t,m = source.reader.form_objects(data_training)



# p1 = md.Player(reel_skill=45)
# p1.ts_mu = 25
# p2 = md.Player(reel_skill=10)
# p2.ts_mu = 25
# p = [p1,p2]
#
# t1 = md.Team([p1])
# t2 = md.Team([p2])
# t = [t1,t2]
#
# md.draw_factor = 0.0
# m = df.generateSyntheticMatchesFullTimes(t,100)

for match in m:
    fg = source.builder.build_factor_graph(match)
    [res1, res2]= source.builder.execute_order(fg)
    wt = match.t1 if match.r == 1 or match.r == 0 else match.t2
    lt = match.t2 if match.r == 1 or match.r == 0 else match.t1
    for r, pl in zip(res1, wt.players):
        pl.ts_mu = r.value.mu
        pl.ts_sigma = r.value.sigma
    for r, pl in zip(res2, lt.players):
        pl.ts_mu = r.value.mu
        pl.ts_sigma = r.value.sigma


print 'selam'
