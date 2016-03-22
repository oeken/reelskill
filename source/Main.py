# -*- coding: utf-8 -*-

import DataFactory as df
import Model as md
import FG as fg

#players, teams, matches = df.generateSyntheticData(10)


p1 = md.Player(reel_skill=30)
p2 = md.Player(reel_skill=9)
p3 = md.Player(reel_skill=12)
# p4 = md.Player(reel_skill=20)
#
t1 = md.Team([p1])
t2 = md.Team([p2,p3])
# t3 = md.Team([p4])
#
#
m1 = md.Match({t1:1,t2:2}) # synthetic match


a1 = fg.FactorGraph.makeMatchFactorGraph(m1)



print 'Done'

