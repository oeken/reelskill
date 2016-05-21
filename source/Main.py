# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import DataFactory as df
import Visu as vi
import MC as mc
import Model as md


# p1 = md.Player(reel_skill=30)
# p2 = md.Player(reel_skill=9)
# p3 = md.Player(reel_skill=12)

# t1 = md.Team([p1])
# t2 = md.Team([p2,p3])
# t3 = md.Team([p3])
# t4 = md.Team([p2])
# results = [{t1:1,t2:2},{t4:1,t3:2},{t4:1,t3:2},{t4:1,t3:2}]

p,t,m = df.generateSyntheticData(10,1)
# p1 = md.Player(reel_skill= 10)
# p2 = md.Player(reel_skill= 45)
# p3 = md.Player(reel_skill= 2)
# p = [p1,p2,p3]
#
# t1 = md.Team([p1])
# t2 = md.Team([p2])
# t3 = md.Team([p3])

m = df.generateSyntheticMatchesFull(t)
m += df.generateSyntheticMatchesFull(t)
m += df.generateSyntheticMatchesFull(t)

# m += df.generateSyntheticMatchesFull(t)
# m += df.generateSyntheticMatchesFull(t)



vi.printReelSkill(p,sorted=True)
mc.mh_mcmc(m,1000)
vi.printMeans(p,sorted=True)
vi.printError(p)
vi.plot(p)
plt.show(block=False)


# p = df.generateSyntheticPlayers(4)
# t1 = md.Team(p[0:2])
# t2 = md.Team(p[2:])
# t = [t1,t2]
# print 'T1', t1
# print 'T2', t2
# m = []
# m += df.generateSyntheticMatchesFull(t)
# m += df.generateSyntheticMatchesFull(t)
# m += df.generateSyntheticMatchesFull(t)
# t3 = md.Team([p[0]])
# t4 = md.Team([p[1]])
# m.append(df.simulateTwoTeams(t3,t4))
# m.append(df.simulateTwoTeams(t3,t4))
# m.append(df.simulateTwoTeams(t3,t4))
# m.append(df.simulateTwoTeams(t3,t4))
# m.append(df.simulateTwoTeams(t3,t4))
# m.append(df.simulateTwoTeams(t3,t4))

# liste = md.Versus.produceVs(results)
# mc.mh_mcmc(liste,5000)




# plt.figure()
# ax = sns.distplot(p1.sample_list,bins=50)
# ax = sns.distplot(p2.sample_list,bins=50)
# ax = sns.distplot(p3.sample_list,bins=50)
# plt.figure()
# x = np.arange(0,50,0.1)
# plt.plot(x,p1.prob(x))
# plt.plot(x,p2.prob(x))
# plt.plot(x,p3.prob(x))
# plt.show()

print 'selam'

