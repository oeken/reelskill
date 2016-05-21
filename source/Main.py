# -*- coding: utf-8 -*-

import numpy as np
np.random.seed(123456)
import matplotlib.pyplot as plt
import DataFactory as df
import Visu as vi
import MC as mc
import pdb
import time

import Model as md

p1 = md.Player(reel_skill=5)
p2 = md.Player(reel_skill=15)
p3 = md.Player(reel_skill=25)
p4 = md.Player(reel_skill=35)
p5 = md.Player(reel_skill=45)
p = [p1,p2,p3,p4,p5]

t1 = md.Team([p1])
t2 = md.Team([p2])
t3 = md.Team([p3])
t4 = md.Team([p4])
t5 = md.Team([p5])
t = [t1,t2,t3,t4,t5]

m = df.generateSyntheticMatchesFullTimes(t,3)

# results = [{t1:1,t2:2},{t4:1,t3:2},{t4:1,t3:2},{t4:1,t3:2}]

# p,t,m = df.generateSyntheticData(8,1)

# m = df.generateSyntheticMatchesFullTimes(t,10)


ite = 10000
agr = 0.5

tic = time.time()
decisions = mc.mh_mcmc(m, agr, ite)
toc3 = time.time() - tic

accepted = np.sum(decisions)
rejected = ite - np.sum(decisions)



print 'MCMC time: ', int(toc3), ' seconds'
print 'Accepted: ', accepted, ' -- Rejected: ', rejected

vi.printReelSkill(p,sorted=True)
vi.printMeans(p,sorted=True)
vi.printError(p)

fig1 = plt.figure()
ax = fig1.add_subplot(111)
vi.plot_est(p,ax)



fname = '../img/' + time.ctime().replace(' ','_')
plt.savefig(fname)
plt.show(block=True)


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

print 'All Done'

