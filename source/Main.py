# -*- coding: utf-8 -*-

import numpy as np
np.random.seed(123456)
import matplotlib.pyplot as plt
import DataFactory as df
import Visu as vi
import MC as mc
import pdb
import time
import Reader
import Model as md

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
#
# m = df.generateSyntheticMatchesFullTimes(t,3)



### TEST #2
### =======
# p,t = df.generateSyntheticData(6,1)
# m = df.generateSyntheticMatchesFullTimes(t,10)


### TEST #3
### =======
# data = Reader.read_data('../data/tennis/ausopen.csv')
# p,t,m = Reader.form_objects(data)


### TEST #4
### =======
# data = Reader.read_data('../data/football/germany.csv')
# p,t,m = Reader.form_objects(data)


### EXECUTE MCMC
### ============
ite = 100
agr = 0.5

tic = time.time()
decisions = mc.mh_mcmc(p, m, agr, ite)
toc3 = time.time() - tic

accepted = np.sum(decisions)
rejected = ite - np.sum(decisions)


### CONSOLE LOG
### ===========
print 'MCMC time: ', int(toc3), ' seconds'
print 'Accepted: ', accepted, ' -- Rejected: ', rejected
# vi.printReelSkill(p,sorted=True)
vi.printMeans(p,sorted=True)
# vi.printError(p,sorted=True)



### FILE LOG
### ========
# time = time.ctime().replace(' ','_')
# vi.log('../logs/reelskill.log', time, 'Synthetic', m, p, t, ite, agr, toc3, accepted, rejected)


### SAVE PLOT
### =========
# fig1 = plt.figure()
# ax = fig1.add_subplot(111)
# vi.plot_est(p,ax)
# fname = '../img/' + time
# plt.savefig(fname)
# plt.show(block=True)


print 'All Done'

