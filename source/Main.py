# -*- coding: utf-8 -*-

import numpy as np
# np.random.seed(123457)
import matplotlib.pyplot as plt
import DataFactory as df
import Visu as vi
import MC as mc
import pdb
import time
import Reader
import Model as md
import Test as ts
import trueskill




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
# m = df.generateSyntheticMatchesFullTimes(t,20)


### TEST #1.1
### =========
# p1 = md.Player(reel_skill=20)
# p2 = md.Player(reel_skill=35)
# p = [p1,p2]
#
# t1 = md.Team([p1])
# t2 = md.Team([p2])
# t = [t1,t2]
#
# md.draw_factor = 0.33
# m = df.generateSyntheticMatchesFullTimes(t,10)



### TEST #2
### =======
# md.draw_factor = 0
# p,t = df.generateSyntheticData(6,1)
# m = df.generateSyntheticMatchesFullTimes(t,100)


### TEST #3
### =======
# data_training = Reader.read_data('../data/tennis/ausopen.csv','../data/tennis/rg.csv')
# data_test = Reader.read_data('../data/tennis/wimbledon.csv','../data/tennis/usopen.csv')
# p,t,m = Reader.form_objects(data_training)


### TEST #4
### =======
# data = Reader.read_data('../data/football/turkey.csv')
# N = data.shape[0]
# data_training = data[:N/2,:]
# data_test = data[N/2:,:]
# p,t,m = Reader.form_objects(data_training)

### TEST #5
### =======
md.draw_factor = 0
data = Reader.read_data('../data/basketball/nba.csv')
N = data.shape[0]
data_training = data[:N/2,:]
data_test = data[N/2:,:]
p,t,m = Reader.form_objects(data_training)


### PREPARE AXES
### ============
fig1 = plt.figure(figsize=(16.0, 10.0))
ax_metro = fig1.add_subplot(212)
ax_metro.set_title('Based on Metropolis Hastings')
ax_metro.set_xlabel('Skill')
ax_metro.set_ylabel('Probability')

ax_gibbs = fig1.add_subplot(211)
ax_gibbs.set_title('Based on Gibbs Sampler')
ax_gibbs.set_xlabel('Skill')
ax_gibbs.set_ylabel('Probability')




vi.printReelSkill(p,sorted=True)
current_time = time.ctime().replace(' ','_')

### EXECUTE MCMC
### ============
np.random.seed(None)

### GIBBS
ite_gibbs = 4000
tic = time.time()
mc.gibbs_mcmc(p, m, ite_gibbs)
toc_gibbs = time.time() - tic
print 'GIBBS time: ', int(toc_gibbs), ' seconds'
vi.printMeans(p,sorted=True)
vi.plot_est(p, ax_gibbs, eco=True)
pred, probs, cor, test_text = ts.compare(data_test,p)
vi.log('../logs/reelskill.log', current_time, 'Football', m, p, t, ite_gibbs, toc_gibbs, 'Gibbs', test_text=test_text)

## Fancy up plot
textstr = 'Iterations = %d'%(ite_gibbs)
props = dict(boxstyle='round', facecolor='wheat', alpha=0.7)
ax_gibbs.text(0.05, 0.95, textstr, transform=ax_gibbs.transAxes, fontsize=14,
                            verticalalignment='top', bbox=props)


### METROPOLIS-HASTINGS
ite_metro = 5000
agr = 0.8
tic = time.time()
decisions = mc.mh_mcmc(p, m, agr, ite_metro)
toc_metro = time.time() - tic
accepted = np.sum(decisions)
rejected = ite_metro - np.sum(decisions)
print 'METRO time: ', int(toc_metro), ' seconds'
print 'Accepted: ', accepted, ' -- Rejected: ', rejected
vi.printMeans(p,sorted=True)
vi.plot_est(p, ax_metro, eco=True)
pred, probs, cor, test_text = ts.compare(data_test,p)
vi.log('../logs/reelskill.log', current_time, 'Football', m, p, t, ite_metro, toc_metro, 'Metropolis', test_text=test_text, aggr=agr,  accepted=accepted, rejected=rejected, br=True)


## Fancy up plot
textstr = 'Iterations = %d\nProposal Deviation = %.2f\nAccepted = %d, Rejected = %d'%(ite_metro, agr, accepted, rejected)
props = dict(boxstyle='round', facecolor='wheat', alpha=0.7)
ax_metro.text(0.05, 0.95, textstr, transform=ax_metro.transAxes, fontsize=14,
                            verticalalignment='top', bbox=props)


## Save figure
fname = '../img/' + current_time
plt.savefig(fname)
plt.show(block=True)


print 'All Done'

