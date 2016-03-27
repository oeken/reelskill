# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import Model as md



def probFromEmpirical(data, value):
    freqs = np.histogram(data,bins=50,range=[0,50],density=True)[0]
    return freqs[int(value)]

def sampleFromEmpirical(data, size):
    freqs = np.histogram(data,bins=50,range=[0,50],density=True)[0]
    cum = np.cumsum(freqs)
    rv = np.zeros([size,1])
    for i in range(size):
        u = np.random.rand() < cum
        rv[i] = np.argmax(u) + np.random.rand()
    return rv


def propose(s1,s2):
    # the proposal is multivariate gaussian
    [s1_new, s2_new] = np.random.multivariate_normal([s1,s2],np.eye(2)*1)
    s1_new = 0 if s1_new < 0 else s1_new
    s2_new = 0 if s2_new < 0 else s2_new
    s1_new = 49.9 if s1_new > 49.9 else s1_new
    s2_new = 49.9 if s2_new > 49.9 else s2_new
    return s1_new, s2_new

def mh_mcmc(a_match):  # a_match --> atomic match --> 1vs1
    t1 = a_match.teamsOrdered()[0]
    t2 = a_match.teamsOrdered()[1]
    t1_pri = t1.prior
    t2_pri = t2.prior
    res = -1 * (a_match.standings[t1] - a_match.standings[t2])  # 1: P1 wins, 0: Draw, -1: P2 wins

    N = 1000
    samples = np.zeros([N,3])
    samples[:,2] = res  # res is fixed
    samples[0,0] = 25  # s1 = 25
    samples[0,1] = 25  # s2 = 25
    for i in range(1,N):
        s1 = samples[i-1,0]
        s2 = samples[i-1,1]
        s1_new, s2_new = propose(s1,s2) # q(x-->x')

        prob_s1_new = probFromEmpirical(t1_pri,s1_new)  # P(S1_new)
        prob_s2_new = probFromEmpirical(t2_pri,s2_new)  # P(S2_new)
        prob_r_new = md.sigmoid(s1_new-s2_new) if res == 1 else md.sigmoid(s2_new-s1_new)  # P(R|S1_new,S2_new)
        pi_new = prob_r_new * prob_s1_new * prob_s2_new  # P(S1_new, S2_new, R)

        prob_s1 = probFromEmpirical(t1_pri,s1)  # P(S1)
        prob_s2 = probFromEmpirical(t2_pri,s2)  # P(S2)
        prob_r = md.sigmoid(s1-s2) if res == 1 else md.sigmoid(s2-s1)  # P(R|S1,S2)
        pi = prob_r * prob_s1 * prob_s2  # P(S1, S2, R)

        ratio = pi_new / pi
        accept = min(1,ratio)  # alpha(x-->x')
        if np.random.rand() < accept:
            samples[i,0] = s1_new
            samples[i,1] = s2_new
        else:
            samples[i,0] = s1
            samples[i,1] = s2
    t1.prior = samples[25:,0]
    t2.prior = samples[25:,1]
    print "Atomix executed"




def updateAtomic(a_matches):
    if len(a_matches) == 0:
        return
    else:
        current = a_matches[0]
        mh_mcmc(current)
        update(a_matches[1:0])

def update(matches):
    if len(matches) == 0:
        return
    else:
        a_matches = matches[0].atomicMatches()
        updateAtomic(a_matches)
        plt.figure()
        ax = sns.distplot(t1.prior,bins=50)
        ax = sns.distplot(t2.prior,bins=50)
        ax.set_xlim([0, 50])
        ax.set_ylim([0, 1])
        update(matches[1:])


p1 = md.Player(reel_skill=30)
p2 = md.Player(reel_skill=9)
# p3 = md.Player(reel_skill=12)

t1 = md.Team([p1])
print "Allan : " , t1.prior
t2 = md.Team([p2])
# t3 = md.Team([p3])

m1 = md.Match({t1:1,t2:2})
m2 = md.Match({t1:1,t2:2})
m3 = md.Match({t1:1,t2:2})
m4 = md.Match({t1:1,t2:2})
m5 = md.Match({t1:2,t2:1})
m6 = md.Match({t1:2,t2:1})
# m7 = md.Match({t1:2,t2:1})
# m8 = md.Match({t1:2,t2:1})
# m9 = md.Match({t1:1,t2:2})
# m10 = md.Match({t1:1,t2:2})
# m11 = md.Match({t1:1,t2:2})
# m12 = md.Match({t1:1,t2:2})
update([m1,m2,m3,m4,m5,m6])
# ,m7,m8,m9,m10,m11,m12




plt.show()
print 'Done'

