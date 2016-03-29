# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import Model as md

# Player, sample, new_sample, samples, new_samples

players = set()

def mh_mcmc(versus_list, n=1000):
    getPlayers(versus_list)
    setupSamples()  # sample, new_sample, new_samples
    for i in range(n):

        print '==='
        print 'old'
        pi_old = calculatePi(versus_list,new=False)
        propose()
        print 'new'
        pi_new = calculatePi(versus_list,new=True)
        ratio = pi_new / pi_old
        alpha = min(1,ratio)  # alpha(x-->x')
        acc = np.random.rand() < alpha
        addSample(accepted=acc)
    updateSamples()

def getPlayers(versus_list):
    global players
    for v in versus_list:
        for p in v.t1.players:
            players.add(p)
        for p in v.t2.players:
            players.add(p)


def setupSamples():
    global players
    for p in players:
        p.sample = 25
        p.new_sample = 25
        p.new_sample_list = []

def calculatePi(versus_list, new):
    global players
    rv = 1
    for p in players:
        temp = p.new_sample if new else p.sample
        print 'p', temp
        rv *= p.prob(temp)
    for v in versus_list:
        temp = v.t1.new_sample()-v.t2.new_sample() if new else v.t1.sample()-v.t2.sample()
        print 'r ', temp
        rv *= v.prob(temp)
    return rv

def propose():
    global players
    for p in players:
        temp = np.random.normal(p.sample,3)
        temp = 0 if temp <= 0 else temp
        temp = 49.9 if temp >= 50 else temp
        p.new_sample = temp

def addSample(accepted):
    global players
    for p in players:
        if accepted:
            p.new_sample_list.append(p.new_sample)
            p.sample = p.new_sample
        else:
            p.new_sample_list.append(p.sample)


def updateSamples():
    global players
    for p in players:
        p.sample_list = p.new_sample_list
        p.updateKernel()





# def mh_mcmc(match):  # a_match --> atomic match --> 1vs1
#     teams = match.teamsOrdered()
#     players = match.players()
#     a_match = 5
#     res = -1 * (a_match.standings[t1] - a_match.standings[t2])  # 1: P1 wins, 0: Draw, -1: P2 wins
#
#     N = 3000
#     samples = np.zeros([N,3])
#     samples[:,2] = res  # res is fixed
#     samples[0,0] = 25  # s1 = 25
#     samples[0,1] = 25  # s2 = 25
#     for i in range(1,N):
#         s1 = samples[i-1,0]
#         s2 = samples[i-1,1]
#         s1_new, s2_new = propose(s1,s2) # q(x-->x')
#
#         # prob_s1_new = probFromEmpirical(t1_pri,s1_new)  # P(S1_new)
#         # prob_s2_new = probFromEmpirical(t2_pri,s2_new)  # P(S2_new)
#         prob_s1_new = t1.pOfX(s1_new)  # P(S1_new)
#         prob_s2_new = t2.pOfX(s2_new)  # P(S2_new)
#         prob_r_new = md.sigmoid(s1_new-s2_new) if res == 1 else md.sigmoid(s2_new-s1_new)  # P(R|S1_new,S2_new)
#         pi_new = prob_r_new * prob_s1_new * prob_s2_new  # P(S1_new, S2_new, R)
#
#         prob_s1 = t1.pOfX(s1)  # P(S1)
#         prob_s2 = t2.pOfX(s2)  # P(S2)
#         prob_r = md.sigmoid(s1-s2) if res == 1 else md.sigmoid(s2-s1)  # P(R|S1,S2)
#         pi = prob_r * prob_s1 * prob_s2  # P(S1, S2, R)
#
#         ratio = pi_new / pi
#         accept = min(1,ratio)  # alpha(x-->x')
#         if np.random.rand() < accept:
#             samples[i,0] = s1_new
#             samples[i,1] = s2_new
#         else:
#             samples[i,0] = s1
#             samples[i,1] = s2
#     t1.samples = samples[25:,0]
#     t2.samples = samples[25:,1]
#     print "Atomix executed"