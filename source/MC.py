# -*- coding: utf-8 -*-
import numpy as np


players = set()

def mh_mcmc(versus_list, agression, n=1000):
    decisions = np.zeros([n,1])
    getPlayers(versus_list)
    setupSamples()  # sample, new_sample, new_samples
    for i in xrange(n):
        if i%50 == 0: print i
        pi_old = calculatePi(versus_list,new=False)
        propose(agression)
        pi_new = calculatePi(versus_list,new=True)
        ratio = pi_new - pi_old  # subtract since working with log prob
        alpha = min(0,ratio)  # alpha(x-->x')
        acc = np.log(np.random.rand()) < alpha
        decisions[i,0] = acc
        addSample(accepted=acc)
    updateSamples()
    return decisions

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
    rv = 0
    for p in players:
        temp = p.new_sample if new else p.sample
        rv += np.log(p.prob(temp))  # log probability
    for v in versus_list:
        temp = v.t1.new_sample()-v.t2.new_sample() if new else v.t1.sample()-v.t2.sample()
        rv += np.log(v.prob(temp))
    return rv

def propose(agression):
    global players
    for p in players:
        temp = np.random.normal(p.sample,agression)
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



