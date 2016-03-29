# -*- coding: utf-8 -*-
import numpy as np

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



