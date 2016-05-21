# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import Model as md
import MC as mc
import DataFactory as df

def error(players):
    rv = [None] * len(players)
    for i in range(len(players)):
        rv[i] = players[i].mean() - players[i].reel_skill

    return rv, np.sum(np.array(rv)**2), np.var(np.array(rv))

def sortByReelSkill(players):
    rs = np.zeros([len(players)])
    for i in range(len(players)):
        rs[i] = players[i].reel_skill
    sorted = np.argsort(rs)
    return sorted

def sortByMean(players):
    rs = np.zeros([len(players)])
    for i in range(len(players)):
        rs[i] = players[i].mean()
    sorted = np.argsort(rs)
    return sorted


def printMeans(players,sorted=False):
    print 'MEANS'
    print '====='
    index_list = sortByMean(players) if sorted else np.arange(len(players))
    counter = 1
    for i in index_list:
        print counter,". Mean of",players[i].name,": ",players[i].mean()
        counter += 1
    print " "

def printReelSkill(players,sorted=False):
    print "REEL SKILLS"
    print "==========="
    index_list = sortByReelSkill(players) if sorted else np.arange(len(players))
    counter = 1
    for i in index_list:
        print counter,". ReelSkill of",players[i].name,": ",players[i].reel_skill
        counter += 1
    print " "

def printError(players):
    print "ERRORS"
    print "======"
    err, sse, var = error(players)
    print "SSE : ", sse
    print "Variance : ", var
    for i in range(len(players)):
        print "Error on",players[i].name,": ", err[i]
    print " "

def plot(players, sample_list=True, reel_skill=True, mean=True):
    plt.figure()
    colors = ['r','g','b','m','c','y','w']
    for p,col in zip(players,colors):
        if sample_list: sns.distplot(p.sample_list,color=col)
        if reel_skill: plt.plot(p.reel_skill,0.001,'o',ms=10,color=col)
        if mean: plt.plot(p.mean(),0.001,'*',ms=10,color=col)

