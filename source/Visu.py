# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import Model as md
import MC as mc
import DataFactory as df


def error(players):
    N = len(players)
    rv = [None] * N
    for i in range(len(players)):
        rv [i] = players[i].mean() - players[i].reel_skill
    return rv, np.sum(np.array(rv)**2)/N, np.std(np.array(rv))

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
    err, mse, std = error(players)
    print "MSE : ", mse
    print "STD : ", std
    for i in range(len(players)):
        print "Error on",players[i].name,": ", err[i]
    print " "

def plot_est(players, ax, sample_list=True, reel_skill=True):
    ax.set_title('Estimated Skill Distribution of Players')
    ax.set_xlim([0,50])
    colors = ['r','g','b','m','c','y','w']
    for p,col in zip(players,colors):

        if sample_list:
            sns.kdeplot(np.array(p.sample_list),shade=True,color=col,ax=ax)
        if reel_skill:
            ax.axvline(p.reel_skill,color=col,label=p.name,ls='dashed')
        # if mean: plt.plot(p.mean(),0.001,'*',ms=10,color=col)
    ax.legend()
