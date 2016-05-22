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


def printMeans(players,sorted=False, show=True):
    rv = 'MEANS\n=====\n'
    index_list = sortByMean(players) if sorted else np.arange(len(players))
    index_list = index_list[::-1]
    counter = 1
    for i in index_list:
        rv += str(counter) + ". Mean of " + players[i].name + ": " + str(players[i].mean()) + '\n'
        counter += 1
    if show: print rv
    return rv

def printReelSkill(players,sorted=False, show=True):
    rv = 'REEL SKILLS\n===========\n'
    index_list = sortByReelSkill(players) if sorted else np.arange(len(players))
    index_list = index_list[::-1]
    counter = 1
    for i in index_list:
        rv = rv + str(counter) + '. ReelSkill of ' + players[i].name + ': ' + str(players[i].reel_skill) + '\n'
        counter += 1
    if show: print rv
    return rv

def printError(players, sorted=False, show=True):
    rv = 'ERRORS\n======\n'
    err, mse, std = error(players)
    rv += 'MSE : ' + str(mse) + '\n' + 'STD : ' + str(std) + '\n'

    index_list = sortByReelSkill(players) if sorted else np.arange(len(players))
    index_list = index_list[::-1]
    for i in index_list:
        rv += "Error on " + players[i].name + ": " + str(err[i]) + '\n'
    if show: print rv
    return rv

def plot_est(players, ax, sample_list=True, reel_skill=True):
    ax.set_title('Estimated Skill Distribution of Players')
    ax.set_xlim([0,50])
    colors = ['r','g','b','m','c','y','w']
    for p,col in zip(players,colors):

        if sample_list:
            sns.kdeplot(np.array(p.sample_list),shade=True,color=col,ax=ax)
        if reel_skill:
            ax.axvline(p.reel_skill,color=col,label=p.name,ls='dashed')
    ax.legend()


def log(fname, time, dname, matches, players, teams, iterations, aggr, elapsed, acc, rej):
    f= open(fname,mode='a')
    f.write(time+'\n'+'='*30+'\n')
    f.write('Dataset Name          : '+dname+'\n')
    f.write('# of matches          : '+str(len(matches))+'\n')
    f.write('# of players          : '+str(len(players))+'\n')
    f.write('# of teams            : '+str(len(teams))+'\n')
    f.write('# of iterations       : '+str(iterations)+'\n')
    f.write('Elapsed time(seconds) : '+str(int(elapsed))+'\n')
    f.write('Deviation in proposal : '+str(aggr)+'\n')
    f.write('Proposals accepted    : '+str(acc)+'\n')
    f.write('Proposals rejected    : '+str(rej)+'\n')

    index = sortByMean(players)
    f.write('\nSkills\n'+'='*30+'\n')
    for i in range(len(players)):
        p = players[index[-1-i]]

        f.write(p.name+' - '+'{:.2f}'.format(p.mean())+'\n')

    f.write('\n'*3)
    f.write('-x-'*15)
    f.write('\n'*3)
    f.close()



