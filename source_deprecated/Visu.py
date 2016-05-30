# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import source.model as md
import source.mc as mc
import source.factory as df


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
    rv += '\n'
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

def plot_est(players, ax, burnin=0, sample_list=True, reel_skill=True, eco=False):

    ax.set_xlim([0,50])
    colors = ['r','g','b','m','c','y','k','#b5cf11','#cb071c','#34736d','#d26b34','#160b60','#616235','#3a1679','#ec44e1','#8e0e12','#22c8d3']

    index_list = sortByMean(players)
    index_list_true = index_list[::-1]
    if eco: index_list = np.append(index_list[-5:][::-1], index_list[:5][::-1])
    def rank(p):
        return np.nonzero(index_list_true == players.index(p))[0][0]+1

    for index,col in zip(index_list,colors):
        p = players[index]

        if sample_list:
            sns.kdeplot(np.array(p.sample_list)[burnin:],shade=True,color=col,ax=ax,label='%d. %s (%.2f)'%(rank(p), p.name, p.mean()))
        if reel_skill:
            ax.axvline(p.reel_skill,color=col,ls='dashed')
    ax.legend()


def log(file_name, date_time, data_name, m, p, t, ite, elapsed,  algo, test_text=None, aggr=None, accepted=None, rejected=None, br=False):
    f= open(file_name,mode='a')
    f.write(date_time+'\n'+'='*30+'\n')
    f.write('Algorithm             : '+algo+'\n')
    f.write('Dataset Name          : '+data_name+'\n')
    f.write('# of matches          : '+str(len(m))+'\n')
    f.write('# of players          : '+str(len(p))+'\n')
    f.write('# of teams            : '+str(len(t))+'\n')
    f.write('# of iterations       : '+str(ite)+'\n')
    f.write('Elapsed time(seconds) : '+str(int(elapsed))+'\n')
    f.write('Deviation in proposal : '+str(aggr)+'\n')
    f.write('Proposals accepted    : '+str(accepted)+'\n')
    f.write('Proposals rejected    : '+str(rejected)+'\n')


    c = 0
    index = sortByReelSkill(p)
    f.write('\nReel Skills\n'+'='*30+'\n')
    for i in range(len(p)):
        player = p[index[-1-i]]
        f.write(str(c)+'. '+player.name+' - '+str(player.reel_skill)+'\n')
        c+= 1

    c = 0
    index = sortByMean(p)
    f.write('\nSkills\n'+'='*30+'\n')
    for i in range(len(p)):
        player = p[index[-1-i]]
        f.write(str(c)+'. '+player.name+' - '+'{:.2f}'.format(player.mean())+'\n')
        c += 1

    f.write('\nTEST\n'+'='*30+'\n')
    f.write(str(test_text))

    f.write('\n'*3)
    f.write('-x-'*15)
    f.write('\n'*3)

    if br: f.write('\n' * 20)

    f.close()



