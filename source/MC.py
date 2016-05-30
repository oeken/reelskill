# -*- coding: utf-8 -*-
"""This module contains metropolis-hastings and gibbs sampler (domain-specific) algorithms"""

import numpy as np
import model as md

players = None
versuses = None


def mh_mcmc(player_list, versus_list, aggression, n=1000):
    global players
    players = player_list

    decisions = np.zeros([n,1])
    reset_players()  # sample, new_sample, new_samples
    for i in xrange(n):
        if i % 50 == 0: print i
        pi_old = calculate_pi(versus_list,new=False)
        propose(aggression)
        pi_new = calculate_pi(versus_list,new=True)
        ratio = pi_new - pi_old  # subtract since working with log prob
        alpha = min(0,ratio)  # alpha(x-->x')
        acc = np.log(np.random.rand()) < alpha
        decisions[i,0] = acc
        add_sample(accepted=acc)
    update_samples()
    return 'MH', decisions


def gibbs_mcmc(player_list, versus_list, n=1000):
    global players, versuses
    players = player_list
    versuses = versus_list

    reset_players()
    all_reports = all_player_reports()
    for i in xrange(n):
        if i % 50 == 0: print i
        x = np.arange(-50,100)

        # order_to_follow = np.random.permutation(len(player_list))
        for p,p_reports in zip(player_list, all_reports):
        # for index in order_to_follow:
        #     p = player_list[index]
        #     p_reports = all_reports[index]

            conditional = np.zeros([1,150])
            for report in p_reports:
                vs, wld, t = report
                additive = vs.t1.mc_sample - p.mc_sample if t == 1 else vs.t2.mc_sample - p.mc_sample
                subtractive = vs.t2.mc_sample if t == 1 else vs.t1.mc_sample
                if wld == 0:
                    temp = md.bi_expo(x,mean=subtractive-additive)
                elif wld == 1:
                    temp = md.sigmoid(x-(subtractive-additive))
                elif wld == -1:
                    temp = md.sigmoid(-x+(subtractive-additive))
                else:
                    raise ValueError('Somethings wrong!')
                conditional = conditional + np.log(temp)
            truncate = conditional[0,50:100]
            sample = x[np.nonzero(md.multinomial_log(1,truncate))[0][0]] + 50

            # ax = f1.add_subplot(10,6,counter)
            # ax.plot(x[50:100],np.exp(truncate))
            # print 'Counter ', counter, 'Player ', p.name, 'RS ', p.reel_skill
            # counter += 1
            # print 'Sample ',p.name,' -> ', sample
            # plt.figure()
            # plt.plot(x[50:100],np.exp(truncate))
            # plt.show()

            # p.mc_sample = sample #
            # p.mc_sample_list_new = np.append(p.mc_sample_list_new, sample) #

            p.mc_sample_new = sample ###
            p.mc_sample_list_new = np.append(p.mc_sample_list_new, sample) ###

        for p in player_list:  ###
            p.mc_sample = p.mc_sample_new ###
    update_samples()
    return 'Gibbs'


def all_player_reports():
    global players
    rv = []
    for p in players:
        rv.append(player_reports(p))
    return rv


def player_reports(player):
    global players, versuses
    rv = []
    for vs in versuses:
        involved, wld, team = vs.report_of_player(player)
        if involved:
            rv.append( (vs, wld, team) )
    return rv


def reset_players(random=False):
    global players
    for p in players: p.reset_samples(random)


def calculate_pi(versus_list, new):
    # global players

    # for p in players:
    #     temp = p.new_sample if new else p.sample
    #     rv += np.log(p.prob(temp))  # log probability
    # TODO delete the code above
    rv = 0
    for v in versus_list:
        temp = v.t1.mc_sample_new - v.t2.mc_sample_new if new else v.t1.mc_sample - v.t2.mc_sample
        rv += np.log(v.prob(temp))
    return rv


def propose(aggression):
    global players
    for p in players:
        temp = np.random.normal(p.mc_sample, aggression)
        temp = 0 if temp <= 0 else temp
        temp = 49.9 if temp >= 50 else temp
        p.mc_sample_new = temp


def add_sample(accepted):
    global players
    for p in players:
        if accepted:
            p.mc_sample_list_new.append(p.mc_sample_new)
            p.mc_sample = p.mc_sample_new
        else:
            p.mc_sample_list_new.append(p.mc_sample)


def update_samples():
    global players
    for p in players:
        p.mc_sample_list = p.mc_sample_list_new
        p.update_kernel()



