# -*- coding: utf-8 -*-
import numpy as np
import Model as md
import matplotlib.pyplot as plt

players = None
# players = set()

def mh_mcmc(player_list, versus_list, agression, n=1000):
    global players
    players = player_list
    decisions = np.zeros([n,1])
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

def gibbs_mcmc(player_list, versus_list, n=1000):
    global players
    players = player_list
    setupSamples(fixed=True)
    # f1 = plt.figure()
    # counter = 1
    all_reports = []
    for p in player_list:
        all_reports.append(player_reports(versus_list, p))

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
                additive = vs.t1.sample() - p.sample if t == 1 else vs.t2.sample() - p.sample
                subtractive = vs.t2.sample() if t == 1 else vs.t1.sample()
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


            # p.sample = sample #
            # p.new_sample_list = np.append(p.new_sample_list, sample) #
            p.new_sample = sample ###
            p.new_sample_list = np.append(p.new_sample_list, sample) ###

        for p in player_list:  ###
            p.sample = p.new_sample ###

    updateSamples()


    #             if vs.r == 0:
    #                 if p in vs.t1.players:
    #                     additive = vs.t1.sample() - p.sample
    #                     subtractive = vs.t2.sample()
    #                     expo = True
    #                 elif p in vs.t2.players:
    #                     additive = vs.t2.sample() - p.sample
    #                     subtractive = vs.t1.sample()
    #                     expo = True
    #                 else:
    #                     continue
    #             else:
    #                 expo = False
    #                 additive = vs.t1.sample() - p.sample if p in vs.t1.players else vs.t2.sample() - p.sample
    #                 subtractive = vs.t2.sample() if p in vs.t1.players else vs.t1.sample()
    #                 if vs.r == 1 and p in vs.t1.players: w = True
    #                 elif vs.r == 1 and p in vs.t2.players: w = False
    #                 elif vs.r == -1 and p in vs.t1.players: w = False
    #                 elif vs.r == -1 and p in vs.t2.players: w = True
    #
    #             if expo:
    #                 temp = md.bi_expo(x,mean=subtractive-additive)
    #             elif w:
    #                 temp = md.sigmoid(x-(subtractive-additive))
    #             else:
    #                 temp = md.sigmoid(-x+(subtractive-additive))
    #             conditional = conditional + np.log(temp)
    #             # conditional = conditional * temp
    #             # conditional = conditional / np.sum(conditional)
    #         dummy = conditional[0,50:100]
    #         # dummy = dummy / np.sum(dummy)
    #         dummy2 = np.exp(dummy)
    #         dummy2 = dummy2 / np.sum(dummy2)
    #
    #
    #
    #         sample = x[np.nonzero(md.multinomial_log(1,dummy))[0][0]] + 50
    #         print 'Sample ', sample
    #
    #         plt.figure()
    #         plt.plot(x[50:100],dummy2)
    #         plt.show()
    #         # sample = x[np.nonzero(np.random.multinomial(1,dummy))[0][0]] + 50
    #         p.sample = sample
    #         p.new_sample_list = np.append(p.new_sample_list, sample)
    # updateSamples()


def player_reports(versus_list, player):
    rv = []
    for vs in versus_list:
        involved, wld, team = vs.report_of_player(player)
        if involved:
            rv.append( (vs, wld, team) )
    return rv





# x = np.arange(0,50)
# y = md.sigmoid(x)
# y = y / np.sum(y)
# hi = np.random.multinomial(10000,y)
# hi = hi.astype(float)
# hi = hi / np.sum(hi)

# def sample_gibbs()


# def getPlayers(versus_list):
#     global players
#     for v in versus_list:
#         for p in v.t1.players:
#             players.add(p)
#         for p in v.t2.players:
#             players.add(p)

def setupSamples(fixed=True):
    global players
    for p in players:
        p.reset(fixed)
        # p.sample = 25
        # p.new_sample = 25
        # p.new_sample_list = []



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



