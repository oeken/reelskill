# -*- coding: utf-8 -*-
""" This module contains the functions to evaluate results and calculate some stats """

import numpy as np
import model as mo
import os
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import seaborn as sns
import pandas as pd


colors = ['r','g','b','m','c','y','k','#b5cf11','#cb071c','#34736d','#d26b34','#160b60','#616235','#3a1679','#ec44e1','#8e0e12','#22c8d3']


class Output:
    def __init__(self, test_time, m_train, algo, dataname, elapsed, m_test=None, aggr=None, decisions=None, ite=None):
        # Phase I
        self.test_time = test_time
        self.players = mo.Player.players_all
        self.teams = mo.Team.teams_all
        self.train_matches = m_train
        self.algo = algo
        self.dataname = dataname
        self.num_players = len(self.players)
        self.num_teams = len(self.teams)
        self.num_matches = len(m_train)
        self.elapsed_time = elapsed

        # Phase I.5
        self.proposal_variance = aggr
        self.proposals_accepted = np.sum(decisions) if decisions is not None else None
        self.proposals_rejected = ite - self.proposals_accepted if decisions is not None else None
        self.num_ite = ite

        # Phase II
        self.reel_skills = distill_reel_skills(self.players)

        # Phase III
        s, u, se, re = distill_skills(algo, self.players, self.reel_skills)
        self.skills = s
        self.uncertainties = u
        self.skill_errors = se
        self.ranking_errors = re

        # Phase IV
        self.test_matches = m_test

        self.csv_path = '../out/%s/%s/csv/' % (self.test_time, self.algo)
        self.plot_path = '../out/%s/%s/plot/' % (self.test_time, self.algo)

    def signature_csv(self):
        string = 'Date,%s\nAlgorithm,%s\nDataset,%s\n' \
                 'Number of matches,%d\nNumber of players,%d\nNumber of teams,%d\n' \
                 'Number of iterations,%s\nElapsed time,%d\nVariance of proposal,%s\n' \
                 'Proposals accepted,%s\nProposals rejected,%s' %(self.test_time, self.algo, self.dataname,
                                                                  self.num_matches, self.num_players, self.num_teams,
                                                                  str(self.num_ite), self.elapsed_time, str(self.proposal_variance),
                                                                  str(self.proposals_accepted), str(self.proposals_rejected))

        if not os.path.exists(self.csv_path):
            os.makedirs(self.csv_path)

        f_name = self.csv_path + '1_signature.csv'
        f = open(f_name,mode='w')
        f.write(string)
        f.close()

    def reel_skills_csv(self):
        if self.reel_skills is not None:
            string = '#,Player,Reel Skill\n'
            sorted_players = sort_according_to(self.reel_skills, self.players)
            counter = 1
            for p in sorted_players:
                string += '%d,%s,%.2f\n' %(counter, p.name, p.reel_skill)
                counter += 1

            if not os.path.exists(self.csv_path):
                os.makedirs(self.csv_path)

            f_name = self.csv_path + '2_reel_skills.csv'
            f = open(f_name,mode='w')
            f.write(string)
            f.close()

    def skills_csv(self):
        string = '#,P1,Skill,Uncertainty\n'
        sorted_players = sort_according_to(self.skills, self.players)
        counter = 1
        for p in sorted_players:
            mu = p.ep_mu if self.algo == 'EP' else p.mc_mu
            sigma = p.ep_sigma if self.algo == 'EP' else p.mc_sigma
            string += '%d,%s,%.2f,%.2f\n' %(counter, p.name, mu, sigma)
            counter += 1

        if not os.path.exists(self.csv_path):
            os.makedirs(self.csv_path)

        f_name = self.csv_path + '3_skills.csv'
        f = open(f_name,mode='w')
        f.write(string)
        f.close()

    def predictions_csv(self):
        if self.test_matches is not None:
            string = '#,P1,P2,Result,P(Win P1),P(Loss P1),P(Draw),Prediction,Correct\n'
            counter = 1
            for m in self.test_matches:
                w, l, d, expect, correct = m.wld_ep if self.algo == 'EP' else m.wld_mc
                string += '%d,%s,%s,%d,%.2f,%.2f,%.2f,%d,%d\n' %(counter, m.t1.players[0].name, m.t2.players[0].name,
                                                                 m.r, w, l, d, expect, correct)
                counter += 1

            if not os.path.exists(self.csv_path):
                os.makedirs(self.csv_path)

            f_name = self.csv_path + '4_predictions.csv'
            f = open(f_name,mode='w')
            f.write(string)
            f.close()

    def correct_incorrect_csv(self):
        if self.test_matches is not None:
            string = ''
            N = len(self.test_matches)
            corr = np.zeros([N,1])
            exp = np.zeros([N,1])
            i = 0
            for m in self.test_matches:
                w, l, d, expect, correct = m.wld_ep if self.algo == 'EP' else m.wld_mc
                exp[i,0] = np.max([w,l,d])
                corr[i,0] = correct
                i += 1

            c = np.sum(corr)
            ic = N - c
            p = float(c) / N
            e = np.mean(exp)
            string = 'Correct,%d\nIncorrect,%d\nPercentage,%.3f\nMean of expected,%.3f' %(c, ic, p, e)

            if not os.path.exists(self.csv_path):
                os.makedirs(self.csv_path)

            f_name = self.csv_path + '5_correct_incorrect.csv'
            f = open(f_name,mode='w')
            f.write(string)
            f.close()

    def errors_csv(self):
        if self.reel_skills is not None:
            sorted_pl = sort_according_to(self.reel_skills, self.players)
            sorted_s = sort_according_to(self.reel_skills, self.skills)
            sorted_rs = sort_according_to(self.reel_skills, self.reel_skills)
            err_1 = sorted_rs - sorted_s

            s_ranks = find_ranks(sorted_s)+1
            rs_ranks = find_ranks(sorted_rs)+1
            err_2 = rs_ranks - s_ranks

            string = 'Player,Reel Skill,Skill,Skill Error,RS Rank,Skill Rank,Rank Error\n'
            for i in range(self.num_players):

                string += '%s,%.2f,%.2f,%0.2f,%d,%d,%d\n' %(sorted_pl[i].name, sorted_rs[i], sorted_s[i], err_1[i],
                                                          rs_ranks[0,i], s_ranks[0,i], err_2[0,i])

            if not os.path.exists(self.csv_path):
                os.makedirs(self.csv_path)

            f_name = self.csv_path + '6_errors.csv'
            f = open(f_name,mode='w')
            f.write(string)
            f.close()


            err_1 = np.abs(err_1)
            err_2 = np.abs(err_2)

            string = 'Mean Skill Error,%.2f\nStd Skill Error,%.2f\n' \
                     'Mean Rank Error,%.2f\nStd Rank Error,%.2f' % (np.mean(err_1),
                                                                    np.std(err_1),
                                                                    np.mean(err_2),
                                                                    np.std(err_2))
            f_name = self.csv_path + '7_errors.csv'
            f = open(f_name,mode='w')
            f.write(string)
            f.close()

    def trigger_csv(self):
        self.signature_csv()
        self.reel_skills_csv()
        self.skills_csv()
        self.predictions_csv()
        self.correct_incorrect_csv()
        self.errors_csv()


    def skill_distributions_plot(self):
        f = plt.figure(figsize=(18.0, 9.0))
        ax = f.add_subplot(111)
        ax.set_xlim([0,50])
        ax.set_title(self.algo)
        ax.set_xlabel('Skill')
        ax.set_ylabel('Probability')
        textstr = 'Iterations = %s\nProposal Deviation = %s\nAccepted = %s, Rejected = %s'%(str(self.num_ite),
                                                                                            str(self.proposal_variance),
                                                                                            str(self.proposals_accepted),
                                                                                            str(self.proposals_rejected))

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.7)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
                            verticalalignment='top', bbox=props)

        sorted_players = sort_according_to(self.skills, self.players)
        sorted_players_true = sorted_players
        if self.num_players >= 10: sorted_players = np.append(sorted_players[:5], sorted_players[-5:])

        sorted_players_true = sorted_players_true.tolist()

        def rank(p):
            return sorted_players_true.index(p) + 1

        burnin = 10
        for p, col in zip(sorted_players,colors):
            if self.algo == 'EP':
                x = np.arange(0,50,0.2)
                y = mlab.normpdf(x, p.ep_mu, p.ep_sigma)
                ax.plot(x,y,color=col,label='%d. %s (%.2f)'%(rank(p), p.name, p.ep_mu))
                ax.fill_between(x, 0, y, facecolor=col, alpha=0.1)
            else:
                sns.kdeplot(np.array(p.mc_sample_list)[burnin:],shade=True,color=col,ax=ax,label='%d. %s (%.2f)'%(rank(p), p.name, p.mc_mu))

            if self.reel_skills is not None:
                ax.axvline(p.reel_skill,color=col,ls='dashed')
        ax.legend()

        if not os.path.exists(self.plot_path):
                os.makedirs(self.plot_path)

        fname = self.plot_path + 'skill_distribution'
        plt.savefig(fname)



    def samples_joint_plot(self):
        if self.num_players == 2 and not self.algo == 'EP' :

            p1 = self.players[0]
            p2 = self.players[1]
            d1 = np.array(p1.mc_sample_list)
            d2 = np.array(p2.mc_sample_list)
            data = np.vstack([d1,d2]).T
            df = pd.DataFrame(data, columns=[p1.name, p2.name])
            g = sns.jointplot(x=p1.name, y=p2.name, data=df, kind="kde", size=12)
            g.plot_joint(plt.scatter, c="w", s=30, linewidth=0.5, marker=".", color='k')
            g.ax_joint.collections[0].set_alpha(0)
            g.ax_joint.set_ylim([0,50])
            g.ax_joint.set_xlim([0,50])
            g.set_axis_labels(p1.name, p2.name)
            plt.plot(d1[:100],d2[:100],'-',color='r',alpha=0.6)

            if not os.path.exists(self.plot_path):
                os.makedirs(self.plot_path)

            fname = self.plot_path + 'joint_scatter_plot'
            plt.savefig(fname)


    def trigger_plots(self):
        self.skill_distributions_plot()
        self.samples_joint_plot()




#
# ### PREPARE AXES
# ### ============
# fig1 = plt.figure(figsize=(16.0, 10.0))
# ax_metro = fig1.add_subplot(212)
# ax_metro.set_title('Based on Metropolis Hastings')
# ax_metro.set_xlabel('Skill')
# ax_metro.set_ylabel('Probability')
#
# ax_gibbs = fig1.add_subplot(211)
# ax_gibbs.set_title('Based on Gibbs Sampler')
# ax_gibbs.set_xlabel('Skill')
# ax_gibbs.set_ylabel('Probability')
#
#
#
#
# vi.printReelSkill(p,sorted=True)
# current_time = time.ctime().replace(' ','_')
#
# ### EXECUTE MCMC
# ### ============
# np.random.seed(None)
#
# ### GIBBS
# ite_gibbs = 4000
# tic = time.time()
# mc.gibbs_mcmc(p, m, ite_gibbs)
# toc_gibbs = time.time() - tic
# print 'GIBBS time: ', int(toc_gibbs), ' seconds'
# vi.printMeans(p,sorted=True)
# vi.plot_est(p, ax_gibbs, eco=True)
# pred, probs, cor, test_text = ts.compare(data_test,p)
# vi.log('../logs/reelskill.log', current_time, 'Football', m, p, t, ite_gibbs, toc_gibbs, 'Gibbs', test_text=test_text)
#
# ## Fancy up plot
# textstr = 'Iterations = %d'%(ite_gibbs)
# props = dict(boxstyle='round', facecolor='wheat', alpha=0.7)
# ax_gibbs.text(0.05, 0.95, textstr, transform=ax_gibbs.transAxes, fontsize=14,
#                             verticalalignment='top', bbox=props)
#
#
# ### METROPOLIS-HASTINGS
# ite_metro = 5000
# agr = 0.8
# tic = time.time()
# decisions = mc.mh_mcmc(p, m, agr, ite_metro)
# toc_metro = time.time() - tic
# accepted = np.sum(decisions)
# rejected = ite_metro - np.sum(decisions)
# print 'METRO time: ', int(toc_metro), ' seconds'
# print 'Accepted: ', accepted, ' -- Rejected: ', rejected
# vi.printMeans(p,sorted=True)
# vi.plot_est(p, ax_metro, eco=True)
# pred, probs, cor, test_text = ts.compare(data_test,p)
# vi.log('../logs/reelskill.log', current_time, 'Football', m, p, t, ite_metro, toc_metro, 'Metropolis', test_text=test_text, aggr=agr,  accepted=accepted, rejected=rejected, br=True)
#
#
# ## Fancy up plot
# textstr = 'Iterations = %d\nProposal Deviation = %.2f\nAccepted = %d, Rejected = %d'%(ite_metro, agr, accepted, rejected)
# props = dict(boxstyle='round', facecolor='wheat', alpha=0.7)
# ax_metro.text(0.05, 0.95, textstr, transform=ax_metro.transAxes, fontsize=14,
#                             verticalalignment='top', bbox=props)
#
#
# ## Save figure
# fname = '../img/' + current_time
# plt.savefig(fname)
# plt.show(block=True)










def distill_reel_skills(players):
    rs = []
    for p in players:
        crs = p.reel_skill
        if crs is not None:
            rs.append(crs)
        else:
            return None
    return rs



def distill_skills(algo, players, reel_skills):
    s = np.array([])
    u = np.array([])
    se = np.array([])
    re = np.array([])
    check = algo == 'EP'
    for p in players:
        cs = p.ep_mu if check else p.mc_mu
        cu = p.ep_sigma if check else p.mc_sigma
        s = np.append(s, cs)
        u = np.append(u, cu)
    if reel_skills is not None:
        se = s - reel_skills
        r1 = find_ranks(s)
        r2 = find_ranks(reel_skills)
        re = r1 - r2
    else:
        (se, re) = (None, None)
    return s, u, se, re



def find_ranks(values):
    N = len(values)
    rv = np.zeros([1,N])
    sorted = np.argsort(values)[::-1]
    for i, s in zip(xrange(N), sorted):
        rv[0,s] = i
    return rv


def sort_according_to(values, players, descending=True):
    indexes = np.argsort(values)
    temp = np.array(players)
    sorted = temp[indexes]
    if descending:
        return sorted[::-1]
    else:
        return sorted











