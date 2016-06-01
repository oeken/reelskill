# -*- coding: utf-8 -*-
""" This module is used to run ep algorithm on the built factor graph """

import numpy as np
from scipy.stats import norm
from enum import Enum
Kind = Enum('Skill', 'Perf', 'Sum', 'Diff')

draw_probability = 0.1
mu = 25.0
sigma = mu / 3
beta = sigma / 2
dynamic = sigma / 100


def draw_margin(draw_probability, size, beta):
    return norm.ppf((draw_probability + 1) / 2.) * np.sqrt(size) * beta


def v_win(x, epsilon):
    return norm.pdf(x-epsilon) / norm.cdf(x-epsilon)


def v_draw(x, epsilon):
    nom = norm.pdf(-epsilon-x) - norm.pdf(epsilon - x)
    denom = norm.cdf(epsilon-x) - norm.cdf(-epsilon-x)
    return nom / denom


def w_win(x, epsilon):
    return v_win(x, epsilon) * (v_win(x, epsilon) + x - epsilon)


def w_draw(x, epsilon):
    term = v_draw(x, epsilon) ** 2
    nom = (epsilon - x) * norm.pdf(epsilon - x) + (epsilon + x) * norm.pdf(epsilon + x)
    denom = norm.cdf(epsilon - x) - norm.cdf(-epsilon - x)
    return term + nom / denom


class Gaussian(object):
    def __init__(self):
        self.pi = 0
        self.tau = 0

    @staticmethod
    def with_standard(mu, sigma):
        rv = Gaussian()
        rv.pi = rv.calc_pi(sigma)
        rv.tau = rv.calc_tau(rv.pi, mu)
        return rv

    @staticmethod
    def with_precision(pi, tau):
        rv = Gaussian()
        rv.pi = pi
        rv.tau = tau
        return rv

    @property
    def sigma(self):
        return self.pi ** (-0.5) if self.pi != 0 else np.inf

    @property
    def mu(self):
        return self.tau / self.pi if self.pi != 0 else 0

    @staticmethod
    def calc_pi(sigma):
        return sigma ** -2

    @staticmethod
    def calc_tau(pi, mu):
        return pi * mu

    def __mul__(self, other):
        new_pi = self.pi + other.pi
        new_tau = self.tau + other.tau
        return Gaussian.with_precision(new_pi, new_tau)

    def __truediv__(self, other):
        new_pi = self.pi - other.pi
        new_tau = self.tau - other.tau
        return Gaussian.with_precision(new_pi, new_tau)

    __div__ = __truediv__

    def __str__(self):
        return 'mu= %.2f sigma= %.2f || pi= %.2f tau= %.2f' %(self.mu, self.sigma, self.pi, self.tau)

    def __repr__(self):
        return self.__str__()



# FACTOR NODES
class SkillFactor:
    def __init__(self, mu, sigma, skill_var, dynamic):
        self.skill_var = skill_var
        self.mu = mu
        self.sigma = sigma
        self.dynamic = dynamic

    def message_down(self):
        new_pi, new_tau = self.calculate_update()
        new_value = Gaussian.with_precision(new_pi, new_tau)
        self.skill_var.set_value_one(self, new_value)

    def calculate_update(self):
        sigma = np.sqrt(self.sigma ** 2 + self.dynamic ** 2)
        new_pi = self.skill_var.value.pi + (sigma ** -2)
        new_tau = self.skill_var.value.tau + (self.mu * self.sigma ** -2)
        return new_pi, new_tau


class PerfFactor:
    def __init__(self, skill_var, perf_var, beta):
        self.skill_var = skill_var
        self.perf_var = perf_var
        self.sigma = beta

    def calculate_a(self, message):
        m_pi = message.pi
        return (1 + (self.sigma ** 2) *  m_pi) ** -1

    def calculate_update(self, message):
        a = self.calculate_a(message)
        new_pi = a * message.pi
        new_tau = a * message.tau
        return new_pi, new_tau

    def message_up(self):
        m = self.perf_var.message_from(self)
        new_pi, new_tau = self.calculate_update(self.perf_var.value / m)
        new_value = Gaussian.with_precision(new_pi, new_tau)
        self.skill_var.set_value_two(self, new_value)

    def message_down(self):
        m = self.skill_var.message_from(self)
        new_pi, new_tau = self.calculate_update(self.skill_var.value / m)
        new_value = Gaussian.with_precision(new_pi, new_tau)
        self.perf_var.set_value_two(self, new_value)



class SumFactor:
    def __init__(self, perf_vars, sum_var):
        self.perf_vars = perf_vars
        self.sum_var = sum_var
        self.all_vars = perf_vars + [sum_var]

    def get_values(self, indexes):
        N = len(indexes)
        v_pis = np.zeros([N,1])
        v_taus = np.zeros([N,1])
        for i, index in zip(xrange(N), indexes):
            v = self.all_vars[index].value / self.all_vars[index].message_from(self)
            v_pis[i], v_taus[i] = v.pi, v.tau
        return v_pis, v_taus


    def update_helper(self, indexes, coeffs):
        v_pis, v_taus = self.get_values(indexes)
        new_pi = np.sum(((coeffs ** 2) / v_pis)) ** -1
        new_tau = new_pi * np.sum(coeffs * v_taus / v_pis)
        return new_pi, new_tau


    def calculate_update(self, target):
            N = len(self.perf_vars)
            if target == N:  # its DOWN
                indexes = np.arange(N)
                coeffs = np.ones([N, 1])
                return self.update_helper(indexes, coeffs)
            else:  # its UP
                indexes = np.append(np.arange(target), np.arange(target+1,N+1))
                coeffs = np.zeros([N,1]) * -1
                coeffs[-1] = 1
                return self.update_helper(indexes, coeffs)

    def message_up(self):
        N = len(self.perf_vars)
        for i in xrange(N):
            new_pi, new_tau = self.calculate_update(i)
            new_value = Gaussian.with_precision(new_pi, new_tau)
            self.perf_vars[i].set_value_two(self, new_value)

    def message_down(self):
        N = len(self.perf_vars)
        new_pi, new_tau = self.calculate_update(N)
        new_value = Gaussian.with_precision(new_pi, new_tau)
        self.sum_var.set_value_two(self, new_value)



class DiffFactor:
    def __init__(self, left_sum_var, right_sum_var, diff_var):
        self.sum_vars = [left_sum_var, right_sum_var]
        self.diff_var = diff_var
        self.all_vars = [left_sum_var, right_sum_var, diff_var]

    def get_values(self, indexes):
        N = len(indexes)
        v_pis = np.zeros([1,N])
        v_taus = np.zeros([1,N])
        for i, index in zip(xrange(N), indexes):
            v = self.all_vars[index].value / self.all_vars[index].message_from(self)
            v_pis[0,i] ,v_taus[0,i] = v.pi, v.tau
        return v_pis, v_taus

    def update_helper(self, indexes, coeffs):
        v_pis, v_taus = self.get_values(indexes)
        new_pi = np.sum(((coeffs ** 2) / v_pis)) ** -1
        new_tau = new_pi * np.sum(coeffs * v_taus / v_pis)
        return new_pi, new_tau

    def calculate_update(self, target):  # 0 is left, 1 is right, 2 is down
        if target == 0:
            indexes = np.array([1,2])
            coeffs = np.array([1,1])
        elif target == 1:
            indexes = np.array([0,2])
            coeffs = np.array([1,-1])
        else:
            indexes = np.array([0,1])
            coeffs = np.array([1,-1])
        return self.update_helper(indexes, coeffs)

    def message_up(self, target = None):
        if target is None:
            self.message_up(0)
            self.message_up(1)
        else:
            new_pi, new_tau = self.calculate_update(target)
            new_value = Gaussian.with_precision(new_pi, new_tau)
            self.sum_vars[target].set_value_two(self, new_value)

    def message_down(self):
        new_pi, new_tau = self.calculate_update(2)
        new_value = Gaussian.with_precision(new_pi, new_tau)
        self.diff_var.set_value_two(self, new_value)


class ResultFactor:
    def __init__(self, diff_var, v_fn, w_fn, epsilon):
        self.diff_var = diff_var
        self.v_fn = v_fn
        self.w_fn = w_fn
        self.epsilon = epsilon

    def calculate_update(self):
        diff = self.diff_var.value / self.diff_var.message_from(self)
        c,d = diff.pi, diff.tau
        arg1 = d / np.sqrt(c)
        arg2 = self.epsilon * np.sqrt(c)
        v_val = self.v_fn(arg1, arg2)
        w_val = self.w_fn(arg1, arg2)
        new_pi = c / (1 - w_val)
        new_tau = (d + np.sqrt(c) * v_val) / (1 - w_val)
        return new_pi, new_tau

    def message_up(self):
        new_pi, new_tau = self.calculate_update()
        new_value = Gaussian.with_precision(new_pi, new_tau)
        self.diff_var.set_value_one(self, new_value)


# VARIABLE NODES
class Variable:
    def __init__(self, kind):
        self.kind = kind
        self.value = Gaussian()
        self.old_value = self.value
        self.messages = dict()

    def __str__(self):
        return 'mu= %.2f sigma= %.2f || pi= %.2f tau= %.2f' %(self.value.mu, self.value.sigma, self.value.pi, self.value.tau)

    def __repr__(self):
        return self.__str__()

    def message_from(self, factor):
        try:
            message = self.messages[factor]
        except KeyError:
            self.messages[factor] = Gaussian()
            message = self.messages[factor]
        return message

    def set_value(self, value):
        self.old_value = self.value
        self.value = value

    def set_value_one(self, factor, message):  # used by skill and result (terminal) factors
        prev_message = self.message_from(factor)
        self.messages[factor] = message * prev_message / self.value
        self.set_value(message)

    def set_value_two(self, factor, message):  # used by non-terminal factors
        prev_message = self.message_from(factor)
        self.messages[factor] = message
        self.set_value(message * self.value / prev_message)  # take out prev message plug in the new message

    def change(self):
        t1 = np.abs(self.value.pi - self.old_value.pi)
        t2 = np.abs(self.value.tau - self.old_value.tau)
        return max(t1, t2)


def build_factor_graph(versus):
    epsilon = draw_margin(draw_probability,len(versus.t1.players)*2, beta)

    wt = versus.t1 if versus.r == 1 or versus.r == 0 else versus.t2
    lt = versus.t2 if versus.r == 1 or versus.r == 0 else versus.t1
    layer_v_diff = [Variable(Kind.Diff)]  ## 2
    if versus.r == 0:
        layer_f_result = [ResultFactor(layer_v_diff[0], v_draw, w_draw, epsilon)]
    else:
        layer_f_result = [ResultFactor(layer_v_diff[0], v_win, w_win, epsilon)]  ## 1

    layer_v_sum = [Variable(Kind.Sum), Variable(Kind.Sum)]  ## 4

    layer_f_diff = [DiffFactor(layer_v_sum[0], layer_v_sum[1], layer_v_diff[0])]  ## 3

    layer_v_perf = [[], []]  ## 6
    for pl in wt.players:
        layer_v_perf[0].append(Variable(Kind.Perf))
    for pl in lt.players:
        layer_v_perf[1].append(Variable(Kind.Perf))

    layer_f_sum = [SumFactor(layer_v_perf[0],layer_v_sum[0]), SumFactor(layer_v_perf[1], layer_v_sum[1])]  ## 5

    layer_v_skill = [[], []]  ## 8
    for pl in wt.players:
        layer_v_skill[0].append(Variable(Kind.Skill))
    for pl in lt.players:
        layer_v_skill[1].append(Variable(Kind.Skill))

    layer_f_perf = [[], []]  ## 7
    for i in xrange(len(wt.players)):
        layer_f_perf[0].append(PerfFactor(layer_v_skill[0][i], layer_v_perf[0][i], beta))
    for i in xrange(len(lt.players)):
        layer_f_perf[1].append(PerfFactor(layer_v_skill[1][i], layer_v_perf[1][i], beta))

    layer_f_skill = [[], []]  ## 9
    for i in xrange(len(wt.players)):
        layer_f_skill[0].append(SkillFactor(wt.players[i].ep_mu, wt.players[i].ep_sigma, layer_v_skill[0][i], dynamic))
    for i in xrange(len(lt.players)):
        layer_f_skill[1].append(SkillFactor(lt.players[i].ep_mu, lt.players[i].ep_sigma, layer_v_skill[1][i], dynamic))

    return layer_f_skill, layer_v_skill, layer_f_perf, layer_v_perf, layer_f_sum, layer_v_sum, layer_f_diff, layer_v_diff, layer_f_result


def execute_order(layer_list):
    skill_factors = layer_list[0]
    sfs_merged = skill_factors[0] + skill_factors[1]
    for sf in sfs_merged:
        sf.message_down()

    perf_factors = layer_list[2]
    pfs_merged = perf_factors[0] + perf_factors[1]
    for pf in pfs_merged:
        pf.message_down()

    sum_factors = layer_list[4]
    for sf in sum_factors:
        sf.message_down()

    diff_factor   = layer_list[6][0]
    diff_var      = layer_list[7][0]
    result_factor = layer_list[8][0]

    delta = 10
    while delta > 0.1:
        diff_factor.message_down()
        result_factor.message_up()
        delta = diff_var.change()

    diff_factor.message_up()
    for sf in sum_factors:
        sf.message_up()
    for pf in pfs_merged:
        pf.message_up()

    skill_vars = layer_list[1]
    return skill_vars


def run(matches):
    matches = permute_list(matches)
    for match in matches:
        fg = build_factor_graph(match)
        [res1, res2]= execute_order(fg)
        wt = match.t1 if match.r == 1 or match.r == 0 else match.t2
        lt = match.t2 if match.r == 1 or match.r == 0 else match.t1
        for r, pl in zip(res1, wt.players):
            pl.ep_mu = r.value.mu
            pl.ep_sigma = r.value.sigma
        for r, pl in zip(res2, lt.players):
            pl.ep_mu = r.value.mu
            pl.ep_sigma = r.value.sigma
    return 'EP'


def permute_list(list):
    N = len(list)
    indexes = np.random.permutation(N)
    rv = [None] * N
    for i in xrange(N):
        rv[i] = list[indexes[i]]
    return rv