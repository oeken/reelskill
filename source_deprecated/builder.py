# -*- coding: utf-8 -*-

import source.ep as n
import numpy as np
from scipy.stats import norm
from enum import Enum
Kind = Enum('Skill', 'Perf', 'Sum', 'Diff')


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


def build_factor_graph(versus):
    draw_probability = 0.1
    mu = 25.0
    sigma = mu / 3
    beta = sigma / 2
    dynamic = sigma / 100
    epsilon = draw_margin(draw_probability,len(versus.t1.players)*2, beta)

    wt = versus.t1 if versus.r == 1 or versus.r == 0 else versus.t2
    lt = versus.t2 if versus.r == 1 or versus.r == 0 else versus.t1
    layer_v_diff = [n.Variable(Kind.Diff)]  ## 2
    if versus.r == 0:
        layer_f_result = [n.ResultFactor(layer_v_diff[0], v_draw, w_draw, epsilon)]
    else:
        layer_f_result = [n.ResultFactor(layer_v_diff[0], v_win, w_win, epsilon)]  ## 1

    layer_v_sum = [n.Variable(Kind.Sum), n.Variable(Kind.Sum)]  ## 4

    layer_f_diff = [n.DiffFactor(layer_v_sum[0], layer_v_sum[1], layer_v_diff[0])]  ## 3

    layer_v_perf = [[], []]  ## 6
    for pl in wt.players:
        layer_v_perf[0].append(n.Variable(Kind.Perf))
    for pl in lt.players:
        layer_v_perf[1].append(n.Variable(Kind.Perf))

    layer_f_sum = [n.SumFactor(layer_v_perf[0],layer_v_sum[0]), n.SumFactor(layer_v_perf[1], layer_v_sum[1])]  ## 5

    layer_v_skill = [[], []]  ## 8
    for pl in wt.players:
        layer_v_skill[0].append(n.Variable(Kind.Skill))
    for pl in lt.players:
        layer_v_skill[1].append(n.Variable(Kind.Skill))

    layer_f_perf = [[], []]  ## 7
    for i in xrange(len(wt.players)):
        layer_f_perf[0].append(n.PerfFactor(layer_v_skill[0][i], layer_v_perf[0][i], beta))
    for i in xrange(len(lt.players)):
        layer_f_perf[1].append(n.PerfFactor(layer_v_skill[1][i], layer_v_perf[1][i], beta))

    layer_f_skill = [[], []]  ## 9
    for i in xrange(len(wt.players)):
        layer_f_skill[0].append(n.SkillFactor(wt.players[i].ts_mu, wt.players[i].ts_sigma, layer_v_skill[0][i], dynamic))
    for i in xrange(len(lt.players)):
        layer_f_skill[1].append(n.SkillFactor(lt.players[i].ts_mu, lt.players[i].ts_sigma, layer_v_skill[1][i], dynamic))

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
