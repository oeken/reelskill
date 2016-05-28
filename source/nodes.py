# -*- coding: utf-8 -*-

import numpy as np
from enum import Enum

Kind = Enum('Skill', 'Perf', 'Sum', 'Diff')
epsilon = 0.1  # draw margin
beta = 25.0 / 3 / 2  # sigma div 2
dynamic = 25.0 / 3 / 100  # sigma div 100


# MESSAGE
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
    def __init__(self, mu, sigma, skill_var):
        self.skill_var = skill_var
        self.mu = mu
        self.sigma = sigma

    def message_down(self):
        new_pi, new_tau = self.calculate_update()
        new_value = Gaussian.with_precision(new_pi, new_tau)
        self.skill_var.set_value(new_value)

    def calculate_update(self):
        sigma = np.sqrt(self.sigma ** 2 + dynamic ** 2)
        new_pi = self.skill_var.value.pi + (sigma ** -2)
        new_tau = self.skill_var.value.tau + (self.mu * self.sigma ** -2)
        return new_pi, new_tau
        # TODO test it and dont forget to add dynamics



class PerfFactor:
    def __init__(self, skill_var, perf_var):
        self.skill_var = skill_var
        self.perf_var = perf_var
        self.mu = None
        self.sigma = beta

    def calculate_a(self, dir):
        if dir == 'up':
            return (1 + (self.sigma ** 2) * (self.perf_var.value.pi)) ** -1
        else:
            return (1 + (self.sigma ** 2) * (self.skill_var.value.pi)) ** -1

    def calculate_update(self, dir):
        a = self.calculate_a(dir)
        if dir == 'up':
            new_pi = a * self.perf_var.value.pi
            new_tau = a * self.perf_var.value.tau
        else:
            new_pi = a * self.skill_var.value.pi
            new_tau = a * self.skill_var.value.tau
        return new_pi, new_tau
        # TODO test it out

    def message_up(self):
        new_pi, new_tau = self.calculate_update('up')
        new_value = Gaussian.with_precision(new_pi, new_tau)
        self.skill_var.set_value(new_value)

    def message_down(self):
        new_pi, new_tau = self.calculate_update('down')
        new_value = Gaussian.with_precision(new_pi, new_tau)
        self.perf_var.set_value(new_value)



class SumFactor:
    def __init__(self, perf_vars, sum_var):
        self.perf_vars = perf_vars
        self.sum_var = sum_var
        self.all_vars = perf_vars + [sum_var]

    def get_values(self, indexes):
        N = len(indexes)
        pis = np.zeros([N,1])
        taus = np.zeros([N,1])
        for i, index in zip(xrange(N), indexes):
            pis[i] = self.all_vars[index].value.pi
            taus[i] = self.all_vars[index].value.tau
        return pis, taus

    def update_helper(self, indexes, coeffs):
        pis, taus = self.get_values(indexes)
        new_pi = np.sum(((coeffs ** 2) / pis)) ** -1
        new_tau = new_pi * np.sum(coeffs * taus / pis)
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
            # TODO test it out

    def message_up(self):
        N = len(self.perf_vars)
        for i in xrange(N):
            new_pi, new_tau = self.calculate_update(i)
            new_value = Gaussian.with_precision(new_pi, new_tau)
            self.perf_vars[i].set_value(new_value)

    def message_down(self):
        N = len(self.perf_vars)
        new_pi, new_tau = self.calculate_update(N)
        new_value = Gaussian.with_precision(new_pi, new_tau)
        self.sum_var.set_value(new_value)



class DiffFactor:
    def __init__(self, left_sum_var, right_sum_var, diff_var):
        self.sum_vars = [left_sum_var, right_sum_var]
        self.diff_var = diff_var
        self.all_vars = [left_sum_var, right_sum_var, diff_var]

    def get_values(self, indexes):
        N = len(indexes)
        pis = np.zeros([1,N])
        taus = np.zeros([1,N])
        for i, index in zip(xrange(N), indexes):
            pis[0,i] = self.all_vars[index].value.pi
            taus[0,i] = self.all_vars[index].value.tau
        return pis, taus

    def update_helper(self, indexes, coeffs):
        pis, taus = self.get_values(indexes)
        new_pi = np.sum(((coeffs ** 2) / pis)) ** -1
        new_tau = new_pi * np.sum(coeffs * taus / pis)
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
        # TODO test it out

    def message_up(self, target = None):
        if target is None:
            self.message_up(0)
            self.message_up(1)
        else:
            new_pi, new_tau = self.calculate_update(target)
            new_value = Gaussian.with_precision(new_pi, new_tau)
            self.sum_vars[target].set_value(new_value)

    def message_down(self):
        new_pi, new_tau = self.calculate_update(2)
        new_value = Gaussian.with_precision(new_pi, new_tau)
        self.diff_var.set_value(new_value)


class ResultFactor:
    def __init__(self, diff_var, v_fn, w_fn):
        self.diff_var = diff_var
        self.v_fn = v_fn
        self.w_fn = w_fn

    def calculate_update(self):
        c = self.diff_var.value.pi
        d = self.diff_var.value.tau
        arg1 = d / np.sqrt(c)
        arg2 = epsilon * np.sqrt(c)
        new_pi = c * (1 - self.w_fn(arg1, arg2)) ** -1
        new_tau = (d + np.sqrt(c) * self.v_fn(arg1, arg2)) * (1 - self.w_fn(d / np.sqrt(c), epsilon * np.sqrt(c))) ** -1
        return new_pi, new_tau
        # TODO test it out, implement v and w functions

    def message_up(self):
        new_pi, new_tau = self.calculate_update()
        new_value = Gaussian.with_precision(new_pi, new_tau)
        self.diff_var.set_value(new_value)


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

    def set_value(self, new_value):
        self.old_value = self.value
        self.value = new_value

    def change(self):
        t1 = np.abs(self.value.pi - self.old_value.pi)
        t2 = np.abs(self.value.tau - self.old_value.tau)
        return max(t1, t2)


