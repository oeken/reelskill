# -*- coding: utf-8 -*-
import numpy as np

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
        # TODO see if m_pi has correct value

    def calculate_update(self, message):
        a = self.calculate_a(message)
        new_pi = a * message.pi
        new_tau = a * message.tau
        return new_pi, new_tau
        # TODO see if m_pi m_tau has correct values

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
    # TODO see if m_pis etc are fetched correctly

    def update_helper(self, indexes, coeffs):
        v_pis, v_taus = self.get_values(indexes)
        new_pi = np.sum(((coeffs ** 2) / v_pis)) ** -1
        new_tau = new_pi * np.sum(coeffs * v_taus / v_pis)
        return new_pi, new_tau
    # TODO see if everything (calculation) is fine

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


