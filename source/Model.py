# -*- coding: utf-8 -*-
""" This module contains model classes and assumed model functions """

import numpy as np
from faker import Faker
from scipy import stats
from ep import beta,draw_probability

fake = Faker()
fake.seed(100)

draw_factor = 0.33

# TODO test this out
def win_lose_draw_ep(player, opponent):
    delta_mu = player.ep_mu - opponent.ep_mu
    denom = np.sqrt(2 * (beta * beta) + pow(player.ep_sigma, 2) + pow(opponent.ep_sigma, 2))
    w = stats.norm.cdf(delta_mu / denom)
    d = draw_probability
    l = 1 - w - d
    return w, l, d

def sigmoid(x):
    c1 = 0.06
    return 1 / (1 + np.e ** (-c1 * x));  # x=50 --> 0.95 prob. of win


def expo(x):
    global draw_factor
    c1 = draw_factor
    c2 = 0.05
    return c1 * np.exp(-1 * c2 * x)


def bi_expo(x, mean=0):
    x = np.array(x)
    x -= mean
    return expo(np.abs(x))


def multinomial_log(N, logp):
    log_rand = -np.random.exponential(size=N)
    logp_cuml = np.logaddexp.accumulate(np.hstack([[-np.inf], logp]))
    logp_cuml -= logp_cuml[-1]
    return np.histogram(log_rand, bins=logp_cuml)[0]


def win_lose_draw(x):
    d = expo(abs(x))
    w = (1-d) * sigmoid(x)
    l = (1-d) - w
    return w,l,d


class Player:
    players_all = []
    count_player = 101

    @staticmethod
    def find_player(name):
        for p in Player.players_all:
            if p.name == name: return p
        return None

    # use this method to construct a player object
    @staticmethod
    def with_name(name=None, reel_skill=None):
        if name is None:
            name = fake.name()
            return Player(name, reel_skill)
        else:
            pl = Player.find_player(name)
            rv = pl if pl is not None else Player(name, reel_skill)
            return rv

    def __init__(self, name=None, reel_skill=None):
        self.id = Player.count_player
        self.name = fake.name() if name is None else name
        self.reel_skill = reel_skill

        self.mc_sample_list = None
        self.mc_sample_list_new = None
        self.mc_sample = None
        self.mc_sample_new = None
        self.mc_kernel = None
        # self.mc_mu = None  # @property
        # self.mc_sigma = None  # @property

        self.ep_mu = 25.0  # default for trueskill
        self.ep_sigma = self.ep_mu / 3

        self.reset_samples()  # init sample related stuff
        self.update_kernel()  # init kde

        Player.count_player += 1

        Player.players_all.append(self)

    @property
    def mc_mu(self):
        return np.mean(self.mc_sample_list)

    @property
    def mc_sigma(self):
        return np.std(self.mc_sample_list)

    def mc_prob(self, x):
        return self.mc_kernel.evaluate(x)

    def reset_samples(self, random=False):
        self.mc_sample_list = np.random.rand(1000)*50  # initial : uniform dist
        self.mc_sample_list_new = []
        self.mc_sample = 25 if not random else np.random.rand()*50
        self.mc_sample_new = 25 if not random else np.random.rand()*50

    def update_samples(self):
        self.mc_sample_list_new = self.mc_sample_list

    def update_kernel(self):
        self.mc_kernel = stats.gaussian_kde(self.mc_sample_list)

    def __str__(self):
        return '%s - %d - %s - MC{%.2f,%.2f} - EP{%.2f,%.2f}' %(self.name, self.id, str(self.reel_skill), self.mc_mu, self.mc_sigma, self.ep_mu, self.ep_sigma)

    def __repr__(self):
        return '%s(%d)' %(self.name, self.id)

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.name == other.name

    def __ne__(self, other):
        return not self.__eq__(other)


class Team:
    teams_all = []
    team_count = 101

    @staticmethod
    def find_team(members):
        for t in Team.teams_all:
            if t.players == members: return t
        return None

    # use this method to construct a team object
    @staticmethod
    def with_players(players=[]):
        # notice no duplicate check in the players list!
        if players == []: raise ValueError('Team cannot be blank')
        tm = Team.find_team(players)
        if tm is not None:
            return tm
        else:
            return Team(players)

    def __init__(self, players):
        self.id = Team.team_count
        self.players = players
        # self.size  # @property
        # self.reel_skill  # @property
        # self.mc_sample  # @property
        # self.mc_sample_new  # @property

        Team.team_count += 1
        Team.teams_all.append(self)

    @property
    def size(self):
        return len(self.players)

    @property
    def reel_skill(self):
        rs = 0
        for p in self.players:
            rs += p.reel_skill
        return rs

    @property
    def mc_sample(self):
        rv = 0
        for p in self.players:
            rv += p.mc_sample
        return rv

    @property
    def mc_sample_new(self):
        rv = 0
        for p in self.players:
            rv += p.mc_sample_new
        return rv

    @property
    def mc_mu(self):
        rv = 0
        for p in self.players:
            rv += p.mc_mu
        return rv

    @property
    def mc_sigma(self):
        rv = 0
        for p in self.players:
            rv += p.mc_sigma
        return rv

    @property
    def ep_mu(self):
        rv = 0
        for p in self.players:
            rv += p.ep_mu
        return rv

    @property
    def ep_sigma(self):
        rv = 0
        for p in self.players:
            rv += p.ep_sigma
        return rv

    def add_player(self, player):
        self.players.append(player)

    def __str__(self):
        return repr(self)

    def __repr__(self):
        s = '<<'
        for p in self.players:
            s += repr(p) + '-'
        return s+'>>'

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.players == other.players

    def __ne__(self, other):
        return not self.__eq__(other)


class Versus:
    def __init__(self, t1, t2, r):
        self.t1 = t1
        self.t2 = t2
        self.r = r  # -1,0,1

    # assume single player teams
    @property
    def wld_ep(self):
        w, l, d = win_lose_draw_ep(self.t1.players[0], self.t2.players[0])
        r = [1, -1, 0]
        index = np.argsort([w,l,d])[::-1][0]
        expected = r[index]
        return w, l, d, expected, int(expected == self.r)

    @property
    def wld_mc(self):
        w, l, d = win_lose_draw(self.t1.mc_mu - self.t2.mc_mu)
        r = [1, -1, 0]
        index = np.argsort([w,l,d])[::-1][0]
        expected = r[index]
        return w, l, d, expected, int(expected == self.r)

    def prob(self, x):  # x --> pl1 - pl2 (order matters) (always)
        w, l, d = win_lose_draw(x)
        if self.r == 1:
            return w
        elif self.r == -1:
            return l
        else:
            return d

    def report_of_player(self, player):
        #(Involved, W/L/D, Team#)
        if player in self.t1.players: return True, self.r, 1
        if player in self.t2.players: return True, -1*self.r, 2
        else: return False, -13, -13

def empty():
    Team.teams_all = []
    Player.players_all = []





# TODO check if i'll need these methods

# @staticmethod
# def produce_vs(standings):
#     rv = []
#     for s in standings:
#         teams_ordered = Versus.ordered(s)
#         for i in range(1,len(teams_ordered)):
#             t1 = teams_ordered[i-1]
#             t2 = teams_ordered[i]
#             r = (-1 * (s[t1] - s[t2]))
#             rv.append(Versus(t1,t2,r))
#     return rv
#
#
# def keyByValue(a_dict, value):
#     rv = []
#     for key in a_dict.keys():
#         if a_dict[key] == value: rv.append(key)
#     return rv
#
#
# def keysOrdered(a_dict):
#     rv = []
#     current_rank = 1
#     current_keys = Versus.findByValue(a_dict,current_rank)
#     while current_keys != []:
#         rv += current_keys
#         current_rank += 1
#         current_keys = Versus.findByValue(a_dict,current_rank)
#     return rv
#
