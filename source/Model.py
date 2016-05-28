# -*- coding: utf-8 -*-
import numpy as np
from faker import Faker
from scipy import stats as st

fake = Faker()
fake.seed(100)

draw_factor = 0.33

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
    x = x - mean
    return expo(np.abs(x))

def multinomial_log(N, logp):
    log_rand = -np.random.exponential(size=N)
    logp_cuml = np.logaddexp.accumulate(np.hstack([[-np.inf], logp]))
    logp_cuml -= logp_cuml[-1]
    return np.histogram(log_rand, bins=logp_cuml)[0]



# def sample_expo(s1, s2, mean=0):
#     global draw_factor
#     c1 = draw_factor
#     e = (-1 / c1) * np.log(np.random.rand(s1,s2))
#     mirror = np.random.binomial(1,0.5,[s1,s2]) * 2 - 1
#     return e * mirror + mean

def win_lose_draw(x):
    d = expo(abs(x))
    w = (1-d) * sigmoid(x)
    l = (1-d) - w
    return w,l,d

class Player:
    pool_player = []
    count_player = 101

    def __init__(self, name='Doe', reel_skill=None):
        self.reset()
        # self.sample = 25
        # self.new_sample = 25
        # self.new_sample_list = []
        # self.sample_list = np.random.rand(1000)*50  # initial : uniform dist

        self.name = fake.name() if name == 'Doe' else name
        self.reel_skill = reel_skill
        self.id = Player.count_player

        self.updateKernel()
        Player.count_player += 1
        Player.pool_player.append(self)


        self.ts_mu = 25.0
        self.ts_sigma = 25.0 / 3

    def __str__(self):
        return self.name + ', ' + str(self.id)+', '+ str(self.reel_skill) + ', m:' + str(self.mu) + ', s:' + str(self.sigma)

    def __repr__(self):
        return self.name+'('+str(self.id)+')'

    def updateKernel(self):
        self.kernel = st.gaussian_kde(self.sample_list)

    def prob(self,x):
        return self.kernel.evaluate(x)

    def mean(self):
        return np.mean(self.sample_list)

    def reset(self, fixed=True):
        self.sample_list = np.random.rand(1000)*50  # initial : uniform dist
        self.new_sample_list = []

        self.sample = 25 if fixed else np.random.rand()*50
        self.new_sample = 25 if fixed else np.random.rand()*50



    # def __eq__(self, other):
    #     return isinstance(other, self.__class__) and self.name == other.name
    #
    # def __ne__(self, other):
    #     return not self.__eq__(other)

class Team:
    pool_team = []
    team_count = 101
    def __init__(self, players=[]):
        self.id = Team.team_count
        self.players = list() if players == [] else players
        Team.pool_team.append(self)
        Team.team_count += 1

    def addPlayer(self,player):
        self.players.append(player)

    def size(self):
        return len(self.players)

    def reel_skill(self):
        rs = 0
        for p in self.players:
            rs += p.reel_skill
        return rs

    def sample(self):
        rv = 0
        for p in self.players:
            rv += p.sample
        return rv

    def new_sample(self):
        rv = 0
        for p in self.players:
            rv += p.new_sample
        return rv


    def __str__(self):
        return repr(self)

    def __repr__(self):
        s = '<<'
        for p in self.players:
            s += repr(p) + '-'
        return s+'>>'

    # def __eq__(self, other):
    #     return isinstance(other, self.__class__) and self.players == other.players


class Versus:
    versus_all = []

    def __init__(self,t1,t2,r):
        self.t1 = t1
        self.t2 = t2
        self.r = r  # -1,0,1
        Versus.versus_all.append(self)

    def prob(self,x):  # x --> pl1 - pl2 (order matters) (always)
        # rv = sigmoid(x)
        w,l,d = win_lose_draw(x)
        if self.r == 1:
            return w
        elif self.r == -1:
            return l
        else:
            return d
            # raise ValueError('This should not happen')
            # # return DRAW_MARGIN

    def report_of_player(self,player):
        #(Involved, W/L/D, Team#)
        if player in self.t1.players: return True, self.r, 1
        if player in self.t2.players: return True, -1*self.r, 2
        else: return False, -13, -13

def produceVs(standings):
    rv = []
    for s in standings:
        teams_ordered = Versus.ordered(s)
        for i in range(1,len(teams_ordered)):
            t1 = teams_ordered[i-1]
            t2 = teams_ordered[i]
            r = (-1 * (s[t1] - s[t2]))
            rv.append(Versus(t1,t2,r))
    return rv


def keyByValue(a_dict, value):
    rv = []
    for key in a_dict.keys():
        if a_dict[key] == value: rv.append(key)
    return rv


def keysOrdered(a_dict):
    rv = []
    current_rank = 1
    current_keys = Versus.findByValue(a_dict,current_rank)
    while current_keys != []:
        rv += current_keys
        current_rank += 1
        current_keys = Versus.findByValue(a_dict,current_rank)
    return rv


#
# class Result:
#     def __init__(self, standings):
#         self.outcomes = []
#
#         if isinstance(standings, dict):
#             teams_ordered = Result.ordered(standings)
#             for i in range(1,len(teams_ordered)):
#
#
#         else:
#             raise TypeError('Invalid data type passed')
#
#     @staticmethod
#     def findByValue(dict, value):
#         rv = []
#         for key in dict.keys():
#             if dict[key] == value: rv.append(key)
#         return rv
#
#     @staticmethod
#     def ordered(a_dict):
#         rv = []
#         currentRank = 1
#         currentKeys = Result.findByValue(a_dict,currentRank)
#         while currentKeys != []:
#             rv += currentKeys
#             currentRank += 1
#             currentKeys = Result.findByValue(a_dict,currentRank)
#         return rv
#
#
#
#

#
# class Match:
#     """
#     Match class
#     """
#
#     def __init__(self, teams):
#         self.synthetic = None
#         self.standings = None
#
#         if isinstance(teams,dict):
#             for t in teams.keys():
#                 if t.isSynthetic():
#                     self.synthetic = True
#                     break
#             self.synthetic = False
#             self.standings = teams
#         elif isinstance(teams,list):
#             for t in teams:
#                 if not t.isSynthetic(): raise TypeError('Non synthetic team passed without standings')
#             self.synthetic = True
#             self.standings = Match.simulate(teams)
#         else:
#             raise TypeError('Invalid data type passed')
#
#     def __str__(self):
#         rv = '{{ '
#         cr = 1
#         lr = self.lowestRank()
#         while cr <= lr:
#             rv += str(self.teamsOnRank(cr))+'['+str(cr)+']' + '  --  '
#             cr += 1
#         return rv + ' }}'
#
#     def __repr__(self):
#         return str(self)
#
#     def players(self):
#         rv = set()
#         for t in self.teamsOrdered():
#             for p in t.players:
#                 rv.add(p)
#         return list(rv)
#
#     def teamsOrdered(self):
#         rv = []
#         cr = 1
#         lr = self.lowestRank()
#         while cr <= lr:
#             for t in self.teamsOnRank(cr): rv.append(t)
#             cr += 1
#         return rv
#
#     def lowestRank(self):
#         return max(self.standings.values())
#
#     def teamsOnRank(self, rank):
#         if rank > self.lowestRank():
#             return []   # no player with such a rank
#         else:
#             rv = []
#             for t in self.standings.keys():
#                 if self.standings[t] == rank: rv.append(t)
#             return rv
#
#     def teamCount(self):
#         return len(self.standings.keys())
#
#     def isSynthetic(self):
#         return self.synthetic
#
#     def atomicMatches(self):
#         rv = []
#         p = 1
#         while p < self.teamCount():
#             t1 = self.teamsOrdered()[p-1]
#             t2 = self.teamsOrdered()[p]
#             m = Match({t1:self.standings[t1], t2:self.standings[t2]})
#             m.synthetic = self.synthetic
#             rv.append(m)
#             p += 1
#         return rv
#
#
#
#     @staticmethod
#     def simulate(teams):
#         rv = {}
#         wincounts = np.zeros(len(teams))
#         for i in range(len(teams)):
#             for j in range(i+1,len(teams)):
#                 if Match.simulateTwoTeams(teams[i],teams[j]):
#                     wincounts[i] += 1
#                 else:
#                     wincounts[j] += 1
#         raw_standings = np.argsort(wincounts)
#         raw_standings = np.fliplr([raw_standings])[0] # reverse the array
#         rv[teams[raw_standings[0]]] = 1 # the winner (rank is 1)
#         for i in range(1,len(raw_standings)):
#             prev_wc = wincounts[raw_standings[i-1]] # wincout of previous team
#             next_wc = wincounts[raw_standings[i]] # wincout of next team
#             prev_team = teams[raw_standings[i-1]] # previous team (ranked before)
#             next_team = teams[raw_standings[i]] # previous team (ranked before)
#             # if next team has less wins then take prev teams rank add 1 and set next team's rank
#             rv[next_team] = rv[prev_team]+1 if next_wc < prev_wc else rv[prev_team]
#         return rv
#
#     @staticmethod
#     def playersInMatchList(matches):
#         """
#         :param matches: is a LIST
#         :return: list of players (no duplicate) involved in the matches
#         """
#         rv = set()
#         for m in matches:
#             rv.add(m.players())
#         return rv
#
#         # teams = Match.teamsInMatches(matches)
#         # for t in teams:
#         #     for p in t.players:
#         #         rv.add(p)
#         # return list(rv)
#
#     @staticmethod
#     def teamsInMatchList(matches):
#         """
#         :param mathes: is a LIST
#         :return: list of teams (no duplicate) involved in the matches
#         """
#         rv = set()
#         for m in matches:
#             for t in m.teamsOrdered():
#                 rv.add(t)
#         return list(rv)
#
#     @staticmethod
#     def simulateTwoTeams(t1, t2):
#         s1 = t1.reel_skill()
#         s2 = t2.reel_skill()
#         win1_probability = sigmoid(s1-s2)
#         return np.random.rand() <= win1_probability # return True if first team wins
#
#
