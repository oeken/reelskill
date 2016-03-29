# -*- coding: utf-8 -*-

"""
This module contains model classes :'Player' 'Team', 'Match'
"""

import numpy as np
from faker import Faker
from scipy import stats as st

fake = Faker()
fake.seed(100)

# np.random.seed(100)

MU = 25.0
SIGMA = MU / 3.0

def sigmoid(x):
    return 1/(1+np.e ** (-0.2*x));

class Player:
    """
    Player class
    """
    player_count = 101

    def __init__(self, mu=MU, sigma=SIGMA, name='Doe', reel_skill=None):
        self.samples = np.random.rand(1000)*50
        self.mu = mu
        self.sigma = sigma
        self.name = fake.name() if name == 'Doe' else name
        self.reel_skill = reel_skill
        self.id = Player.player_count
        Player.player_count += 1

    def __str__(self):
        return self.name + ', ' + str(self.id)+', '+ str(self.reel_skill) + ', m:' + str(self.mu) + ', s:' + str(self.sigma)

    def __repr__(self):
        return self.name+'('+str(self.id)+')'

    def isSynthetic(self):
        """
        Synthetic players has a valid reel_skill value.

        :return: True if player is synthetic False otherwise
        """
        return self.reel_skill is not None

class Team:
    """
    Team class

    * A List of players
    """

    def __init__(self, players=[]):
        self.players = players
        self.samples = players[0].samples

    def pOfX(self,x):
        kernel = st.gaussian_kde(self.samples)
        return kernel.evaluate(x)

    # def prior(self):
    #     rv = []
    #     for p in self.players:
    #         rv = np.append(rv, p.prior)
    #     return rv







    def addPlayer(self,player):
        self.players.append(player)

    def size(self):
        return len(self.players)

    def mu(self):
        mu = 0
        for p in self.players:
            mu += p.mu
        return mu

    def sigma(self):
        return None

    def reel_skill(self):
        if self.isSynthetic():
            rs = 0
            for p in self.players:
                rs += p.reel_skill
            return rs
        else:
            raise ValueError('Team is not synthetic')

    def isSynthetic(self):
        for p in self.players:
            if not(p.isSynthetic()): return False
        return True

    def __str__(self):
        return repr(self)

    def __repr__(self):
        s = '<<'
        for p in self.players:
            s += repr(p) + '-'
        return s+'>>'


class Match:
    """
    Match class
    """

    def __init__(self, teams):
        self.synthetic = None
        self.standings = None

        if isinstance(teams,dict):
            for t in teams.keys():
                if t.isSynthetic():
                    self.synthetic = True
                    break
            self.synthetic = False
            self.standings = teams
        elif isinstance(teams,list):
            for t in teams:
                if not t.isSynthetic(): raise TypeError('Non synthetic team passed without standings')
            self.synthetic = True
            self.standings = Match.simulate(teams)
        else:
            raise TypeError('Invalid data type passed')

    def __str__(self):
        rv = '{{ '
        cr = 1
        lr = self.lowestRank()
        while cr <= lr:
            rv += str(self.teamsOnRank(cr))+'['+str(cr)+']' + '  --  '
            cr += 1
        return rv + ' }}'

    def __repr__(self):
        return str(self)

    def teamsOrdered(self):
        rv = []
        cr = 1
        lr = self.lowestRank()
        while cr <= lr:
            for t in self.teamsOnRank(cr): rv.append(t)
            cr += 1
        return rv

    def lowestRank(self):
        return max(self.standings.values())

    def teamsOnRank(self, rank):
        if rank > self.lowestRank():
            return []   # no player with such a rank
        else:
            rv = []
            for t in self.standings.keys():
                if self.standings[t] == rank: rv.append(t)
            return rv

    def teamCount(self):
        return len(self.standings.keys())

    def isSynthetic(self):
        return self.synthetic

    def atomicMatches(self):
        rv = []
        p = 1
        while p < self.teamCount():
            t1 = self.teamsOrdered()[p-1]
            t2 = self.teamsOrdered()[p]
            m = Match({t1:self.standings[t1], t2:self.standings[t2]})
            m.synthetic = self.synthetic
            rv.append(m)
            p += 1
        return rv



    @staticmethod
    def simulate(teams):
        rv = {}
        wincounts = np.zeros(len(teams))
        for i in range(len(teams)):
            for j in range(i+1,len(teams)):
                if Match.simulateTwoTeams(teams[i],teams[j]):
                    wincounts[i] += 1
                else:
                    wincounts[j] += 1
        raw_standings = np.argsort(wincounts)
        raw_standings = np.fliplr([raw_standings])[0] # reverse the array
        rv[teams[raw_standings[0]]] = 1 # the winner (rank is 1)
        for i in range(1,len(raw_standings)):
            prev_wc = wincounts[raw_standings[i-1]] # wincout of previous team
            next_wc = wincounts[raw_standings[i]] # wincout of next team
            prev_team = teams[raw_standings[i-1]] # previous team (ranked before)
            next_team = teams[raw_standings[i]] # previous team (ranked before)
            # if next team has less wins then take prev teams rank add 1 and set next team's rank
            rv[next_team] = rv[prev_team]+1 if next_wc < prev_wc else rv[prev_team]
        return rv

    @staticmethod
    def playersInMatches(matches):
        teams = Match.teamsInMatches(matches)
        rv = set()
        for t in teams:
            for p in t.players:
                rv.add(p)
        return list(rv)

    @staticmethod
    def teamsInMatches(mathes):
        rv = set()
        for m in mathes:
            for t in m.teamsOrdered():
                rv.add(t)
        return list(rv)

    @staticmethod
    def simulateTwoTeams(t1, t2):
        s1 = t1.reel_skill()
        s2 = t2.reel_skill()
        win1_probability = sigmoid(s1-s2)
        return np.random.rand() <= win1_probability # return True if first team wins


