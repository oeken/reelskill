# -*- coding: utf-8 -*-

import numpy as np
import Model as md
np.random.seed(100)


def customRand(l,h):
    return int(np.random.rand()*(h-l+1) + l)

def generatePlayerWithLevel(level):
    skill = int(np.random.rand() * 10 + (level+1)*10)
    return md.Player(reel_skill=skill)


def generateSyntheticPlayers(count):
    player_list = [None] * count
    for i in range(0,count):
        level = customRand(0,2)
        player_list[i] = generatePlayerWithLevel(level)
    return player_list

def generateSyntheticTeams(players):
    teams = [None] * len(players)
    for i in range(0,len(players)):
        teams[i] = md.Team([players[i]])
    return teams

def generateSyntheticMatchesFull(teams):
    matches = []
    tc = len(teams)
    for i in range(0, tc):
        for j in range(i+1, tc):
            m = md.Match([teams[i],teams[j]])
            matches.append(m)
    return matches

def generateSyntheticData(count):
    players = generateSyntheticPlayers(count)
    teams = generateSyntheticTeams(players)
    matches = generateSyntheticMatchesFull(teams)
    return players, teams, matches


