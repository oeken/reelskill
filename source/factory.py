# -*- coding: utf-8 -*-
import numpy as np
import model as md

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

def generateSyntheticTeams(players, size):
    N = len(players) / size
    teams = [None] * N
    players = np.array(players)
    for i in range(N):
        sta = i * size
        end = sta + size
        teams[i] = md.Team.with_players(players[sta:end])
    return teams


    # n = len(players)-(len(players)%size)
    # number_of_teams = n / size
    # teams = teamList(number_of_teams)
    # teams = [None] * number_of_teams
    # for i in number_of_teams:
    #     team[i] = md.Team.with_players()
    #     index = i % number_of_teams
    #     current_team = teams[index]
    #     current_team.addPlayer(players[i])
    # return teams

def generateSyntheticMatchesFull(teams):
    rv = []
    tc = len(teams)
    for i in range(0, tc):
        for j in range(i+1, tc):
            t1 = teams[i]
            t2 = teams[j]
            rv.append(simulateTwoTeams(t1,t2))
    return rv

def generateSyntheticMatchesFullTimes(teams, times):
    m = []
    for i in range(times):
        m += generateSyntheticMatchesFull(teams)
    return m

def generateSyntheticData(player_count, team_size):
    players = generateSyntheticPlayers(player_count)
    teams = generateSyntheticTeams(players,team_size)
    return players, teams

def simulateTwoTeams(t1, t2):
    s1 = t1.reel_skill
    s2 = t2.reel_skill
    w,l,d = md.win_lose_draw(s1-s2)
    u = np.random.rand()
    if u < d:
        return md.Versus(t1,t2,0)
    elif u < d+l:
        return md.Versus(t1,t2,-1)
    elif u < d+l+w:
        return md.Versus(t1,t2,1)

def teamList(size):
    rv = []
    for i in range(size):
        rv.append(md.Team())
    return rv






# def simulate(teams):
#     rv = {}
#     wincounts = np.zeros(len(teams))
#     for i in range(len(teams)):
#         for j in range(i+1,len(teams)):
#             if Match.simulateTwoTeams(teams[i],teams[j]):
#                 wincounts[i] += 1
#             else:
#                 wincounts[j] += 1
#     raw_standings = np.argsort(wincounts)
#     raw_standings = np.fliplr([raw_standings])[0] # reverse the array
#     rv[teams[raw_standings[0]]] = 1 # the winner (rank is 1)
#     for i in range(1,len(raw_standings)):
#         prev_wc = wincounts[raw_standings[i-1]] # wincout of previous team
#         next_wc = wincounts[raw_standings[i]] # wincout of next team
#         prev_team = teams[raw_standings[i-1]] # previous team (ranked before)
#         next_team = teams[raw_standings[i]] # previous team (ranked before)
#         # if next team has less wins then take prev teams rank add 1 and set next team's rank
#         rv[next_team] = rv[prev_team]+1 if next_wc < prev_wc else rv[prev_team]
#     return rv


