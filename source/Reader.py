# -*- coding: utf-8 -*-

import numpy as np
import model as mo


def read_data(*args):
    data = np.genfromtxt(args[0],delimiter=',',dtype=str)
    for i in range(1,len(args)):
        temp = np.genfromtxt(args[i],delimiter=',',dtype=str)
        data = np.vstack([data, temp])
    return data

# assuming teams consists of single player because our data has this formation
def form_objects(data):
    N = data.shape[0]
    p = []
    t = []
    m = [None] * N
    for i in range(N):
        p1 = mo.Player.with_name(data[i,0])
        t1 = mo.Team.with_players([p1])
        p2 = mo.Player.with_name(data[i,1])
        t2 = mo.Team.with_players([p2])
        m[i] = mo.Versus(t1,t2,int(data[i,2]))

        no_duplicate_insert(p, [p1, p2])
        no_duplicate_insert(t, [t1, t2])
    return p, t, m


def no_duplicate_insert(a_list, objects):
    for an_object in objects:
        if not (an_object in a_list):
            a_list.append(an_object)

# def find_by_name(players, name):
#     for p in players:
#         if p.name == name: return p
#     return None
#
#
# def find_by_members(teams, members):
#     for t in teams:
#         if t.players == members: return t
#     return None

