# -*- coding: utf-8 -*-

import numpy as np
import Model as md

def read_data(*args):
    data = np.genfromtxt(args[0],delimiter=',',dtype=str)
    for i in range(1,len(args)):
        temp = np.genfromtxt(args[i],delimiter=',',dtype=str)
        data = np.vstack([data, temp])
    return data

def form_objects(data):
    p_names = np.unique(data[:,:2])
    p_num = p_names.shape[0]
    p = [None] * p_num
    t = [None] * p_num
    for i in range(p_num):
        p[i] = md.Player(name=p_names[i])
        t[i] = md.Team([p[i]])

    N = data.shape[0]
    m = [None] * N
    for i in range(N):
        p1 = find_by_name(p,data[i,0])
        t1 = find_by_members(t, [p1])
        p2 = find_by_name(p,data[i,1])
        t2 = find_by_members(t, [p2])
        m[i] = md.Versus(t1,t2,int(data[i,2]))
    return p,t,m


def find_by_name(players, name):
    for p in players:
        if p.name == name: return p
    return None

def find_by_members(teams, members):
    for t in teams:
        if t.players == members: return t
    return None

