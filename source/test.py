# -*- coding: utf-8 -*-
""" This module contains pre-specified tests """
import time
import ep
import evaluate as ev
import factory as fa
import mc
import model as mo
import reader as re
import numpy as np


def test1(num, num_of_matches, draw_factor):
    test_time = time.ctime().replace(' ','_')
    test_time = 'T1_' + str(num) + '_' + test_time

    # players
    p1 = mo.Player(reel_skill=5)
    p2 = mo.Player(reel_skill=15)
    p3 = mo.Player(reel_skill=25)
    p4 = mo.Player(reel_skill=35)
    p5 = mo.Player(reel_skill=45)
    p_train = [p1,p2,p3,p4,p5]

    # teams
    t1 = mo.Team([p1])
    t2 = mo.Team([p2])
    t3 = mo.Team([p3])
    t4 = mo.Team([p4])
    t5 = mo.Team([p5])
    t_train = [t1,t2,t3,t4,t5]

    # matches
    mo.draw_factor = draw_factor
    m_train = fa.generateSyntheticMatchesFullTimes(t_train, num_of_matches)
    m_test = fa.generateSyntheticMatchesFullTimes(t_train, num_of_matches)

    execute_algorithms(test_time, p_train, t_train, m_train, m_test, 'Synthetic')


def test2(num, num_of_matches, draw_factor):
    test_time = time.ctime().replace(' ','_')
    test_time = 'T2_' + str(num) + '_' + test_time

    # data
    mo.draw_factor = draw_factor
    p_train, t_train = fa.generateSyntheticData(5,1)
    m_train = fa.generateSyntheticMatchesFullTimes(t_train, num_of_matches)
    m_test = fa.generateSyntheticMatchesFullTimes(t_train, num_of_matches)

    execute_algorithms(test_time, p_train, t_train, m_train, m_test, 'Synthetic')


def test3(num, num_of_matches, draw_factor):
    test_time = time.ctime().replace(' ','_')
    test_time = 'T3_' + str(num) + '_' + test_time

    # data
    mo.draw_factor = draw_factor
    p_train, t_train = fa.generateSyntheticData(15,1)
    m_train = fa.generateSyntheticMatchesFullTimes(t_train, num_of_matches)
    m_test = fa.generateSyntheticMatchesFullTimes(t_train, num_of_matches)

    execute_algorithms(test_time, p_train, t_train, m_train, m_test, 'Synthetic')


def test4(num, num_of_matches, draw_factor):
    test_time = time.ctime().replace(' ','_')
    test_time = 'T4_' + str(num) + '_' + test_time

    # data
    # players
    p1 = mo.Player(reel_skill=5)
    p2 = mo.Player(reel_skill=10)
    p3 = mo.Player(reel_skill=15)
    p4 = mo.Player(reel_skill=20)
    p5 = mo.Player(reel_skill=25)
    p6 = mo.Player(reel_skill=30)
    p7 = mo.Player(reel_skill=35)
    p8 = mo.Player(reel_skill=40)
    p9 = mo.Player(reel_skill=45)
    p_train = [p1,p2,p3,p4,p5,p6,p7,p8,p9]

    # teams
    t1 = mo.Team([p1,p5,p3])
    t2 = mo.Team([p2,p9,p8])
    t3 = mo.Team([p4,p6,p7])
    t_train = [t1,t2,t3]

    mo.draw_factor = draw_factor
    m_train = fa.generateSyntheticMatchesFullTimes(t_train, num_of_matches)
    m_test = fa.generateSyntheticMatchesFullTimes(t_train, num_of_matches)

    execute_algorithms(test_time, p_train, t_train, m_train, m_test, 'Synthetic')



def test5(num, i):
    test_time = time.ctime().replace(' ','_')
    test_time = 'T5_' + str(num) + '_' + test_time

    # data
    data_paths = np.array(['../data/tennis/ausopen.csv', '../data/tennis/rg.csv', '../data/tennis/wimbledon.csv', '../data/tennis/usopen.csv'])
    data_train = re.read_data(*data_paths[:i])
    p_train, t_train, m_train = re.form_objects(data_train)

    if i == 4:
        m_test = None
    else:
        data_test = re.read_data(*data_paths[i:])
        p_test, t_test, m_test = re.form_objects(data_test)

    execute_algorithms(test_time, p_train, t_train, m_train, m_test, 'Tennis')


def test6(num, i):
    test_time = time.ctime().replace(' ','_')
    test_time = 'T6_' + str(num) + '_' + test_time

    # data
    d_train = np.array(['../data/football/germany1.csv', '../data/football/turkey1.csv', '../data/football/spain1.csv', '../data/football/england1.csv'])
    d_test = np.array(['../data/football/germany2.csv', '../data/football/turkey2.csv', '../data/football/spain2.csv', '../data/football/england2.csv'])
    if i == 4:
        data_train = re.read_data(d_train[1], d_test[1])  # turkey
        p_train, t_train, m_train = re.form_objects(data_train)

        execute_algorithms(test_time, p_train, t_train, m_train, None, 'Football')
    else:
        data_train = re.read_data(d_train[i-1])
        p_train, t_train, m_train = re.form_objects(data_train)

        data_test = re.read_data(d_test[i-1])
        p_test, t_test, m_test = re.form_objects(data_test)

        execute_algorithms(test_time, p_train, t_train, m_train, m_test, 'Football')


def test7(num, i):
    test_time = time.ctime().replace(' ','_')
    test_time = 'T7_' + str(num) + '_' + test_time

    # data
    if i == 1:
        data_train = re.read_data('../data/basketball/nba1.csv')
        p_train, t_train, m_train = re.form_objects(data_train)

        data_test = re.read_data('../data/basketball/nba2.csv')
        p_test, t_test, m_test = re.form_objects(data_test)

        execute_algorithms(test_time, p_train, t_train, m_train, m_test,'Basketball')
    else:
        data_train = re.read_data('../data/basketball/nba1.csv', '../data/basketball/nba2.csv')
        p_train, t_train, m_train = re.form_objects(data_train)

        execute_algorithms(test_time, p_train, t_train, m_train, None, 'Basketball')

def test8(num, draw_factor, num_of_matches):
    test_time = time.ctime().replace(' ','_')
    test_time = 'T8_' + str(num) + '_' + test_time

    # players
    p1 = mo.Player(reel_skill=10)
    p2 = mo.Player(reel_skill=40)
    p_train = [p1,p2]

    # teams
    t1 = mo.Team([p1])
    t2 = mo.Team([p2])
    t_train = [t1,t2]

    # matches
    mo.draw_factor = draw_factor
    m_train = fa.generateSyntheticMatchesFullTimes(t_train, num_of_matches)
    m_test = fa.generateSyntheticMatchesFullTimes(t_train, num_of_matches)

    execute_algorithms(test_time, p_train, t_train, m_train, m_test, 'Synthetic')



def execute_algorithms(test_time, p_train, t_train, m_train, m_test, dataname):
    # # MH
    # ite = 8000
    # aggr = 0.7
    # tic = time.time()
    # algo, decisions = mc.mh_mcmc(p_train, m_train, aggr, ite)
    # toc = time.time() - tic
    # out = ev.Output(test_time, m_train, algo, dataname, toc,  ite=ite, aggr=aggr, decisions=decisions, m_test=m_test)
    # out.trigger_csv()
    # out.trigger_plots()
    #
    # # Gibbs
    # ite = 4000
    # tic = time.time()
    # algo = mc.gibbs_mcmc(p_train, m_train, ite)
    # toc = time.time() - tic
    # out = ev.Output(test_time, m_train, algo, dataname, toc , ite=ite, m_test=m_test)
    # out.trigger_csv()
    # out.trigger_plots()

    # # EP
    tic = time.time()
    algo = ep.run(m_train)
    toc = time.time() - tic
    out = ev.Output(test_time, m_train, algo, dataname, toc, m_test=m_test)
    out.trigger_csv()
    out.trigger_plots()

    mo.empty()




