# DataFactory

'''
def createPlayers(self,e,m,h):
    player_list = [None] * (e+m+h)
    for i in range(0,e+m+h):
        if i < e : p = self.createPlayer('easy')
        elif i < e+m : p = self.createPlayer('medium')
        else: p = self.createPlayer('hard')
        player_list[i] = p
    return player_list

def generateMatches(self,playerList,count):
    pc = len(playerList)
    matches = -1 * np.ones([pc,pc,count])
    for i in range(0,pc):
        for j in range(i+1,pc):
            p1 = playerList[i]
            p2 = playerList[j]
            p1_chance = ut.sigmoid(p1.reel_skill-p2.reel_skill)
            result = ut.processChance(p1_chance,1,count)

            matches[i,j,:] = result
    return matches

def generateSyntheticMatchesFull(teams):
    matches = []
    tc = len(teams)
    for i in range(0, tc):
        for j in range(i+1, tc):
            m = md.Match([teams[i],teams[j]])
            # p1 = teams[i]
            # p2 = teams[j]
            # p = sigmoid(p1.reel_skill-p2.reel_skill)
            # p1_wins = np.random.rand() < p
            # result = [0,1] if p1_wins else [1,0]
            # matches.append([[[p1],[p2]],result])
            matches.append(m)
    return matches

# def generateSyntheticMatchesSparse(players, count=10):
#     matches = [None] * count
#     for i in range(0,count):
#         team_size = customRand(1,3)
#         competitor_number = customRand(2,2)
#         competitors = np.random.choice(len(players),competitor_number*team_size,replace=False)
#         t1 = competitors[0:team_size]
#         t2 = competitors[team_size:]
#         result = generateMatchResult([t1,t2])
#         matches[i] = [[t1,t2],result]
#     return matches
'''






# class Environment:
#     instance = None
#
#     def __init__(self):
#
#         self.players = set()
#         self.teams = set()
#         self.results = set()
#
#     def newPlayer(self, player):
#         self.players.add(player)
#
#     def newTeam(self, team):
#         self.teams.add(team)
#
#     def newResult(self,result):
#         self.results.add(result)
#
#     def completeResult(self):
#         rv = None
#         for r in self.results:
#             rv = rv.combine(r)
#         return rv




# def mh_mcmc(match):  # a_match --> atomic match --> 1vs1
#     teams = match.teamsOrdered()
#     players = match.players()
#     a_match = 5
#     res = -1 * (a_match.standings[t1] - a_match.standings[t2])  # 1: P1 wins, 0: Draw, -1: P2 wins
#
#     N = 3000
#     samples = np.zeros([N,3])
#     samples[:,2] = res  # res is fixed
#     samples[0,0] = 25  # s1 = 25
#     samples[0,1] = 25  # s2 = 25
#     for i in range(1,N):
#         s1 = samples[i-1,0]
#         s2 = samples[i-1,1]
#         s1_new, s2_new = propose(s1,s2) # q(x-->x')
#
#         # prob_s1_new = probFromEmpirical(t1_pri,s1_new)  # P(S1_new)
#         # prob_s2_new = probFromEmpirical(t2_pri,s2_new)  # P(S2_new)
#         prob_s1_new = t1.pOfX(s1_new)  # P(S1_new)
#         prob_s2_new = t2.pOfX(s2_new)  # P(S2_new)
#         prob_r_new = md.sigmoid(s1_new-s2_new) if res == 1 else md.sigmoid(s2_new-s1_new)  # P(R|S1_new,S2_new)
#         pi_new = prob_r_new * prob_s1_new * prob_s2_new  # P(S1_new, S2_new, R)
#
#         prob_s1 = t1.pOfX(s1)  # P(S1)
#         prob_s2 = t2.pOfX(s2)  # P(S2)
#         prob_r = md.sigmoid(s1-s2) if res == 1 else md.sigmoid(s2-s1)  # P(R|S1,S2)
#         pi = prob_r * prob_s1 * prob_s2  # P(S1, S2, R)
#
#         ratio = pi_new / pi
#         accept = min(1,ratio)  # alpha(x-->x')
#         if np.random.rand() < accept:
#             samples[i,0] = s1_new
#             samples[i,1] = s2_new
#         else:
#             samples[i,0] = s1
#             samples[i,1] = s2
#     t1.samples = samples[25:,0]
#     t2.samples = samples[25:,1]
#     print "Atomix executed"





# def probFromEmpirical(data, value):
#     freqs = np.histogram(data,bins=50,range=[0,50],density=True)[0]
#     return freqs[int(value)]

# def sampleFromEmpirical(data, size):
#     freqs = np.histogram(data,bins=50,range=[0,50],density=True)[0]
#     cum = np.cumsum(freqs)
#     rv = np.zeros([size,1])
#     for i in range(size):
#         u = np.random.rand() < cum
#         rv[i] = np.argmax(u) + np.random.rand()
#     return rv

#
# def propose(s1,s2):
#     # the proposal is multivariate gaussian
#     [s1_new, s2_new] = np.random.multivariate_normal([s1,s2],np.eye(2)*1)
#     s1_new = 0 if s1_new < 0 else s1_new
#     s2_new = 0 if s2_new < 0 else s2_new
#     s1_new = 49.9 if s1_new > 49.9 else s1_new
#     s2_new = 49.9 if s2_new > 49.9 else s2_new
#     return s1_new, s2_new
#
# def mh_mcmc(match):  # a_match --> atomic match --> 1vs1
#     teams = match.teamsOrdered()
#     players = match.players()
#     a_match = 5
#     res = -1 * (a_match.standings[t1] - a_match.standings[t2])  # 1: P1 wins, 0: Draw, -1: P2 wins
#
#     N = 3000
#     samples = np.zeros([N,3])
#     samples[:,2] = res  # res is fixed
#     samples[0,0] = 25  # s1 = 25
#     samples[0,1] = 25  # s2 = 25
#     for i in range(1,N):
#         s1 = samples[i-1,0]
#         s2 = samples[i-1,1]
#         s1_new, s2_new = propose(s1,s2) # q(x-->x')
#
#         # prob_s1_new = probFromEmpirical(t1_pri,s1_new)  # P(S1_new)
#         # prob_s2_new = probFromEmpirical(t2_pri,s2_new)  # P(S2_new)
#         prob_s1_new = t1.pOfX(s1_new)  # P(S1_new)
#         prob_s2_new = t2.pOfX(s2_new)  # P(S2_new)
#         prob_r_new = md.sigmoid(s1_new-s2_new) if res == 1 else md.sigmoid(s2_new-s1_new)  # P(R|S1_new,S2_new)
#         pi_new = prob_r_new * prob_s1_new * prob_s2_new  # P(S1_new, S2_new, R)
#
#         prob_s1 = t1.pOfX(s1)  # P(S1)
#         prob_s2 = t2.pOfX(s2)  # P(S2)
#         prob_r = md.sigmoid(s1-s2) if res == 1 else md.sigmoid(s2-s1)  # P(R|S1,S2)
#         pi = prob_r * prob_s1 * prob_s2  # P(S1, S2, R)
#
#         ratio = pi_new / pi
#         accept = min(1,ratio)  # alpha(x-->x')
#         if np.random.rand() < accept:
#             samples[i,0] = s1_new
#             samples[i,1] = s2_new
#         else:
#             samples[i,0] = s1
#             samples[i,1] = s2
#     t1.samples = samples[25:,0]
#     t2.samples = samples[25:,1]
#     print "Atomix executed"


# def mh_mcmc(a_match):  # a_match --> atomic match --> 1vs1
#     t1 = a_match.teamsOrdered()[0]
#     t2 = a_match.teamsOrdered()[1]
#     # t1_pri = t1.prior
#     # t2_pri = t2.prior
#     res = -1 * (a_match.standings[t1] - a_match.standings[t2])  # 1: P1 wins, 0: Draw, -1: P2 wins
#
#     N = 3000
#     samples = np.zeros([N,3])
#     samples[:,2] = res  # res is fixed
#     samples[0,0] = 25  # s1 = 25
#     samples[0,1] = 25  # s2 = 25
#     for i in range(1,N):
#         s1 = samples[i-1,0]
#         s2 = samples[i-1,1]
#         s1_new, s2_new = propose(s1,s2) # q(x-->x')
#
#         # prob_s1_new = probFromEmpirical(t1_pri,s1_new)  # P(S1_new)
#         # prob_s2_new = probFromEmpirical(t2_pri,s2_new)  # P(S2_new)
#         prob_s1_new = t1.pOfX(s1_new)  # P(S1_new)
#         prob_s2_new = t2.pOfX(s2_new)  # P(S2_new)
#         prob_r_new = md.sigmoid(s1_new-s2_new) if res == 1 else md.sigmoid(s2_new-s1_new)  # P(R|S1_new,S2_new)
#         pi_new = prob_r_new * prob_s1_new * prob_s2_new  # P(S1_new, S2_new, R)
#
#         prob_s1 = t1.pOfX(s1)  # P(S1)
#         prob_s2 = t2.pOfX(s2)  # P(S2)
#         prob_r = md.sigmoid(s1-s2) if res == 1 else md.sigmoid(s2-s1)  # P(R|S1,S2)
#         pi = prob_r * prob_s1 * prob_s2  # P(S1, S2, R)
#
#         ratio = pi_new / pi
#         accept = min(1,ratio)  # alpha(x-->x')
#         if np.random.rand() < accept:
#             samples[i,0] = s1_new
#             samples[i,1] = s2_new
#         else:
#             samples[i,0] = s1
#             samples[i,1] = s2
#     t1.samples = samples[25:,0]
#     t2.samples = samples[25:,1]
#     print "Atomix executed"


#
#
# def updateAtomic(a_matches):
#     if len(a_matches) == 0:
#         return
#     else:
#         current = a_matches[0]
#         mh_mcmc(current)
#         update(a_matches[1:0])
#
# def update(matches):
#     if len(matches) == 0:
#         return
#     else:
#         a_matches = matches[0].atomicMatches()
#         updateAtomic(a_matches)
#         plt.figure()
#         # ax = sns.distplot(t1.samples,bins=50)
#         # ax = sns.distplot(t2.samples,bins=50)
#         x = np.arange(0,50,0.1)
#         ax = plt.subplot(111)
#         ax.plot(x,t1.pOfX(x))
#         ax.plot(x,t2.pOfX(x))
#         ax.set_xlim([0, 50])
#         # ax.set_ylim([0, 1])
#         update(matches[1:])




# def isSynthetic(self):
    #     """
    #     Synthetic players has a valid reel_skill value.
    #
    #     :return: True if player is synthetic False otherwise
    #     """
    #     return self.reel_skill is not None


    # def mu(self):
    #     mu = 0
    #     for p in self.players:
    #         mu += p.mu
    #     return mu
    #
    # def sigma(self):
    #     return None


    # def isSynthetic(self):
    #     for p in self.players:
    #         if not(p.isSynthetic()): return False
    #     return True















# p = df.generateSyntheticPlayers(4)
# t1 = md.Team(p[0:2])
# t2 = md.Team(p[2:])
# t = [t1,t2]
# print 'T1', t1
# print 'T2', t2
# m = []
# m += df.generateSyntheticMatchesFull(t)
# m += df.generateSyntheticMatchesFull(t)
# m += df.generateSyntheticMatchesFull(t)
# t3 = md.Team([p[0]])
# t4 = md.Team([p[1]])
# m.append(df.simulateTwoTeams(t3,t4))
# m.append(df.simulateTwoTeams(t3,t4))
# m.append(df.simulateTwoTeams(t3,t4))
# m.append(df.simulateTwoTeams(t3,t4))
# m.append(df.simulateTwoTeams(t3,t4))
# m.append(df.simulateTwoTeams(t3,t4))

# liste = md.Versus.produceVs(results)
# mc.mh_mcmc(liste,5000)























# class Point:
#     counter = 101
#     def __init__(self):
#         self.id = Point.counter
#         self.x=5
#         self.y=5
#         Point.counter += 1
#
# l = []
# for i in range(5):l.append(Point())
# l[0].x = 10
# print "selam"


import Model as md

# x = np.arange(0,50)
# y = md.sigmoid(x)
# y = y / np.sum(y)
# hi = np.random.multinomial(10000,y)
# hi = hi.astype(float)
# hi = hi / np.sum(hi)



# samples = md.sample_expo(1000,1)
# f = plt.figure()
# ax = f.add_subplot(111)
# sns.distplot(samples,ax=ax)
# plt.show()

# print md.bi_expo([-5,0,4,5])


# # print md.multinomial_log(10000,np.log([0.4, 0.4, 1.2]))
# md.draw_factor = 0.99
# x = np.arange(0,50)
# y1 = md.sigmoid(x-20)
# # y2 = md.sigmoid(-x+25)
# y3 = md.bi_expo(x,mean=10)
# plt.figure()
# plt.plot(x,y1)
# # plt.plot(x,y2)
# plt.plot(x,y3)
# plt.plot(x,y1*y3)
#
# md.draw_factor = 0.33
# x = np.arange(0,50)
# y1 = md.sigmoid(x-20)
# # y2 = md.sigmoid(-x+25)
# y3 = md.bi_expo(x,mean=10)
# plt.figure()
# plt.plot(x,y1)
# # plt.plot(x,y2)
# plt.plot(x,y3)
# plt.plot(x,3*y1*y3)
#
#
# plt.show(block=True)

# r1 = ts.Rating()  # 1P's skill
# r2 = ts.Rating()  # 2P's skill
#
# new_r1, new_r2 = ts.rate_1vs1(r1, r2)











# message = self.perf_var.message_from(self)
            # m_pi, m_tau = message.pi, message.tau
            # new_pi = a * (self.perf_var.value.pi - m_pi)
            # new_tau = a * (self.perf_var.value.tau - m_tau)
            #
            # message = self.skill_var.message_from(self)
            # m_pi, m_tau = message.pi, message.tau
            # new_pi = a * (self.skill_var.value.pi - m_pi)
            # new_tau = a * (self.skill_var.value.tau - m_tau)