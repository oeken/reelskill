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


import source.Model as md

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







    #             if vs.r == 0:
    #                 if p in vs.t1.players:
    #                     additive = vs.t1.sample() - p.sample
    #                     subtractive = vs.t2.sample()
    #                     expo = True
    #                 elif p in vs.t2.players:
    #                     additive = vs.t2.sample() - p.sample
    #                     subtractive = vs.t1.sample()
    #                     expo = True
    #                 else:
    #                     continue
    #             else:
    #                 expo = False
    #                 additive = vs.t1.sample() - p.sample if p in vs.t1.players else vs.t2.sample() - p.sample
    #                 subtractive = vs.t2.sample() if p in vs.t1.players else vs.t1.sample()
    #                 if vs.r == 1 and p in vs.t1.players: w = True
    #                 elif vs.r == 1 and p in vs.t2.players: w = False
    #                 elif vs.r == -1 and p in vs.t1.players: w = False
    #                 elif vs.r == -1 and p in vs.t2.players: w = True
    #
    #             if expo:
    #                 temp = md.bi_expo(x,mean=subtractive-additive)
    #             elif w:
    #                 temp = md.sigmoid(x-(subtractive-additive))
    #             else:
    #                 temp = md.sigmoid(-x+(subtractive-additive))
    #             conditional = conditional + np.log(temp)
    #             # conditional = conditional * temp
    #             # conditional = conditional / np.sum(conditional)
    #         dummy = conditional[0,50:100]
    #         # dummy = dummy / np.sum(dummy)
    #         dummy2 = np.exp(dummy)
    #         dummy2 = dummy2 / np.sum(dummy2)
    #
    #
    #
    #         sample = x[np.nonzero(md.multinomial_log(1,dummy))[0][0]] + 50
    #         print 'Sample ', sample
    #
    #         plt.figure()
    #         plt.plot(x[50:100],dummy2)
    #         plt.show()
    #         # sample = x[np.nonzero(np.random.multinomial(1,dummy))[0][0]] + 50
    #         p.sample = sample
    #         p.new_sample_list = np.append(p.new_sample_list, sample)
    # updateSamples()







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




# def find_by_name(players, name):
#     for p in players:
#         if p.name == name: return p
#     return None

# def find_by_members(teams, members):
#     for t in teams:
#         if t.players == members: return t
#     return None








#
# def compare(data_test, players):
#     N = data_test.shape[0]
#     predictions = np.zeros([N,1]) - 5
#     probabilities = np.zeros([N,1]) - 5
#     correct = np.zeros([N,1]) - 5
#     text = ''
#     for i in range(N):
#         p1_name = data_test[i,0]
#         p2_name = data_test[i,1]
#         result  = int(data_test[i,2])
#         text += p1_name+' VS. ' + p2_name + ' => '+ str(result) + ' ||| '
#
#         p1 = re.find_by_name(players, p1_name)
#         p2 = re.find_by_name(players, p2_name)
#         w,l,d = mo.win_lose_draw(p1.mean()-p2.mean())
#         text += 'W: '+'{:.2f}'.format(w)+', L: '+'{:.2f}'.format(l)+', D: '+'{:.2f}'.format(d)
#         index = np.array([w,l,d])
#         best = np.argsort(index)[::-1][0]
#         probabilities[i,0] = np.array([w,l,d])[best]
#         if best == 0:
#             prediction = 1
#         elif best == 1:
#             prediction = -1
#         elif best == 2:
#             prediction = 0
#         text += ', Prediction: '+str(prediction)
#
#         if prediction == result:
#             correct[i,0] = 1
#             text += '  [O]\n'
#         else:
#             correct[i,0] = 0
#             text += '  [X]\n'
#         predictions[i,0] = prediction
#
#     N = data_test.shape[0]
#     cor = np.sum(correct)
#     incor = N - cor
#     text += '\nCorrect: '+str(cor)+'  Incorrect: '+str(incor)+' Percentage: '+str(cor*100/N)+'\n'
#     text += 'Mean of Expected : ' + str(np.mean(probabilities)) + '\n'
#     return predictions, probabilities, correct, text
#
#
















# def distill_testdata(algo, test_matches):
    # for i in xrange(N):
    #     p1 = find_by_name(players, test_data[i,0])
    #     t1 = find_by_members(teams, [p1])
    #     p2 = find_by_name(players, test_data[i,1])
    #     t2 = find_by_members(teams, [p2])
    #     matches[i] = mo.Versus(t1, t2, int(test_data[i,2]))
    # prob = np.zeros([1,N])
    # guess = np.zeros([1,N])
    # cor = np.zeros([1,N])
    # for i in xrange(N):
    #     m = matches[i]
    #     if algo == 'EP':
    #         prob[i] = mo.win_lose_draw_ep(m.t1.players[0].mc_mu - m.t2.players[0].mc_mu)
    #     else:
    #         prob[i] = mo.win_lose_draw(m.t1.mc_mu-m.t2.mc_mu)
    #     best = np.argsort[prob[i]][::-1]
    #     if best == 0:
    #         guess[i] = 1
    #     elif best == 1:
    #         guess[i] = -1
    #     else:
    #         guess[i] = 0
    #     cor[i] = int(guess[i] == m.r)
    # return prob, guess, cor















    # data_training = re.read_data('../data/tennis/ausopen.csv','../data/tennis/rg.csv')
    # data_test = re.read_data('../data/tennis/wimbledon.csv','../data/tennis/usopen.csv')
    # p_train, t_train, m_train = re.form_objects(data_training)
    # p_test, t_test, m_test = re.form_objects(data_test)

    # # players
    # p1 = mo.Player.with_name(reel_skill=20)
    # p2 = mo.Player.with_name(reel_skill=35)
    # p_train = [p1,p2]
    # # # teams
    # t1 = mo.Team.with_players([p1])
    # t2 = mo.Team.with_players([p2])
    # t_train = [t1,t2]
    # # # matches
    # mo.draw_factor = 0.0
    # m_train = fa.generateSyntheticMatchesFullTimes(t_train,10)




# players
    # p1 = mo.Player(reel_skill=5)
    # p2 = mo.Player(reel_skill=15)
    # p3 = mo.Player(reel_skill=25)
    # p4 = mo.Player(reel_skill=35)
    # p5 = mo.Player(reel_skill=45)
    # p = [p1,p2,p3,p4,p5]
    #
    # # teams
    # t1 = mo.Team([p1])
    # t2 = mo.Team([p2])
    # t3 = mo.Team([p3])
    # t4 = mo.Team([p4])
    # t5 = mo.Team([p5])
    # t = [t1,t2,t3,t4,t5]

    # matches
    # mo.draw_factor = 0.0
    # m = fa.generateSyntheticMatchesFullTimes(t,5)


#
# def test2():
#     # players
#     p1 = mo.Player(reel_skill=20)
#     p2 = mo.Player(reel_skill=35)
#     p = [p1,p2]
#
#     # teams
#     t1 = mo.Team([p1])
#     t2 = mo.Team([p2])
#     t = [t1,t2]
#
#     # matches
#     mo.draw_factor = 0.33
#     m = fa.generateSyntheticMatchesFullTimes(t,10)
#
#     # execute ep
#     ep.process(m)
#
#     # evaluate results
#     results = ev.a_fun(p)
#
#     # visualize results
#     vi.plot_sample(results)
#
#
# def test3():
#     mo.draw_factor = 0
#     p,t = fa.generateSyntheticData(6,1)
#     m = fa.generateSyntheticMatchesFullTimes(t,100)
#
#
# def test4():
#     data_training = re.read_data('../data/tennis/ausopen.csv','../data/tennis/rg.csv')
#     data_test = re.read_data('../data/tennis/wimbledon.csv','../data/tennis/usopen.csv')
#     p,t,m = re.form_objects(data_training)
#
#
# def test5():
#     data = re.read_data('../data/football/turkey.csv')
#     N = data.shape[0]
#     data_training = data[:N/2,:]
#     data_test = data[N/2:,:]
#     p,t,m = re.form_objects(data_training)
#
#
# def test6():
#     mo.draw_factor = 0
#     data = re.read_data('../data/basketball/nba.csv')
#     N = data.shape[0]
#     data_training = data[:N/2,:]
#     data_test = data[N/2:,:]
#     p,t,m = re.form_objects(data_training)
#


