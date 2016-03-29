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
