# -*- coding: utf-8 -*-

"""
This module contains classes :'FactorGraph' 'SubsetNode', 'VariableNode'
"""


class FactorGraph:

    def __init__(self):
        self.root_nodes = []
        self.leaf_nodes = []

    @staticmethod
    def makePlayerFactorGraph(player):
        fg = FactorGraph()

        f_1 = FactorNode(1)
        s_i = VariableNode('s')
        f_2 = FactorNode(2)
        p_i = VariableNode('p')

        f_1.neighbours.append(s_i) # append a single item
        s_i.neighbours += ([f_1, f_2]) # append multiple items
        f_2.neighbours += ([s_i, p_i])
        p_i.neighbours.append(f_2)

        fg.root_nodes = [f_1]
        fg.leaf_nodes = [p_i]

        return fg


    @staticmethod
    def makeTeamFactorGraph(team):
        fg = FactorGraph()

        f_3 = FactorNode(3)
        t_i = VariableNode('t')

        for a_player in team.players:
            pl_graph = FactorGraph.makePlayerFactorGraph(a_player)
            fg.root_nodes += pl_graph.root_nodes
            p_i = pl_graph.leaf_nodes[0]
            p_i.neighbours.append(f_3)
            f_3.neighbours.append(p_i)
        f_3.neighbours.append(t_i)
        t_i.neighbours.append(f_3)

        fg.leaf_nodes.append(t_i)
        return fg


    @staticmethod
    def makeMatchFactorGraph(match):
        fg = FactorGraph()

        teams_ordered = match.teamsOrdered()
        buffer = None
        for i in range(1,len(teams_ordered)):
            if i == 1 :
                prev = FactorGraph.makeTeamFactorGraph(teams_ordered[i-1])  # this is the top team
                fg.root_nodes += prev.root_nodes
            else:
                prev = buffer  # prev --> team graph
            next = FactorGraph.makeTeamFactorGraph(teams_ordered[i])  # next --> team graph
            buffer = next

            fg.root_nodes += next.root_nodes

            f_3 = FactorNode(3)
            d_i = VariableNode('d')
            f_4 = FactorNode(4)

            t_prev = prev.leaf_nodes[0] # prev team's perf.
            t_next = next.leaf_nodes[0] # next team's perf.

            f_3.neighbours.append(t_prev)
            t_prev.neighbours.append(f_3)

            f_3.neighbours.append(t_next)
            t_next.neighbours.append(f_3)

            f_3.neighbours.append(d_i)
            d_i.neighbours.append(f_3)

            d_i.neighbours.append(f_4)
            f_4.neighbours.append(d_i)

            fg.leaf_nodes.append(f_4)
            return fg



class FactorNode:
    """
    Subset node a.k.a. factor node
    """
    def __init__(self,type):
        """
        Init a subset node object

        :param type: is in {1,2,3,4}
         1 is prior
         2 is likelihood
         3 is to sum up performances (gaussian) and substract team skills (gaussian)
         4 is drawn or non-drawn (match result)
        :return: subset node object
        """
        self.type = type
        self.neighbours = []

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return 'F : ' + str(self.type)



class VariableNode:

    def __init__(self,type):
        """
        :param type: is in {s,p,t,d}
         s is s_i, skill
         p is p_i, performance
         d is t_i, team performance
         t is d_i, team performance difference
        :return:
        """
        self.type = type
        self.neighbours = []

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return 'V : ' + self.type



