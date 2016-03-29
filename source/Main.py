# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import Model as md
import MC as mc
import DataFactory as df

# p1 = md.Player(reel_skill=30)
# p2 = md.Player(reel_skill=9)
# p3 = md.Player(reel_skill=12)

# t1 = md.Team([p1])
# t2 = md.Team([p2,p3])
# t3 = md.Team([p3])
# t4 = md.Team([p2])
# results = [{t1:1,t2:2},{t4:1,t3:2},{t4:1,t3:2},{t4:1,t3:2}]

p,t,m = df.generateSyntheticData(4,1)

# liste = md.Versus.produceVs(results)
# mc.mh_mcmc(liste,5000)




# plt.figure()
# ax = sns.distplot(p1.sample_list,bins=50)
# ax = sns.distplot(p2.sample_list,bins=50)
# ax = sns.distplot(p3.sample_list,bins=50)
# plt.figure()
# x = np.arange(0,50,0.1)
# plt.plot(x,p1.prob(x))
# plt.plot(x,p2.prob(x))
# plt.plot(x,p3.prob(x))
# plt.show()

print 'selam'

