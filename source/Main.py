# # -*- coding: utf-8 -*-
#
import test
counter = 1
num_matches = [3, 10]
draw_factors = [0, 0.1, 0.33]
#
# for nm in num_matches:
#     for df in draw_factors:
#         test.test1(counter, nm, df)
#         counter += 1
#
# for nm in num_matches:
#     test.test2(counter, nm, draw_factors[0])
#     counter += 1
#
# test.test3(counter, num_matches[1], draw_factors[0])
# counter += 1
#
# for nm in num_matches:
#     test.test4(counter, nm, draw_factors[0])
#     counter += 1
#
# # tennis
# for i in xrange(1,5):
#     test.test5(counter, i)
#     counter += 1

# # football
# for i in xrange(5):
i = 4
counter = 5
test.test6(counter, i)

counter += 1

# # basketball
for i in xrange(1,3):
    test.test7(counter, i)
    counter += 1


print 'All done'