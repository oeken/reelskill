import Model as md
import Reader as rd
import numpy as np

def compare(data_test, players):
    N = data_test.shape[0]
    predictions = np.zeros([N,1]) - 5
    probabilities = np.zeros([N,1]) - 5
    correct = np.zeros([N,1]) - 5
    text = ''
    for i in range(N):
        p1_name = data_test[i,0]
        p2_name = data_test[i,1]
        result  = int(data_test[i,2])
        text += p1_name+' VS. ' + p2_name + ' => '+ str(result) + ' ||| '

        p1 = rd.find_by_name(players, p1_name)
        p2 = rd.find_by_name(players, p2_name)
        w,l,d = md.win_lose_draw(p1.mean()-p2.mean())
        text += 'W: '+'{:.2f}'.format(w)+', L: '+'{:.2f}'.format(l)+', D: '+'{:.2f}'.format(d)
        index = np.array([w,l,d])
        best = np.argsort(index)[::-1][0]
        probabilities[i,0] = np.array([w,l,d])[best]
        if best == 0:
            prediction = 1
        elif best == 1:
            prediction = -1
        elif best == 2:
            prediction = 0
        text += ', Prediction: '+str(prediction)

        if prediction == result:
            correct[i,0] = 1
            text += '  [O]\n'
        else:
            correct[i,0] = 0
            text += '  [X]\n'
        predictions[i,0] = prediction

    N = data_test.shape[0]
    cor = np.sum(correct)
    incor = N - cor
    text += '\nCorrect: '+str(cor)+'  Incorrect: '+str(incor)+' Percentage: '+str(cor*100/N)+'\n'
    text += 'Mean of Expected : ' + str(np.mean(probabilities)) + '\n'
    return predictions, probabilities, correct, text


