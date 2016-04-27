# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------

# Multi-layer Perceptron

# By Basile Garcia, based on a Nicolas Rougier's program,  distributed under the terms of the BSD License.

# -----------------------------------------------------------------------------

# This is an implementation of the multi-layer perceptron with retropropagation

# The network learns with transcoded rhythm structures, tries to recognize the song at first, then tries to
#guess the music genre the song belongs to.

# -----------------------------------------------------------------------------
import rpy2.robjects as rob
import numpy as np
from scipy import stats
class MLP:

    """ Multi-layer perceptron class. """

    def __init__(self, *args):

        """ Initialization of the perceptron with given sizes.  """

        self.shape = args
        n = len(args)

        # Build layers
        self.layers = []

        # Input layer (+1 unit for bias)
        self.layers.append(np.ones(self.shape[0]+1))

        # Hidden layer(s) + output layer
        for i in range(1, n):

            self.layers.append(np.ones(self.shape[i]))

        # Build weights matrix (randomly between -0.25 and +0.25)
        self.weights = []

        for i in range(n-1):

            self.weights.append(np.zeros((self.layers[i].size,

                                         self.layers[i+1].size)))

        # dw will hold last change in weights (for momentum)
        self.dw = [0, ]*len(self.weights)

        # Reset weights
        self.reset()

    def reset(self):

        """ Reset weights """

        for i in range(len(self.weights)):

            Z = np.random.random((self.layers[i].size, self.layers[i+1].size))

            self.weights[i][...] = (2*Z-1)*0.25

    def propagate_forward(self, data):

        """ Propagate data from input layer to output layer. """

        # Set input layer

        self.layers[0][0:-1] = data

        # Propagate from layer 0 to layer n-1 using sigmoid as activation function

        for i in range(1, len(self.shape)):

            # Propagate activity

            self.layers[i][...] = self.sigmoid(np.dot(self.layers[i-1], self.weights[i-1]))

        # Return output

        return self.layers[-1]


    def propagate_backward(self, target, lrate=0.1, momentum=0.0):

        """ Back propagate error related to target using lrate. """

        deltas = []

        # Compute error on output layer

        error = target - self.layers[-1]
        delta = error * self.dsigmoid(self.layers[-1])
        deltas.append(delta)

        # Compute error on hidden layers

        for i in range(len(self.shape)-2, 0, -1):

            delta = np.dot(deltas[0], self.weights[i].T) * self.dsigmoid(self.layers[i])
            deltas.insert(0, delta)

        # Update weights

        for i in range(len(self.weights)):

            layer = np.atleast_2d(self.layers[i])
            delta = np.atleast_2d(deltas[i])
            dw = np.dot(layer.T, delta)
            self.weights[i] += lrate*dw + momentum*self.dw[i]
            self.dw[i] = dw

        # Return error

        return np.sum(error**2)

    @staticmethod
    def sigmoid(x):

        """ Sigmoid like function using tanh """

        return np.tanh(x)

    @staticmethod
    def dsigmoid(x):

        """ Derivative of sigmoid above """

        return 1.0 - x**2


# -----------------------------------------------------------------------------


class ChiefOperatingOfficer(object):

    def __init__(self, dataset, presentation_number, *hidden_number):
        self.scores = []
        self.scores2 = []
        self.fail = 0
        self.win = 0
        self.fail2 = 0
        self.win2 = 0
        self.dataset = dataset
        self.input_number = np.size(self.dataset[0]['input'])
        self.output_number = np.size(self.dataset[0]['output'])
        self.id = 0

        if hidden_number:
            self.hidden_number = hidden_number
        else:
            self.hidden_number = self.input_number

        self.n_samples = np.size(self.dataset)

        self.presentation_number = presentation_number

        # Create the network

        self.network = MLP(self.input_number, self.hidden_number, self.output_number)
        self.ask_the_network_to_learn()


    def test1(self,sample):

        a = self.network.propagate_forward(sample)
        a2 = a.tolist()
        b = np.max([a2])
        c = a2.index(b)
        c = int(c)
        k = samples[c]['title']
        if np.all(sample == samples[c]['input']):
            print "Answer is right  ",k
            self.win += 1
        else:
            print "Answer is wrong ",k
            self.fail += 1

    def test2(self,sample):

        a = self.network.propagate_forward(sample)
        a2 = a.tolist()
        b = np.max([a2])
        c = a2.index(b)
        c = int(c)
        e = samples[0]['output']
        f = samples[5]['output']
        g = samples[10]['output']

        print "-----------"
        for i in range(12):
            if np.all(sample == test[i]['input']) and np.all(test[i]['output'] == e):
                print " The tested track belongs to rock music. \n"
                if c == 0 :
                    print " The ANN thinks it's rock. \n"
                    self.win2 += 1
                else:
                    self.fail2 += 1
                    print " The ANN isn't sure about this track. \n"
            elif np.all(sample == test[i]['input']) and np.all(test[i]['output'] == f):
                print " The tested track belongs to reggae music.\n"
                if c == 1 :
                    print " The ANN thinks it's reggae. \n"
                    self.win2 += 1
                else:
                    self.fail2 += 1
                    print " The ANN isn't sure about this track. \n"
            elif np.all(sample == test[i]['input']) and np.all(test[i]['output'] == g):
                print " The tested track belongs to bossa nova music.\n"
                if c == 2 :
                    print " The ANN thinks it's bossa nova\n"
                    self.win2 += 1
                else:
                    self.fail2 += 1
                    print " The ANN isn't sure about this track. \n"




    def ask_the_network_to_learn(self):

        self.network.reset()

        # Create order with index that will be randomized
        order = np.arange(self.n_samples)


        # Each presentation, the perceptron receives all the inputs and outputs of the dataset (but in
        # a different order)

        for i in range(self.presentation_number):

            # Each presentation, use a different order
            np.random.shuffle(order)

            errors = np.ones(self.n_samples)

            for j in order:
                a = self.network.propagate_forward(samples['input'][j])
                errors[j] = self.network.propagate_backward(samples['output'][j])


            print "Presentation n°{i}. Error mean: {e_min}. Error max: {e_max}."\
                .format(i=i, e_min=np.mean(errors), e_max=np.max(errors))

# -----------------------------------------------------------------------------

class Suprachief(object):

    def __init__(self,x):
        self.ann = []
        self.get_tracks(x)
        self.create_networks()

    def get_tracks(self,x):
        if x == 0:
            array = [line.rstrip('\n') for line in open('tracks2.txt')]
            i = 0
            for line in array:
                c = line.split(',')
                samples[i]['input'] = [float(y) for y in c[0]]
                samples[i]['output'] = [float(y) for y in c[1]]
                samples[i]['title'] = c[2]
                tracks.append(c[2])
                i += 1
        else:
            array = [line.rstrip('\n') for line in open('tracks.txt')]
            i = 0
            for line in array:
                c = line.split(',')
                samples[i]['input'] = [float(y) for y in c[0]]
                samples[i]['output'] = [float(y) for y in c[1]]
                samples[i]['title'] = c[2]
                tracks.append(c[2])
                i += 1
            testarray = [line.rstrip('\n') for line in open('test.txt')]
            i = 0
            for line in testarray:
                c = line.split(',')
                test[i]['input'] = [float(y) for y in c[0]]
                test[i]['output'] = [float(y) for y in c[1]]

                i += 1

    def create_networks(self):

        for p in range(0,26):
            self.ann.append('Ann {l}'.format(l=p))
            self.ann[p] = ChiefOperatingOfficer(samples, 1,2)
            self.ann[p].id = 'ANN {nb}'.format(nb=p)

        for p in range(26,51):
            self.ann.append('Ann {l}'.format(l=p))
            self.ann[p] = ChiefOperatingOfficer(samples,1,40)
            self.ann[p].id = 'ANN {nb}'.format(nb=p)
        for p in range(51,76):
            self.ann.append('Ann {l}'.format(l=p))
            self.ann[p] = ChiefOperatingOfficer(samples,100,2)
            self.ann[p].id = 'ANN {nb}'.format(nb=p)
        for p in range(76,101):
            self.ann.append('Ann {l}'.format(l=p))
            self.ann[p] = ChiefOperatingOfficer(samples,100,40)
            self.ann[p].id = 'ANN {nb}'.format(nb=p)


#------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    tracks = []
    samples = np.zeros(12, dtype=[('input',  float, 32), ('output', float, 12),('title',str,50)])
    samples2 = np.zeros(12, dtype=[('input',  float, 32), ('output', float, 12),('title',str,50)])

    print "\n"
    print "-"*10
    print "Learning "
    print "-"*10
    print "\n"
#------------------------------------------------------------------------------------------------
#Condition 1
    sc = Suprachief(0)
    scores = []
    for i in range(0,101):
        print "*-----------------------------------------------------------------*"
        print sc.ann[i].id, ':'


        samples2['input'] = samples['input']
        random = samples2['input']
        np.random.shuffle(random)
        a = 0
        for x in random:
            a += 1
            print "Track n°{a}".format(a=a)
            sc.ann[i].test1(x)
        scores.append(sc.ann[i].win)

#-----------------------------------------------------------------------------------------------
#Condition 2
    scores2 =[]
    samples = np.zeros(12, dtype=[('input',  float, 32), ('output', float, 3),('title',str,50)])
    samples2 = np.zeros(12, dtype=[('input',  float, 32), ('output', float, 3),('title',str,50)])
    test = np.zeros(12, dtype=[('input',  float, 32), ('output', float, 3),('title',str,50)])
    sc = Suprachief(1)

    for i in range(0,101):
        print "*-----------------------------------------------------------------*"
        print sc.ann[i].id, ':'
        samples2['input'] = test['input']
        random = samples2['input']
        np.random.shuffle(random)
        a = 0
        for x in random:
            a += 1
            print "Track n°{a}".format(a=a)
            sc.ann[i].test2(x)
        scores2.append(sc.ann[i].win2)

#------------------------------------------------------------------------------------------------
#Stats
#condition 1 / ANOVA

    tab = np.zeros(101, dtype=[('capacités cognitives',  float, 1), ('entraînement', float, 1),('performances',float,1)])

    for i in range(0,26):
        tab[i]['capacités cognitives'] = 0
        tab[i]['entraînement'] = 0
        tab[i]['performances'] = scores[i]
    for i in range(26,51):
        tab[i]['capacités cognitives'] = 1
        tab[i]['entraînement'] = 0
        tab[i]['performances'] = scores[i]
    for i in range(51,76):
        tab[i]['capacités cognitives'] = 0
        tab[i]['entraînement'] = 1
        tab[i]['performances'] = scores[i]
    for i in range(76,101):
        tab[i]['capacités cognitives'] = 1
        tab[i]['entraînement'] = 1
        tab[i]['performances'] = scores[i]

    print tab

    print stats.f_oneway(tab[0:26],tab[26:51],tab[51:76],tab[76:100])
    r = rob.r

    print r.anova(tab[0:26],tab[26:51],tab[51:76],tab[76:100])

#condition 1 / MAnn withney
    # for i in range(26,51):
    #     c.append(scores[i])
    #
    # for i in range(51,76):
    #     v.append(scores[i])
    #
    #
    # for i in range(76,101):
    #     d.append(scores[i])
    #


    j = []
    c = []
    v = []
    d = []
    for i in range(0,26):
        j.append(scores[i])

    for i in range(76,101):
        d.append(scores[i])
    #
    print stats.shapiro(j)
    print stats.shapiro(d)
    print stats.levene(j,d)
    print stats.mannwhitneyu(j,d)


  # CONDITION 2 / ANOVA-----------------------------------------------------------------------------------


    tab = np.zeros(101, dtype=[('capacités cognitives',  float, 1), ('entraînement', float, 1),('performances',float,1)])

    for i in range(0,26):
        tab[i]['capacités cognitives'] = 0
        tab[i]['entraînement'] = 0
        tab[i]['performances'] = scores2[i]
    for i in range(26,51):
        tab[i]['capacités cognitives'] = 1
        tab[i]['entraînement'] = 0
        tab[i]['performances'] = scores2[i]
    for i in range(51,76):
        tab[i]['capacités cognitives'] = 0
        tab[i]['entraînement'] = 1
        tab[i]['performances'] = scores2[i]
    for i in range(76,101):
        tab[i]['capacités cognitives'] = 1
        tab[i]['entraînement'] = 1
        tab[i]['performances'] = scores2[i]

    

    print stats.f_oneway(tab[0:26],tab[26:51],tab[51:76],tab[76:100])
    print r.anova(tab[0:26],tab[26:51],tab[51:76],tab[76:100])
    #CONDITION 2 / Mann - Withney

    j = []
    c = []
    v = []
    d = []
    for i in range(0,26):
        j.append(scores2[i])

    for i in range(76,101):
        d.append(scores2[i])
    #
    print stats.shapiro(j)
    print stats.shapiro(d)
    print stats.levene(j,d)
    print stats.mannwhitneyu(j,d)



    # j = []
    # c = []
    # v = []
    # d = []
    #
    # for i in range(0,26):
    #     j.append(scores2[i])
    # j = np.mean(j)
    #
    # for i in range(26,51):
    #     c.append(scores2[i])
    # c = np.mean(c)
    #
    # for i in range(51,76):
    #     v.append(scores2[i])
    # v = np.mean(v)
    #
    # for i in range(76,101):
    #     d.append(scores2[i])
    # d = np.mean(d)
    #
    # print j,c,v,d,
    #


