    # def greedyDeterministic(self):
    #     G = Greedy(self.dis_matrix, self.reg)
    #     rep_matrix, obj_func_value = G.deterministic()
    #     return rep_matrix, len(rep_matrix), obj_func_value

    # def greedyRandomized(self):
    #     G = Greedy(self.dis_matrix, self.reg)
    #     rep_matrix, obj_func_value = G.randomized()
    #     return rep_matrix, len(rep_matrix), obj_func_value
        
import numpy as np
from random import shuffle
from math import exp


class Greedy(object):

    """
    :param dis_matrix:  dis-similarity matrix.
    :param reg:         regularization parameter.
    """
    def __init__(self, dis_matrix, reg):
        self.dis_matrix = np.matrix(dis_matrix)
        self.reg = reg
        self.M = dis_matrix.shape[0]
        self.N = dis_matrix.shape[1]

    def objFunction(self, S):
        """
        This function calculates the objective function value for the current number of slected representatives.
        :param S: current set of representatives.
        :returns: objective function value.
        """
        N = len(S)

        representativeness = exp(-1 * np.mean(np.min(self.dis_matrix[S, :], axis=0)))

        diversity = np.sum(self.dis_matrix[S, :][:, S]) / (N * (N-1)) if N > 1 else 0

        sparsity = - self.reg * N

        # print(representativeness, diversity, sparsity)

        value = representativeness + diversity + sparsity
        return value

    def deterministic(self):
        """
        This function runs deterministic greedy algorithm to find the representatives of the data.
        :param : None.
        :returns: representative set and objective function value.
        """

        X = []
        Y= np.arange(self.M)
        num = np.arange(self.M)
        shuffle(num)
        # print("Deterministic algorithm running....")
        for i in num:
            # print("itreation : ", i+1)

            # add new data point to the initially empty set X.
            newX = X + [i]

            # remove same data point as above from set of initially all data points.
            newY = [j for j in Y if j != i]

            # function value for empty set, initial condition.
            # calculate the cost of adding the random data point to X, as 'a'.
            # calculate the cost of removing the random data point from Y, as 'b'.
            if len(X) == 0:
                a = -self.objFunction(newX)
            else:
                a = self.objFunction(newX) - self.objFunction(X)
            b = self.objFunction(newY) - self.objFunction(Y)

            # add datapoint to X, if cost of adding more, else remove from Y.
            if a >= b:
                X = newX

            else:
                Y = newY

        # calculate the final objective function value.
        obj_func_value = self.objFunction(X)

        return X, obj_func_value

    def randomized(self):
        """
        This function runs randomized greedy algorithm to find the representatives of the data.
        :param : None.
        :returns: representative set and objective function value.
        """

        X = []
        Y = np.arange(self.M)
        num = np.arange(self.M)
        shuffle(num)
        # print("Randomized algorithm running....")
        for i in num:
            # print("itreation : ", i + 1)

            # add new data point to the initially empty set X.
            newX = X + [i]

            # remove same data point as above from set of initially all data points.
            newY = [j for j in Y if j != i]

            # function value for empty set, initial condition.
            # calculate the cost of adding the random data point to X, as 'a'.
            # calculate the cost of removing the random data point from Y, as 'b'.
            if len(X) == 0:
                a = -self.objFunction(newX)
            else:
                a = self.objFunction(newX) - self.objFunction(X)
            b = self.objFunction(newY) - self.objFunction(Y)

            # take only positive a, b values
            a_dash = max(0, a)
            b_dash = max(0, b)

            values = [1, 0]

            # add the datapoint to x, if both a, b are 0.
            if a_dash == 0 and b_dash == 0:
                X = newX

            # else, add to X or Y, based on the probability of a, b out of total.
            else:
                a_prob = a_dash / (a_dash + b_dash)
                b_prob = 1 - a_prob
                if np.random.choice(values, p=[a_prob, b_prob]):
                    X = newX

                else:
                    Y = newY

        # calculate the final objective funtion value.
        obj_func_value = self.objFunction(X)

        return X, obj_func_value