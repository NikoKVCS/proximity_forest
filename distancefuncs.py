import numpy as np
from distance_metrics import lcs # pip3 install distance-metrics

class DistanceFuncs:
    def getNumOfMeasures(self):
        return 3

    def measure_by_index(self,index,x,y):
        if index == 0:
            return self.cos_sim(x,y)
        elif index == 1:
            return self.euclideanDistance(x,y)
        elif index == 2:
            return self.LCS(x,y)
        return 0

    def euclideanDistance(self, x1,x2):
        return np.linalg.norm(x1 - x2)

    def LCS(self,x,y):
        return -lcs.llcs(x,y)

    def cos_sim(self,vector_a, vector_b):
        vector_a = np.mat(vector_a)
        vector_b = np.mat(vector_b)
        num = float(vector_a * vector_b.T)
        denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
        cos = num / denom
        sim = 0.5 + 0.5 * cos
        return -sim

disFuncs = DistanceFuncs()
