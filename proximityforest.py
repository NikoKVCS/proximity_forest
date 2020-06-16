import numpy as np
import os
from distancefuncs import disFuncs

class Node:
    def __init__(self, r=3, rand=None):
        self._r = r
        self._left = None
        self._right = None
        self._p = None
        self._q = None
        self._measure = 0
        self.prediction = None
        self._rand = rand
        if rand == None:
            self._rand = np.random
    
    def setChild(self, node, treeType):
        if treeType == 'left':
            self._left = node
        else:
            self._right = node

    def _gen_candidate_splitter(self, Y):
        num = disFuncs.getNumOfMeasures()
        measures = self._rand.choice(num,size=self._r,replace=False)
        exemplars = []
        for i in range(self._r):
            n_class = np.sum(Y,axis=0)
            n_class = np.where(n_class != 0)[0]

            [class1, class2] = self._rand.choice(n_class,size=2,replace=False)
            p_candidate = np.where(Y[:,class1] == 1)[0]
            q_candidate = np.where(Y[:,class2] == 1)[0]

            p = self._rand.choice(p_candidate)
            q = self._rand.choice(q_candidate)
            exemplars.append([p,q])

        return measures, exemplars
        
    def _gini(self, Pk_left, Pk_right, total_left, total_right):

        left_gini = 1 - np.sum(Pk_left**2)
        right_gini = 1 - np.sum(Pk_right**2)

        left_prop = total_left / (total_left + total_right)
        return left_prop * left_gini + (1 - left_prop) * right_gini

    def _gini_balanced(self, Pk_left, Pk_right, total_left, total_right, balance_tradeoff):

        left_gini = 1 - np.sum(Pk_left**2)
        right_gini = 1 - np.sum(Pk_right**2)

        left_prop = total_left / (total_left + total_right)
        gini = left_prop * left_gini + (1 - left_prop) * right_gini

        balance = abs(total_left - total_right) / (total_left + total_right)

        return (1-balance_tradeoff) * gini + balance_tradeoff * balance

    def build(self, task):
        self.nodeId = task.get("nodeId")
        X = task.get("X")
        Y = task.get("Y")
        self.prediction = np.sum(Y,axis=0) / np.sum(Y)
        if 1 in self.prediction:
            return ([], []), ([], [])

        measures, exemplars = self._gen_candidate_splitter(Y)
        n_class = np.sum(Y,axis=0)

        best_metric = 1
        best_splitter = 0
        
        X_left = []
        X_right = []
        y_left = []
        y_right = []

        for i in range(self._r):
            measure = measures[i]
            [p,q] = exemplars[i]
            p = X[p]
            q = X[q]
            sims = np.array([disFuncs.measure_by_index(measure, x, p) - disFuncs.measure_by_index(measure, x, q) for x in X], dtype='float')

            left_indices = np.where(sims <= 0)[0]
            right_indices = np.where(sims > 0)[0]
            total_left = len(left_indices)
            total_right = len(right_indices)
            Pk_left = np.sum(Y[left_indices],axis=0)/ total_left
            Pk_right = np.sum(Y[right_indices],axis=0)/ total_right
            metric = self._gini(Pk_left, Pk_right, total_left, total_right)
            
            if metric < best_metric:
                best_metric = metric
                best_splitter = i
                X_left = X[left_indices, :]
                y_left = Y[left_indices, :]
                X_right = X[right_indices, :]
                y_right = Y[right_indices, :]
        
        self._measure = measures[best_splitter]
        [p,q] = exemplars[best_splitter]
        self._p = X[p]
        self._q = X[q]

        return (X_left, y_left), (X_right, y_right)
        
    
    def buildTopTree(self, task, threshold, balance_tradeoff):
        self.task = task
        self.nodeId = task.get("nodeId")
        X = task.get("X")
        Y = task.get("Y")
        self.prediction = np.sum(Y,axis=0) / np.sum(Y)
        if len(Y) < threshold:
            return ([], []), ([], [])
        
        measures, exemplars = self._gen_candidate_splitter(Y)
        n_class = np.sum(Y,axis=0)

        best_metric = 1
        best_splitter = 0
        
        X_left = []
        X_right = []
        y_left = []
        y_right = []

        for i in range(self._r):
            measure = measures[i]
            [p,q] = exemplars[i]
            p = X[p]
            q = X[q]
            sims = np.array([disFuncs.measure_by_index(measure, x, p) - disFuncs.measure_by_index(measure, x, q) for x in X], dtype='float')

            left_indices = np.where(sims <= 0)[0]
            right_indices = np.where(sims > 0)[0]
            total_left = len(left_indices)
            total_right = len(right_indices)
            Pk_left = np.sum(Y[left_indices],axis=0)/ total_left
            Pk_right = np.sum(Y[right_indices],axis=0)/ total_right
            metric = self._gini_balanced(Pk_left, Pk_right, total_left, total_right, balance_tradeoff)
            
            if metric < best_metric:
                best_metric = metric
                best_splitter = i
                X_left = X[left_indices, :]
                y_left = Y[left_indices, :]
                X_right = X[right_indices, :]
                y_right = Y[right_indices, :]
        
        self._measure = measures[best_splitter]
        [p,q] = exemplars[best_splitter]
        self._p = X[p]
        self._q = X[q]

        return (X_left, y_left), (X_right, y_right)
        
    def predict_proba(self, x):

        if self._left is None or self._right is None:
            return self.prediction
        elif disFuncs.measure_by_index(self._measure, x, self._p)-disFuncs.measure_by_index(self._measure, x, self._q) > 0:
            return self._right.predict_proba(x)
        else:
            return self._left.predict_proba(x)

    def distribute(self, x):
        
        if self._left is None or self._right is None:
            return self.nodeId
        elif disFuncs.measure_by_index(self._measure, x, self._p)-disFuncs.measure_by_index(self._measure, x, self._q) > 0:
            return self._right.distribute(x)
        else:
            return self._left.distribute(x)
    
    def toObject(self):

        obj = dict()
        obj['prediction'] = self.prediction
        obj['_p'] = self._p
        obj['_q'] = self._q
        obj['_measure'] = self._measure
        obj['nodeId'] = self.nodeId
        obj['_left'] = None
        obj['_right'] = None

        if self._left is not None:
            obj['_left'] = self._left.toObject()

        if self._right is not None:
            obj['_right'] = self._right.toObject()
            
        return obj
    
    def fromObject(self, obj):

        nodeidMap = dict()
        nodeidMap[obj.get('nodeId')] = self

        self.prediction = obj.get('prediction')
        self._p = obj.get('_p')
        self._q = obj.get('_q')
        self._measure = obj.get('_measure')
        self.nodeId = obj.get('nodeId')

        if obj.get('_left') is not None:
            self._left = Node(self._r, self._rand)
            d1 = self._left.fromObject(obj.get('_left'))
            nodeidMap.update(d1)
        
        if obj.get('_right') is not None:
            self._right = Node(self._r, self._rand)
            d2 = self._right.fromObject(obj.get('_right'))
            nodeidMap.update(d2)

        return nodeidMap



class ProximityForest:
    def __init__(self, modelName, r=3, random_seed=None):
        self._r = r
        self._modelpath = os.path.join("model", modelName + ".model.npy")
        self._checkpointpath = os.path.join("model", modelName + ".ckpt.npy")
        self._toptreepath = os.path.join("model", modelName + "_top_tree")
        self._forest = []
        self._task_queue = []
        self._nodeIdMap = dict()
        if random_seed is not None:
            self._rand = np.random.RandomState(random_seed)
        else:
            self._rand = np.random
        pass

    def save_model(self):
        objects = np.array([tree.toObject() for tree in self._forest])
        np.save(self._modelpath, objects)

        dic = {"_task_queue":self._task_queue}
        ckpt = np.array([dic])
        np.save(self._checkpointpath,ckpt)
    
    def load_model(self):
        if os.path.exists(self._modelpath) == False:
            return

        objects  = np.load(self._modelpath, allow_pickle=True)
        forest = []
        for obj in objects:
            tree = Node(self._r, self._rand)
            nodeIdMap = tree.fromObject(obj)
            forest.append(tree)
            self._nodeIdMap = nodeIdMap
        self._forest = forest
        if os.path.exists(self._checkpointpath) == False:
            return
        dic = np.load(self._checkpointpath, allow_pickle=True)[0]
        self._task_queue = dic.get('_task_queue')

    def save_top_tree(self, tree):
        model = self._toptreepath + "_model.npy"
        ckpt_path = self._toptreepath +  "_ckpt.npy"

        objects = np.array([tree.toObject()])
        np.save(model, objects)

        dic = {"_task_queue":self._task_queue}
        ckpt = np.array([dic])
        np.save(ckpt_path, ckpt)
        pass
    
    def load_top_tree(self):
        model = self._toptreepath + "_model.npy"
        ckpt_path = self._toptreepath +  "_ckpt.npy"
        
        if os.path.exists(model) == False or os.path.exists(ckpt_path) == False:
            return
        
        obj  = np.load(model, allow_pickle=True)[0]
        tree = Node(self._r, self._rand)
        self._nodeIdMap = tree.fromObject(obj)
        self._forest.append(tree)

        dic = np.load(ckpt_path, allow_pickle=True)[0]
        self._task_queue = dic.get('_task_queue')
        pass
            
    def _bagging(self, X, Y):
        index = np.array(list(set(self._rand.choice(len(Y), size=len(Y)))))
        return X[index], Y[index]

    def _createTask(self, X, Y, nodeId, parentNodeId, tree_type):
        buildInfo = dict()
        buildInfo['X'] = X
        buildInfo['Y'] = Y
        buildInfo['nodeId'] = nodeId
        buildInfo['parentNodeId'] = parentNodeId
        buildInfo['tree_type'] = tree_type
        return buildInfo

    def train(self, X, Y, subset_size, bucket_size, n_estimators=10, share_toptree=False):
        self.load_model()

        while len(self._forest) < n_estimators:
            if len(self._task_queue) == 0 and share_toptree:
                self.load_top_tree()

            if len(self._task_queue) == 0:
                # Phase1 : Build top tree 
                self._nodeIdMap = dict()

                T_x, T_y = self._bagging(X, Y)

                nodeId = 0

                # Build top tree phase
                index = self._rand.choice(len(T_y),size=subset_size,replace=False)
                task = self._createTask(T_x[index], T_y[index], nodeId, -1, "left")
                self._task_queue.append(task)

                tree = None
                print( len(T_y),bucket_size , subset_size)
                threshold = bucket_size * subset_size / len(T_y)

                while len(self._task_queue) > 0:
                    
                    task = self._task_queue.pop()
                    node = Node(self._r, self._rand)
                    (X_left, y_left), (X_right, y_right) = node.buildTopTree(task, threshold, balance_tradeoff=0.5)
                    if task.get('parentNodeId') >= 0:
                        parentId = task.get('parentNodeId')
                        parent = self._nodeIdMap.get(parentId)
                        parent.setChild(node, task.get('tree_type'))
                    else:
                        tree = node

                    self._nodeIdMap[task.get('nodeId')] = node

                    if len(y_left) > 0 and len(y_right) > 0:
                        nodeId += 1
                        task_left = self._createTask(X_left, y_left, nodeId, parentNodeId=task.get('nodeId'), tree_type="left")
                        nodeId += 1
                        task_right = self._createTask(X_right, y_right, nodeId, parentNodeId=task.get('nodeId'), tree_type="right")
                        
                        self._task_queue.append(task_left)
                        self._task_queue.append(task_right)

                # Phase2 : Distribute

                subsets = dict()
                for i in range(len(T_y)):
                    node_id = tree.distribute(T_x[i])
                    subset_x = []
                    subset_y = []
                    if node_id in subsets.keys():
                        dic = subsets.get(node_id)
                        subset_x = dic.get('subset_x')
                        subset_y = dic.get('subset_y')
                    subset_x.append(T_x[i])
                    subset_y.append(T_y[i])
                    dic = dict()
                    dic['subset_x'] = subset_x
                    dic['subset_y'] = subset_y
                    subsets[node_id] = dic
                
                for node_id in subsets.keys():
                    dic = subsets.get(node_id)
                    subset_x = dic.get('subset_x')
                    subset_y = dic.get('subset_y')

                    # 把top tree的叶子替换为bottom tree的root节点
                    node = self._nodeIdMap.get(node_id)
                    task = node.task
                    task['X'] = np.array(subset_x)
                    task['Y'] = np.array(subset_y)
                    self._task_queue.append(task)
                
                if share_toptree:
                    self.save_top_tree(tree)

                self._forest.append(tree)
                self.save_model()
                pass
            
            # Phase3 : Construct bottom tree

            nodeId = self._task_queue[-1].get('nodeId') + 1

            while len(self._task_queue) > 0:

                if nodeId % 10 == 0:
                    self.save_model()
                
                task = self._task_queue.pop()
                node = Node(self._r, self._rand)
                (X_left, y_left), (X_right, y_right) = node.build(task)

                parentId = task.get('parentNodeId')
                parent = self._nodeIdMap.get(parentId)
                parent.setChild(node, task.get('tree_type'))

                self._nodeIdMap[task.get('nodeId')] = node

                if len(y_left) > 0 and len(y_right) > 0:
                    nodeId += 1
                    task_left = self._createTask(X_left, y_left, nodeId, parentNodeId=task.get('nodeId'), tree_type="left")
                    nodeId += 1
                    task_right = self._createTask(X_right, y_right, nodeId, parentNodeId=task.get('nodeId'), tree_type="right")
                    
                    self._task_queue.append(task_left)
                    self._task_queue.append(task_right)

            self.save_model()
            pass

        pass
    
    def test(self, X, Y):
        self.load_model()
        error = 0
        trade = 0
        win_trade = 0
        for i in range(len(Y)):
            y = self.predict(X[i])
            if np.argmax(y) != np.argmax(Y[i]):
                error += 1
            if np.argmax(y) == 1:
                trade += 1
                if np.argmax(y) == np.argmax(Y[i]):
                    win_trade += 1
        print("accuracy:", 1-error/len(Y))
        print(win_trade,trade, " winrate:", win_trade/trade)

        return 1-error/len(Y)
    
    def predict(self, x):
        probs = np.mean([tree.predict_proba(x) for tree in self._forest], axis=0)
        return probs


if __name__ == "__main__":
    
    dataset = np.load('dataset_short.npy', allow_pickle=True)[0]
    traindata = dataset.get('traindata')
    testdata = dataset.get('testdata')

    proxforest = ProximityForest('promixityforest_short', r=2)
    proxforest.train(traindata.get('data_input'), traindata.get('data_output'), 2000, 1500, n_estimators=1)
    
    X = testdata.get('data_input')
    Y = testdata.get('data_output')

    proxforest.test(X,Y)

    pass
