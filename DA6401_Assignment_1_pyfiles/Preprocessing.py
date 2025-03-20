import numpy  as np

class OneHotEncoder:

    def __init__(self,x, y):
        self.y = y
        self.x = x
        #self.num_class = num_class
        self.onehot_encode()

    def onehot_encode(self):
        onehot = np.zeros((self.x.shape[0], 10))

        for i, j in zip(range(len(self.x)), self.y):
            onehot[i, j] = 1
        return onehot.T

class Normalize:

    def __init__(self, unprocessed_X):
        self.unprocessed_X = unprocessed_X
        self.Norm_reshape()

    def Norm_reshape(self):
        X_norm = np.reshape(self.unprocessed_X,(self.unprocessed_X.shape[0],784)).T/255
        X_norm= np.array(X_norm)

        return X_norm


class TrainValSplit:
    def __init__(self, X_train, y_train, Val_split_ratio = 0.9):
        self.X = X_train
        self.y = y_train
        self.vsr = Val_split_ratio

    def Apply_split(self):
        np.random.seed(0)

        i = np.random.permutation(len(self.X))
        split = int(self.X.shape[0] * (1 - self.vsr))

        train = i[:split]
        val = i[split:]

        train_X = self.X[train]
        val_X = self.X[val]

        train_y = self.y[train]
        val_y = self.y[val]

        return train_X , train_y, val_X, val_y