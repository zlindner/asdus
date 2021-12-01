import pandas as pd
import numpy as np
import pickle
from sklearn.cluster import KMeans

def init_model(path='./features.csv', retrain=False):
    if not retrain:
        try:
            return pickle.load(open('kmeans.pkl', 'rb')) 
        except:
            pass

    df = pd.read_csv(path)
    print(df[['X1', 'X2']])
    X = np.array(df[['X1', 'X2']])

    model = KMeans(n_clusters=2)
    model.fit(X)

    pickle.dump(model, open('kmeans.pkl', 'wb'))

    return model


def predict(model, X1, X2):
    return model.predict(X=np.array([[X1, X2]]))


if __name__ == '__main__':
    model = init_model()
    print(predict(model, 0, 0))
