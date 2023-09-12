import pickle
import numpy as np

# with open('rawdata/tripletmining.pickle', 'rb') as f:
#     mypickle = pickle.load(f)

# class_condition_labels = mypickle["Labels"]
# one_hot_labels = np.eye(64)[class_condition_labels]

# mypickle["Labels"] = one_hot_labels.tolist()

# with open('rawdata/tripletmining_onehot.pickle', 'wb') as f:
#     pickle.dump(mypickle, f)


def open_pickle():
    with open('rawdata/tripletmining_onehot.pickle', "rb") as f:
        data = pickle.load(f)
    return data


data = open_pickle()
print(len(data["Labels"][0]))