import pickle
import os

def dir_to_pickle():
    directory_path = 'rawdata/logos'
    list_of_file_paths = [f for f in os.listdir(directory_path)]
    class_condition_labels = [0] * len(list_of_file_paths)
    print(list_of_file_paths[:3])
    mypickle = {"Filenames": list_of_file_paths, "Labels" : class_condition_labels}
    
    with open('rawdata/mypickle.pickle', 'wb') as f:
        pickle.dump(mypickle, f)

def open_pickle():
    with open('rawdata/mypickle.pickle', "rb") as f:
        data = pickle.load(f)
    return data

dir_to_pickle()
data = open_pickle()
print(data["Labels"][:3])