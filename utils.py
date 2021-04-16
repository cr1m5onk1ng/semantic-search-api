import pickle
import os

def load_file(path):
    file_name = open(path, 'rb')
    d = pickle.load(file_name)
    file_name.close()
    return d

def save_file(file, path, name):
    if not os.path.exists(path):
        os.makedirs(path)
    file_name = open(os.path.join(path, name), 'wb')
    pickle.dump(file, file_name)
    file_name.close()