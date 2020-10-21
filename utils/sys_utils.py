import os
import pickle


def make_directories(path, dirs):
    if type(dirs) == str:
        dirs = [dirs]
    for d in dirs:
        if not os.path.isdir(os.path.join(path, d)):
            os.makedirs(os.path.join(path, d))


def pickle_save(obj, path, filename):
    with open(os.path.join(path, filename + '.pickle'), 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
