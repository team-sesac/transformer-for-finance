import pickle
from tqdm import tqdm
import numpy as np

def cat_ndarrs():
    with open('concatenated_array500.pkl', 'rb') as file:
        ndarr1 = pickle.load(file)

    with open('concatenated_array1000.pkl', 'rb') as file:
        ndarr2 = pickle.load(file)

    with open('concatenated_array1500.pkl', 'rb') as file:
        ndarr3 = pickle.load(file)

    with open('concatenated_array2000.pkl', 'rb') as file:
        ndarr4 = pickle.load(file)

    with open('concatenated_array2500.pkl', 'rb') as file:
        ndarr5 = pickle.load(file)

    final = ndarr1.copy()

    for curr in tqdm([ndarr2, ndarr3, ndarr4, ndarr5]):
        final = np.append(final, curr, axis=0)

    with open('concatenated_array_all.pkl', 'wb') as file:
        pickle.dump(final, file)


if __name__ == '__main__':
    cat_ndarrs()