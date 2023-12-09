import os
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm

def get_all_files_in_folder(folder_path):
    file_list = []

    # 지정된 폴더 안의 모든 파일 및 폴더를 가져옵니다.
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_list.append(os.path.join(root, file))

    return file_list


def concat_all_files(files):
    # 각 파일을 읽어들여 데이터프레임으로 변환하고 NumPy 배열로 변환합니다.
    # arrays = [pd.read_csv(file).to_numpy() for file in files]
    curr_ndarray = pd.read_csv(files[2000], index_col=0).to_numpy()

    for file in tqdm(files[2001:]):
        curr = pd.read_csv(file, index_col=0).to_numpy()
        curr_ndarray = np.append(curr_ndarray, curr, axis=0)

    return curr_ndarray


# get_all_files_in_folder 함수로 가져온 파일 리스트를 사용하여 데이터를 합칩니다.
files = get_all_files_in_folder('../../data/tf_dataset')
concatenated_array = concat_all_files(files)

with open('concatenated_array2500.pkl', 'wb') as file:
    pickle.dump(concatenated_array, file)


# with open('concatenated_array.pkl', 'rb') as file:
#     loaded_data = pickle.load(file)

print('here')