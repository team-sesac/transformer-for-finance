import os
import csv

import pandas as pd
from tqdm import tqdm


# def merge_csv_files_alternative(folder_path, output_file):
#
#     # 폴더 내의 모든 CSV 파일을 읽어들임
#     csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
#     print(f"{csv_files=}")
#     # half_csv_files = csv_files[len(csv_files)//2:]
#     # half_csv_files = csv_files[:5]
#     # print(f"start from : {csv_files[len(csv_files)//2]}")
#
#     # 결과 파일을 쓰기 모드로 열기
#     with open(output_file, 'w', newline='', encoding='utf-8') as output_csv:
#         writer = csv.writer(output_csv)
#
#         # 각 CSV 파일을 읽어들여 결과 파일에 쓰기
#         for csv_file in tqdm(csv_files):
#             file_path = os.path.join(folder_path, csv_file)
#             with open(file_path, 'r', newline='', encoding='utf-8') as input_csv:
#                 reader = csv.reader(input_csv)
#                 # 헤더를 제외한 내용을 복사
#                 header_written = False
#                 for row in reader:
#                     if not header_written:
#                         writer.writerow(row)
#                         header_written = True
#                     else:
#                         writer.writerow(row)
#
#     print(f"Successfully merged {len(csv_files)} CSV files into {output_file}")
#

def merge_csv_files_alternative(folder_path, output_file):

    # 폴더 내의 모든 CSV 파일을 읽어들임
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    print(f"{csv_files=}")

    # 결과 파일을 추가 모드로 열기
    with open(output_file, 'w', newline='', encoding='utf-8') as output_csv:
        writer = csv.writer(output_csv)

        # 각 CSV 파일을 읽어들여 결과 파일에 쓰기
        for index, csv_file in enumerate(tqdm(csv_files)):
            file_path = os.path.join(folder_path, csv_file)
            with open(file_path, 'r', newline='', encoding='utf-8') as input_csv:
                reader = csv.reader(input_csv)

                # 헤더를 포함하거나 건너뛰기
                if index == 0:
                    header = next(reader)
                    writer.writerow(header)
                else:
                    next(reader, None)

                # 나머지 내용을 복사
                for row in reader:
                    writer.writerow(row)

    print(f"Successfully merged {len(csv_files)} CSV files into {output_file}")



# 예제 사용
# folder_path = './concat_themed_stocks'  # 실제 폴더 경로로 변경
# output_file = './concat_themed_stocks/all_stocks_all.csv'  # 실제 결과 파일 경로로 변경
folder_path = './final_entry'  # 실제 폴더 경로로 변경
output_file = './final_entry_and_xgboost/all_final_entry_stock.csv'  # 실제 결과 파일 경로로 변경
merge_csv_files_alternative(folder_path, output_file)


