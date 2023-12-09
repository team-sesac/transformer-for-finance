import pandas as pd

from src.transformer.double_axis_dataset import DoubleAxisDataProcessor

#
# # 예시 데이터 프레임 생성 (실제 데이터에 맞게 수정해야 함)
# df1 = pd.DataFrame({
#     'axis': ['2018-01-01', '2018-01-03'],
#     'open': [100, 110],
#     'close': [105, 115],
#     'volume': [1000, 1200]
# })
#
# df2 = pd.DataFrame({
#     'axis': ['2018-01-01', '2018-01-02'],
#     'open': [95, 112],
#     'close': [100, 118],
#     'volume': [900, 1100]
# })
#
# df3 = pd.DataFrame({
#     'axis': ['2018-01-01', '2018-01-02'],
#     'open': [102, 108],
#     'close': [107, 113],
#     'volume': [950, 1300]
# })
#
# # 데이터프레임 리스트 생성
# dfs = [df1, df2, df3]  # 나머지 7개 데이터프레임도 추가해야 함
#
# # 초기 데이터프레임을 첫 번째 데이터프레임으로 설정
# merged_df = dfs[0]
#
# # 나머지 데이터프레임과 순서대로 'axis' 열을 기준으로 병합
# for df in dfs[1:]:
#     merged_df = pd.merge(merged_df, df, on='axis', how='outer')
#
# # 'axis' 열을 기준으로 오름차순 정렬
# merged_df = merged_df.sort_values(by='axis')
#
#
# # 결과 확인
# print(merged_df)

if __name__ == '__main__':
    base_dir = '../../data/tf_dataset/'
    file_paths = ['0_삼성전자_2010.csv',
                  '1_LG에너지솔루션_2010.csv',
                  '2_SK하이닉스_2010.csv',
                  '3_삼성바이오로직스_2010.csv']
    file_paths = [base_dir+i for i in file_paths]

    usecols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume',
               'Change', 'pct_change', 'volume_adi', 'volume_obv', 'volume_cmf']

    data_processor = DoubleAxisDataProcessor(file_paths=file_paths, usecols=usecols)
    merged_df = data_processor.outer_join_df()
    print('here')

