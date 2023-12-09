from src.transformer.data_loader.double_axis_dataset import DoubleAxisDataProcessor
from src.transformer.data_loader.single_axis_dataset import TimeSeriesDataset
from src.transformer.model.CrossAttn import CrossAttnModel
from src.transformer.train.train_stock import train
from torch.utils.data import DataLoader

from config import Config

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

def test():
    import numpy as np

    # 주어진 정보
    y_row = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    len_feature_columns = 4
    label_columns = [0, 2]

    # 한 줄 for문으로 처리
    # result = np.array([y_row[i:i + len_feature_columns][label_columns] for i in range(0, len(y_row), len_feature_columns)])
    result = np.array([y_row[i:i + len_feature_columns][label_columns] for i in range(0, len(y_row), len_feature_columns)]).flatten()

    # 결과 출력
    print(result)

def main():
    config = Config()

    data_processor = DoubleAxisDataProcessor(config)
    np_stock_data = data_processor.get_np_data()
    dataset = TimeSeriesDataset(config, np_stock_data)
    train_loader = DataLoader(dataset=dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)


    model = CrossAttnModel(config.input_size, config.output_size, config.seq_len)

    train(config, model, train_loader)





    print('here')


if __name__ == '__main__':
    test()
    main()

