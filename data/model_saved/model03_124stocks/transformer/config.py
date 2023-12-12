import torch
import os


def get_device(): return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_all_csv_files(directory):
    csv_files = [file for file in os.listdir(directory) if file.endswith('.csv')]
    return csv_files


class Config:
    device = get_device()

    # data
    base_dir = './../ch3_ta/final_entry/'
    # file_paths = ['20_SK이노베이션_2010.csv', '51_롯데케미칼_2010.csv']
    file_paths = get_all_csv_files(base_dir)
    n_files = len(file_paths)

    # use_cols = ['Open', 'High', 'Low', 'Close', 'Volume',
    #             'Change', 'pct_change', 'volume_adi', 'volume_obv', 'volume_cmf']
    # use_cols = [
    #     'Close',
    #     'trend_sma_fast',
    #     'trend_ema_fast',
    #     'volatility_bbh',
    #     'volatility_bbl',
    #     'volume_nvi',
    #     'trend_ichimoku_conv',
    #     'trend_psar_up',
    #     'volatility_dch',
    #     'volatility_kch'
    # ]
    # volume, volatility, trend, momentum
    use_cols = ['High', 'Low', 'Close', 'Change', 'volume_obv',
                'volatility_kchi', 'trend_ema_slow']
                # 'volume_em', 'volume_adi',
                # 'volatility_bbp', 'volatility_kchi',
                # 'trend_ema_slow', 'trend_ema_fast', 'trend_ichimoku_base']

    len_feature_columns = len(use_cols)
    label_columns = [2]
    n_labels = len(label_columns)

    # columns_to_scale 생성
    col_list = [i for i in range(len(use_cols))]
    # label_columns에 있는 원소들을 제외
    cols_to_scale = list()
    for x in col_list:
        if x not in label_columns:
            cols_to_scale.append(x)


    shuffle_train_data = False
    test_size = 0.2
    train_size = 0.8

    # train
    predict_day = 1

    input_size = 40
    output_size = 4

    dropout_rate = 0.2
    seq_len = 10
    heads = 8

    batch_size = 8
    learning_rate = 0.000005
    epochs = 2000 # 300
    random_seed = 42

    # to save model
    model_base_dir = '../../data/model/'
    save_every = 500 # 50
    do_continue_train = False    # 每次训练把上一次的final_state作为下一次的init_state，仅用于RNN类型模型，目前仅支持pytorch
    # model_to_load = 'model_state_dict_epoch_3_20231210121644.pt'
    model_to_load = 'model_state_dict_epoch_301_2312122015.pt'
    # to save visualization figures
    vis_base_dir = '../../data/visual/'

