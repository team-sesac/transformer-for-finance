import torch


def get_device(): return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Config:
    device = get_device()

    # data
    base_dir = '../../data/tf_dataset/'
    file_paths = ['20_SK이노베이션_2010.csv',
                  '51_롯데케미칼_2010.csv',
                  '43_대한항공_2010.csv',
                  '225_아시아나항공_2010.csv']
    # file_paths = ['20_SK이노베이션_2010.csv', '51_롯데케미칼_2010.csv']
    n_files = len(file_paths)

    # use_cols = ['Open', 'High', 'Low', 'Close', 'Volume',
    #             'Change', 'pct_change', 'volume_adi', 'volume_obv', 'volume_cmf']
    use_cols = [
        'Close',
        'trend_sma_fast',
        'trend_ema_fast',
        'volatility_bbh',
        'volatility_bbl',
        'volume_nvi',
        'trend_ichimoku_conv',
        'trend_psar_up',
        'volatility_dch',
        'volatility_kch'
    ]

    len_feature_columns = len(use_cols)
    label_columns = [0]
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
    learning_rate = 0.00001
    epochs = 300 # 300
    random_seed = 42

    # to save model
    model_base_dir = '../../data/model/'
    save_every = 50 # 50
    do_continue_train = False    # 每次训练把上一次的final_state作为下一次的init_state，仅用于RNN类型模型，目前仅支持pytorch
    model_to_load = 'model_state_dict_epoch_3_20231210121644.pt'

    # to save visualization figures
    vis_base_dir = '../../data/visual/'

