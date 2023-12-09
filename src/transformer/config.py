class Config:
    # data
    base_dir = '../../data/tf_dataset/'
    file_paths = ['20_SK이노베이션_2010.csv',
                  '51_롯데케미칼_2010.csv',
                  '43_대한항공_2010.csv',
                  '225_아시아나항공_2010.csv']
    # file_paths = [base_dir+i for i in file_paths]

    use_cols = ['Open', 'High', 'Low', 'Close', 'Volume',
               'Change', 'pct_change', 'volume_adi', 'volume_obv', 'volume_cmf']

    len_feature_columns = len(use_cols)
    label_columns = [3]

    shuffle_train_data = True

    train_data_rate = 0.8
    test_data_rate = 0.2

    # train
    predict_day = 1

    input_size = 2000
    output_size = 2000

    dropout_rate = 0.2
    seq_len = 256
    heads = 8


    batch_size = 8
    learning_rate = 0.00001
    epoch = 300
    random_seed = 42

    do_continue_train = False    # 每次训练把上一次的final_state作为下一次的init_state，仅用于RNN类型模型，目前仅支持pytorch