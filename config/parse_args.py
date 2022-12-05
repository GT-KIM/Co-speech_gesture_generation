import configargparse


def str2bool(v):
    """ from https://stackoverflow.com/a/43357954/1361529 """
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise configargparse.ArgumentTypeError('Boolean value expected.')


def parse_args(name = 'seq2seq'):
    parser = configargparse.ArgParser()
    if name == 'seq2seq' :
        parser.add('-c', '--config', is_config_file=True, help='Config file path', default='../config/seq2seq.yml')
    elif name == 'proposed' :
        parser.add('-c', '--config', is_config_file=True, help='Config file path', default='../config/proposed.yml')
    else :
        assert ValueError("Invalid name")
    parser.add("--name", type=str, default="main")
    parser.add("--train_data_path", action="append", required=True)
    parser.add("--val_data_path", action="append", required=True)
    parser.add("--test_data_path", action="append", required=False)
    parser.add("--model_save_path", required=True)
    parser.add("--random_seed", type=int, default=-1)

    # word embedding
    parser.add("--wordembed_path", type=str, default=None)
    parser.add("--wordembed_dim", type=int, default=200)

    # model
    parser.add("--model", type=str, required=True)
    parser.add("--epochs", type=int, default=10)
    parser.add("--batch_size", type=int, default=50)
    parser.add("--dropout_prob", type=float, default=0.3)
    parser.add("--text_n_layers", type=int, default=2)
    parser.add("--text_hidden_size", type=int, default=200)

    # dataset
    parser.add("--data_mean", action="append", type=float, nargs='*')
    parser.add("--data_std", action="append", type=float, nargs='*')
    parser.add("--motion_resampling_framerate", type=int, default=24)
    parser.add("--n_poses", type=int, default=50)
    parser.add("--n_pre_poses", type=int, default=5)
    parser.add("--subdivision_stride", type=int, default=5)
    parser.add("--loader_workers", type=int, default=0)

    # training
    parser.add("--learning_rate", type=float, default=0.001)
    parser.add("--loss_l1_weight", type=float, default=50)
    parser.add("--loss_cont_weight", type=float, default=0.1)
    parser.add("--loss_var_weight", type=float, default=0.01)

    if name == 'proposed' :
        parser.add("--text_max_len", type=int, default=32)
        parser.add("--pretrain_epochs", type=int, default=50)
        parser.add("--pretrain_model_save_path", type=str, default='../output/pretrain_proposed')
        parser.add("--feature_dim", type=int, default=1024)
        parser.add("--encoder_n_layers", type=int, default=2)
        parser.add("--decoder_n_layers", type=int, default=8)
        parser.add("--pretrain_masking", type=str2bool, default=False)
        parser.add("--mask_length", type=int, default=5)

    args = parser.parse_args()
    return args

if __name__ == '__main__' :
    args = parse_args()
    print(args)
