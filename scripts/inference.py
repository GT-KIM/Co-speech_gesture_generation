import argparse
import math
import pickle
import pprint
import time
import os
import numpy as np
import torch
import joblib as jl

from pathlib import Path
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as R

import utils
from utils.data_utils import SubtitleWrapper, normalize_string
from utils.train_utils import set_logger
from data_loader.data_preprocessor_without_audio import DataPreprocessor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def generate_gestures(args, pose_decoder, lang_model, words, seed_seq=None):
    out_list = []
    clip_length = words[-1][2]  #가장 끝 시간
    print('clip_length', clip_length)
    # pre seq
    pre_seq = torch.zeros((1, args.n_pre_poses, pose_decoder.pose_dim)) #n_pre_pose=10,
    print('pose_decoder.dim', pose_decoder.pose_dim)
    print('pre_seq.shape', pre_seq.shape)

    if seed_seq is not None:
        pre_seq[0, :, :] = torch.Tensor(seed_seq[0:args.n_pre_poses])
    else:
        mean_pose = args.data_mean
        mean_pose = torch.squeeze(torch.Tensor(mean_pose))
        pre_seq[0, :, :] = mean_pose.repeat(args.n_pre_poses, 1)
    print('mean_pose.shape', mean_pose.shape)
    print('args.datamean', len(args.data_mean))

    # divide into inference units and do inferences
    # n_pose는 하나의 포즈의 frame 수, motion_resampling_framerate은 샘플링레이트
    unit_time = args.n_poses / args.motion_resampling_framerate #포즈 하나당 시간

    print('unit_time', unit_time)
    stride_time = (args.n_poses - args.n_pre_poses) / args.motion_resampling_framerate
    print('stride_time', stride_time)

    if clip_length < unit_time: #클립의 길이를 넘어가면
        num_subdivision = 1
    else:
        num_subdivision = math.ceil((clip_length - unit_time) / stride_time) + 1
    print('num_subdivision', num_subdivision)

    print('{}, {}, {}, {}'.format(num_subdivision, unit_time, clip_length, stride_time))

    # num_subdivision = min(num_subdivision, 59)  # DEBUG: generate only for the first N divisions

    out_poses = None
    start = time.time()
    for i in range(0, num_subdivision):
        start_time = i * stride_time
        end_time = start_time + unit_time

        # prepare text input
        word_seq = DataPreprocessor.get_words_in_time_range(word_list=words, start_time=start_time, end_time=end_time)
        word_indices = np.zeros(len(word_seq) + 2)
        word_indices[0] = lang_model.SOS_token
        word_indices[-1] = lang_model.EOS_token
        for w_i, word in enumerate(word_seq):
            print(word[0], end=', ')
            word_indices[w_i + 1] = lang_model.get_word_index(word[0])
        print(' ({}, {})'.format(start_time, end_time))
        print('word_indices', word_indices)
        in_text = torch.LongTensor(word_indices).unsqueeze(0).to(device)
        print(in_text)  #(1, 단어의 수)
        # prepare pre seq
        if i > 0:
            pre_seq[0, :, :] = out_poses.squeeze(0)[-args.n_pre_poses:]
        pre_seq = pre_seq.float().to(device)    #처음에는 평균값, 이후에는 이전 포즈에서 10개 가져옴
        print('pre_seq.shape', pre_seq.shape)

        # inference
        words_lengths = torch.LongTensor([in_text.shape[1]]).to(device)
        print('words_lengths', words_lengths)
        out_poses = pose_decoder(in_text, words_lengths, pre_seq, None)
        out_seq = out_poses[0, :, :].data.cpu().numpy()
        print('out_pose', out_poses.shape)
        # smoothing motion transition
        if len(out_list) > 0:
            last_poses = out_list[-1][-args.n_pre_poses:]
            out_list[-1] = out_list[-1][:-args.n_pre_poses]  # delete the last part
            for j in range(len(last_poses)):
                n = len(last_poses)
                prev = last_poses[j]
                next = out_seq[j]
                out_seq[j] = prev * (n - j) / (n + 1) + next * (j + 1) / (n + 1)
        out_list.append(out_seq)

    print('Avg. inference time: {:.2} s'.format((time.time() - start) / num_subdivision))

    # aggregate results
    out_poses = np.vstack(out_list)
    return out_poses


def main(checkpoint_path, transcript_path, animation_name):
    args, generator, loss_fn, lang_model, out_dim = utils.train_utils.load_checkpoint_and_model(
        checkpoint_path, device)
    pprint.pprint(vars(args))
    print(generator)

    save_path = '../output/infer_sample'
    os.makedirs(save_path, exist_ok=True)

    # load lang_model
    vocab_cache_path = os.path.join(os.path.split(args.train_data_path[0])[0], 'vocab_cache.pkl')
    with open(vocab_cache_path, 'rb') as f:
        lang_model = pickle.load(f)

    # prepare input
    transcript = SubtitleWrapper(transcript_path).get()

    word_list = []
    for wi in range(len(transcript)):
        word_s = float(transcript[wi][0])
        word_e = float(transcript[wi][1])
        word = transcript[wi][2].strip()

        word_tokens = word.split()

        for t_i, token in enumerate(word_tokens):
            token = normalize_string(token)
            if len(token) > 0:
                new_s_time = word_s + (word_e - word_s) * t_i / len(word_tokens)
                new_e_time = word_s + (word_e - word_s) * (t_i + 1) / len(word_tokens)
                word_list.append([token, new_s_time, new_e_time])

    # inference
    out_poses = generate_gestures(args, generator, lang_model, word_list)


    # unnormalize
    mean = np.array(args.data_mean).squeeze()
    std = np.array(args.data_std).squeeze()
    std = np.clip(std, a_min=0.01, a_max=None)
    out_poses = np.multiply(out_poses, std) + mean
    print(out_poses.shape)
    print('../dataset/animation' + '/' + animation_name + '/animation.npz')
    animation = np.load('../dataset/animation' + '/' + animation_name + '/animation.npz')
    npz_dict = {}

    np.savez(save_path + '/' + animation_name + '_' + str(epoch) + '.npz', gender=animation['gender'],
             surface_model_type=animation['surface_model_type'],
             mocap_frame_rate=animation['mocap_frame_rate'],
             mocap_time_length=animation['mocap_time_length'],
             poses=out_poses,
             betas=animation['betas'],
             trans=animation['trans'])



    # # make a BVH
    # filename_prefix = '{}'.format(transcript_path.stem)
    # make_bvh(save_path, filename_prefix, out_poses)



if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument("ckpt_path", type=Path, default="../output/train_seq2seq/baseline_icra19_checkpoint_100.bin")
    # parser.add_argument("transcript_path", type=Path, default='../dataset/tsv/0bRocfcPhHU/10/text.tsv')
    # args = parser.parse_args()

    # main(args.ckpt_path, args.transcript_path)
    epoch = 500
    tsv_name = "trn_2022_v1_000"
    animation_name = tsv_name + "_0000-3600"
    ckpt_path = "../output/train_seq2seq/baseline_icra19_checkpoint_" + str(epoch) + ".bin"

    transcript_path = '../dataset/tsv/' + tsv_name + '.tsv'
    print(transcript_path)

    main(ckpt_path, transcript_path, animation_name)

