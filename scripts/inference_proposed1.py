import os
import sys
import torchaudio.transforms

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import argparse
import math
import pickle
import pprint
import time
import numpy as np
import torch
import joblib as jl
import librosa
from pathlib import Path
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as R
from glob import glob
import torch.nn.functional as F
import utils
from utils.data_utils import SubtitleWrapper, normalize_string
from utils.train_utils import set_logger
from data_loader.data_preprocessor_without_audio import DataPreprocessor
from torchaudio import transforms
from librosa.display import specshow
import matplotlib.pyplot as plt
from matplotlib import cm
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

[sys.path.append(i) for i in ['.', '..']]

def generate_gestures(args, embedder, generator, encoder, decoder, lang_model, words, audio_raw, pose_dim, seed_seq=None):
    animation = np.load(glob('../dataset/animation' + '/' + animation_name + '/animation.npz')[0])
    out_list = []
    clip_length = animation['mocap_time_length']  #가장 끝 시간
    print('clip_length', clip_length)
    # pre seq
    pre_seq = torch.zeros((1, args.n_pre_poses, pose_dim)) #n_pre_pose=10,
    print('pose_decoder.dim', pose_dim)
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

        curr_audio = audio_raw[int(start_time*16000) : int(end_time*16000)]
        audio = torch.from_numpy(curr_audio).float()
        mel_transform = transforms.MelSpectrogram(sample_rate=16000, n_fft=2048, win_length=int(16000*0.06), hop_length=int(16000*0.03),
                                                  n_mels=128)
        in_audio = torch.log(mel_transform(audio).T + 1e-10)
        in_audio = in_audio.unsqueeze(0)
        in_audio = in_audio.to(device)

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
            pre_seq = out_poses[:, -args.n_pre_poses:]
        pre_seq = pre_seq.float().to(device)    #처음에는 평균값, 이후에는 이전 포즈에서 10개 가져옴
        print('pre_seq.shape', pre_seq.shape)

        in_text = F.pad(in_text, (0, args.text_max_len - in_text.shape[1]), value=0)
        pre_seq = F.pad(pre_seq, (0, 0, 0, 30), value=0.)
        _, _, dec_pose_feat = embedder(None, None, pre_seq)

        # inference
        words_lengths = torch.LongTensor([in_text.shape[1]]).to(device)
        print('words_lengths', words_lengths)

        in_audio = None

        text_feat, audio_feat, pose_feat = embedder(in_text, in_audio, pre_seq)
        enc_text, enc_audio, enc_pose = encoder(text_feat, audio_feat, pose_feat)
        #np.set_printoptions(threshold=np.inf, linewidth=np.inf)
        #print(enc_text.cpu().numpy())
        #print(enc_audio.cpu().numpy())
        dec_pose = decoder(dec_pose_feat, torch.cat((enc_text, enc_audio, enc_pose), dim=1), False)
        out_text, out_audio, out_poses = generator(enc_text[..., 2:], enc_audio[..., 2:], dec_pose[..., 2:])

        """
        if start_time == 140.0 :
            print(word_seq)
            print(in_text)

            plt.figure()
            specshow(in_audio.squeeze(0).cpu().numpy(), cmap=cm.jet)
            plt.show()

            plt.figure()
            plt.plot(audio)
            plt.show()
        """
        """
        out_text = torch.argmax(out_text[0], dim=1).detach().cpu()
        out_audio = out_audio[0].detach().cpu()

        inv_mel = torchaudio.transforms.InverseMelScale(n_stft=1025, n_mels=128, sample_rate=16000)
        in_stft = inv_mel(in_audio[0].cpu().T)
        out_stft = inv_mel(out_audio.T)

        griffin_lim = torchaudio.transforms.GriffinLim(n_fft=2048, n_iter=32, win_length=int(16000*0.06), hop_length=int(16000*0.03))
        recon_in_audio = griffin_lim(in_stft)
        recon_out_audio = griffin_lim(out_stft)
        """


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


def main(checkpoint_path, wav_path, transcript_path, animation_name):

    args, embedder, generator, encoder, decoder, lang_model, pose_dim = utils.train_utils.load_proposed_checkpoint(
        checkpoint_path, device)
    args.data_mean[0][0], args.data_mean[1][0], args.data_mean[2][0] = 0, 0, 0
    args.data_std[0][0], args.data_std[1][0], args.data_std[2][0] = 0, 0, 0
    pprint.pprint(vars(args))
    print(generator)

    save_path = '../output/proposed_inference'
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

    audio_raw, audio_sr = librosa.load(wav_path, mono=True, sr=16000, res_type='kaiser_fast')

    embedder.eval()
    generator.eval()
    encoder.eval()
    decoder.eval()
    # inference
    with torch.no_grad() :
        out_poses = generate_gestures(args, embedder, generator, encoder, decoder, lang_model, word_list, audio_raw, pose_dim)


    # unnormalize
    mean = np.array(args.data_mean).squeeze()
    std = np.array(args.data_std).squeeze()
    std = np.clip(std, a_min=0.01, a_max=None)
    out_poses = np.multiply(out_poses, std) + mean
    print(out_poses.shape)
    print('../dataset/animation' + '/' + animation_name + '/animation.npz')
    animation = np.load(glob('../dataset/animation' + '/' + animation_name + '/animation.npz')[0])
    npz_dict = {}

    np.savez(save_path + '/' + animation_name[:-2] + '_' + str(epoch) + '.npz', gender=animation['gender'],
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

    for i in range(40) :
        epoch = 220
        if i == 5 :
            continue
        tsv_name = "val_2022_v1_{0:03}".format(i)
        animation_name = tsv_name + "_*"
        ckpt_path = "../output/train_proposed/proposed_checkpoint_{0:03}.bin".format(epoch)

        wav_path = glob('../dataset/wav/' + tsv_name + '.wav')[0]
        print(wav_path)
        transcript_path = glob('../dataset/tsv/' + tsv_name + '.tsv')[0]
        print(transcript_path)

        main(ckpt_path, wav_path, transcript_path, animation_name)

