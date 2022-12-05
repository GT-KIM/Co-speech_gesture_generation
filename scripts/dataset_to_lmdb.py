import os
import librosa
import lmdb
import pyarrow
import numpy as np

from scripts.utils.data_utils import SubtitleWrapper, normalize_string

def make_lmdb_gesture_dataset(base_path) :
    os.makedirs(os.path.join(base_path, 'lmdb', 'lmdb_train'), exist_ok=True)
    os.makedirs(os.path.join(base_path, 'lmdb', 'lmdb_test'), exist_ok=True)

    gesture_path = os.path.join(base_path, 'animation')
    audio_path = os.path.join(base_path, 'wav')
    text_path = os.path.join(base_path, 'tsv')
    out_path = os.path.join(base_path, 'lmdb')

    map_size = 1024 * 20
    map_size <<= 20
    db = [lmdb.open(os.path.join(out_path, 'lmdb_train'), map_size=map_size),
          lmdb.open(os.path.join(out_path, 'lmdb_test'), map_size=map_size)]

    # delete existing files
    for i in range(2) :
        with db[i].begin(write=True) as txn :
            txn.drop(db[i].open_db())

    all_poses = []
    save_idx = 0
    animation_folders = os.listdir(gesture_path)
    for animation_folder in animation_folders :
        name = animation_folder.split('0000')[0]
        name = name.rstrip('_')
        if name[0:3] == 'trn' :
            dataset_idx = 0
        elif name[0:3] == 'val' :
            dataset_idx = 1
        else :
            dataset_idx = 2
            print("Filename Error")

        # load subtitles
        tsv_path = os.path.join(text_path, name) + '.tsv'
        print(tsv_path)
        if os.path.isfile(tsv_path) :
            subtitle = SubtitleWrapper(tsv_path).get()
        else :
            print('non-file', tsv_path)
            continue

        # load audio
        wav_path = os.path.join(audio_path, '{}.wav'.format(name))
        if os.path.isfile(wav_path) :
            audio_raw, audio_sr = librosa.load(wav_path, mono=True, sr=16000, res_type='kaiser_fast')
        else :
            continue

        clips = [{'vid' : name, 'clips' : []}, # train
                 {'vid' : name, 'clips' : []}] # validation

        word_list = []
        for wi in range(len(subtitle)) : # 하나의 tsv 내의 전체 lines
            word_s = float(subtitle[wi][0])
            word_e = float(subtitle[wi][1])
            word = subtitle[wi][2].strip()

            word_tokens = word.split()
            for t_i, token in enumerate(word_tokens) :
                token = normalize_string(token)
                if len(token) > 0 :
                    new_s_time = word_s + (word_e - word_s) * t_i / len(word_tokens)
                    new_e_time = word_s + (word_e - word_s) * (t_i + 1) / len(word_tokens)
                    word_list.append([token, new_s_time, new_e_time])


        motion = np.load(os.path.join(gesture_path, animation_folder, 'animation.npz'))
        print(os.path.join(gesture_path, animation_folder, 'animation.npz'))
        print(len(word_list))

        poses = motion['poses']
        print(poses.shape)

        # align pervis
        poses[:, :3] = 0.


        clips[dataset_idx]['clips'].append(
            {'words' : word_list,
             'poses' : poses,
             'audio_raw' : audio_raw})

        for i in range(2) :
            with db[i].begin(write=True) as txn :
                if len(clips[i]['clips']) > 0 :
                    k = '{:010}'.format(save_idx).encode('ascii')
                    v = pyarrow.serialize(clips[i]).to_buffer()
                    txn.put(k, v)
        all_poses.append(poses)
        save_idx += 1
    for i in range(2) :
        db[i].sync()
        db[i].close()

    all_poses = np.vstack(all_poses)
    pose_mean = np.mean(all_poses, axis=0, dtype=np.float64)
    pose_std = np.std(all_poses, axis=0, dtype=np.float64)

    print('data_mean/std')
    print('data_mean', str(["{:0.5f}".format(e) for e in pose_mean]).replace("'", ""))
    print('data_std', str(["{:0.5f}".format(e) for e in pose_std]).replace("'", ""))
    if not os.path.isdir('../resource/mean_std_poses/') :
        os.makedirs('../resource/mean_std_poses/')
    with open('../resource/mean_std_poses/mean.txt', 'w') as f :
        f.write(str(["{:0.5f}".format(e) for e in pose_mean]).replace("'", ""))
    with open('../resource/mean_std_poses/std.txt', 'w') as f :
        f.write(str(["{:0.5f}".format(e) for e in pose_std]).replace("'", ""))


if __name__ == "__main__" :
    db_path = '../dataset'
    make_lmdb_gesture_dataset(db_path)