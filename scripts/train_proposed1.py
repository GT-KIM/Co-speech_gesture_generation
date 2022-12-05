import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import datetime
import pprint
import random
import time
import sys
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim

[sys.path.append(i) for i in ['.', '..']]

import utils.train_utils

from model import vocab
from model.seq2seq_net import Seq2SeqNet
from model.proposed_net import Embedder, Generator, Encoder, PoseDecoder, PoseDecoder2
from train_eval.train_proposed import train_iter_proposed, pretrain_iter_proposed, finetune_iter_proposed
from utils.average_meter import AverageMeter
from utils.vocab_utils import build_vocab

from config.parse_args import parse_args

from data_loader.lmdb_data_loader import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def init_model(args, lang_model, pose_dim, _device):
    n_frames = args.n_poses
    generator = Seq2SeqNet(args, pose_dim, n_frames, lang_model.n_words, args.wordembed_dim,
                           lang_model.word_embedding_weights).to(_device)
    # args, 165, 40, 6884, 300, (6884, 300)
    loss_fn = torch.nn.MSELoss()

    return generator, loss_fn

def train_epochs(args, train_data_loader, test_data_loader, lang_model, pose_dim, trial_id=None):
    start = time.time()
    loss_meters = [AverageMeter('loss'), AverageMeter('embed_loss'), AverageMeter('text_loss'), AverageMeter('audio_loss'), AverageMeter('pose_loss1'), AverageMeter('pose_loss2')]

    # interval params
    print_interval = int(len(train_data_loader) / 5)
    save_model_epoch_interval = 10

    # init model
    encoder = Encoder(args, pose_len=40, audio_len=45).to(device)
    embedder = Embedder(args, pose_dim, 128, lang_model.n_words, args.wordembed_dim, pre_trained_embedding=lang_model.word_embedding_weights).to(device)
    generator = Generator(args, pose_dim, 128, lang_model.n_words).to(device)
    decoder = PoseDecoder(args, pose_dim).to(device)

    pretrain_name = '{}/{}_checkpoint_{:03d}.bin'.format(args.pretrain_model_save_path, args.name, 50)
    _, embedder, generator, lang_model, out_dim = utils.train_utils.load_pretrain_checkpoint(
        pretrain_name, embedder, generator, device)

    continue_train = True
    continue_epoch = 100
    if continue_train :
        continue_name = '{}/{}_checkpoint_{:03d}.bin'.format(args.model_save_path, args.name, continue_epoch)
        _, embedder, generator, encoder, decoder, _, _ = utils.train_utils.continue_proposed_checkpoint(continue_name, embedder, generator, encoder, decoder, device)
        print("Continue Train")

    # define optimizers
    #enc_optimizer = optim.Adam(list(embedder.parameters()) + list(encoder.parameters()) + list(generator.parameters()), lr=args.learning_rate, betas=(0.5, 0.999))
    enc_optimizer = optim.Adam(encoder.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
    #dec_optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=args.learning_rate, betas=(0.5, 0.999))
    dec_optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()) + list(embedder.parameters()) + list(generator.parameters()), lr=args.learning_rate, betas=(0.5, 0.999))

    # training
    global_iter = 0
    start_epoch = continue_epoch if continue_train else 1
    for epoch in range(start_epoch, args.epochs+1) :
        # evaluate the test set
        #val_metrics = evaluate_testset(test_data_loader, generator, loss_fn, args)

        # save model
        if epoch % save_model_epoch_interval == 0 and epoch > 0 and global_iter > 0:
            try:  # multi gpu
                enc_state_dict = encoder.module.state_dict()
                emb_state_dict = embedder.module.state_dict()
                gen_state_dict = generator.module.state_dict()
                dec_state_dict = decoder.module.state_dict()
            except AttributeError:  # single gpu
                enc_state_dict = encoder.state_dict()
                emb_state_dict = embedder.state_dict()
                gen_state_dict = generator.state_dict()
                dec_state_dict = decoder.state_dict()

            save_name = '{}/{}_checkpoint_{:03d}.bin'.format(args.model_save_path, args.name, epoch)

            utils.train_utils.save_checkpoint({
                'args': args, 'epoch': epoch, 'lang_model': lang_model,
                'pose_dim': pose_dim, 'enc_dict': enc_state_dict, 'emb_dict': emb_state_dict, 'gen_dict': gen_state_dict,
                'dec_dict' : dec_state_dict
            }, save_name)

        # train iter
        iter_start_time = time.time()
        for iter_idx, data in enumerate(train_data_loader, 0):
            global_iter += 1
            in_text, text_lengths, in_pose, in_audio, aux_info = data

            batch_size = in_pose.size(0)
            #print('target_vec.SHAPE', target_vec.shape)
            in_text = in_text.to(device)
            in_audio = in_audio.to(device)
            in_pose = in_pose.to(device)

            in_text = F.pad(in_text, (0, args.text_max_len-in_text.shape[1]))

            # train
            if epoch < 100 :
                loss = train_iter_proposed(args, epoch, in_text, text_lengths, in_audio, in_pose, embedder, generator, encoder, enc_optimizer, lang_model, device)
            else :
                loss = finetune_iter_proposed(args, epoch, in_text, text_lengths, in_audio, in_pose, embedder, generator, encoder, decoder, dec_optimizer, lang_model, device)


            # loss values
            for loss_meter in loss_meters:
                name = loss_meter.name
                if name in loss:
                    loss_meter.update(loss[name], batch_size)

            # print training status
            if (iter_idx + 1) % print_interval == 0:
                print_summary = 'EP {} ({:3d}) | {:>8s}, {:.0f} samples/s | '.format(
                    epoch, iter_idx + 1, utils.train_utils.time_since(start),
                    batch_size / (time.time() - iter_start_time))
                for loss_meter in loss_meters:
                    if loss_meter.count > 0:
                        print_summary += '{}: {:.3f}, '.format(loss_meter.name, loss_meter.avg)
                        loss_meter.reset()
                logging.info(print_summary)

            iter_start_time = time.time()

def pretrain_epochs(args, train_data_loader, test_data_loader, lang_model, pose_dim, trial_id=None):
    start = time.time()
    loss_meters = [AverageMeter('loss'), AverageMeter('embed_loss'), AverageMeter('text_loss'), AverageMeter('audio_loss'), AverageMeter('pose_loss')]

    # interval params
    print_interval = int(len(train_data_loader) / 5)
    save_model_epoch_interval = 10

    # init model
    embedder = Embedder(args, pose_dim, 128, lang_model.n_words, args.wordembed_dim, pre_trained_embedding=lang_model.word_embedding_weights).to(device)
    generator = Generator(args, pose_dim, 128, lang_model.n_words).to(device)

    pretrain_name = '{}/{}_checkpoint_{:03d}.bin'.format(args.pretrain_model_save_path, args.name, 50)
    args, embedder, generator, lang_model, out_dim = utils.train_utils.load_pretrain_checkpoint(
        pretrain_name, embedder, generator, device)

    # define optimizers
    optimizer = optim.Adam(list(embedder.parameters()) + list(generator.parameters()), lr=args.learning_rate, betas=(0.5, 0.999))

    # training
    global_iter = 0
    for epoch in range(1, args.pretrain_epochs+1):
        # save model
        if epoch % save_model_epoch_interval == 0 and epoch > 0:
            try:  # multi gpu
                emb_state_dict = embedder.module.state_dict()
                gen_state_dict = generator.module.state_dict()
            except AttributeError:  # single gpu
                emb_state_dict = embedder.state_dict()
                gen_state_dict = generator.state_dict()

            save_name = '{}/{}_checkpoint_{:03d}.bin'.format(args.pretrain_model_save_path, args.name, epoch)

            utils.train_utils.save_checkpoint({
                'args': args, 'epoch': epoch, 'lang_model': lang_model,
                'pose_dim': pose_dim, 'emb_dict': emb_state_dict, 'gen_dict' : gen_state_dict
            }, save_name)

        # train iter
        iter_start_time = time.time()
        for iter_idx, data in enumerate(train_data_loader, 0):
            global_iter += 1
            in_text, text_lengths, target_vec, audio, aux_info = data

            batch_size = target_vec.size(0)
            in_text = in_text.to(device)
            audio = audio.to(device)
            target_vec = target_vec.to(device)

            # train
            loss = pretrain_iter_proposed(args, epoch, in_text, audio, target_vec, embedder, generator, optimizer)

            # loss values
            for loss_meter in loss_meters:
                name = loss_meter.name
                if name in loss:
                    loss_meter.update(loss[name], batch_size)

            # print training status
            if (iter_idx + 1) % print_interval == 0:
                print_summary = 'EP {} ({:3d}) | {:>8s}, {:.0f} samples/s | '.format(
                    epoch, iter_idx + 1, utils.train_utils.time_since(start),
                    batch_size / (time.time() - iter_start_time))
                for loss_meter in loss_meters:
                    if loss_meter.count > 0:
                        print_summary += '{}: {:.3f}, '.format(loss_meter.name, loss_meter.avg)
                        loss_meter.reset()
                logging.info(print_summary)

            iter_start_time = time.time()

def evaluate_testset(test_data_loader, generator, loss_fn, args):
    # to evaluation mode
    generator.train(False)

    losses = AverageMeter('loss')
    start = time.time()

    with torch.no_grad():
        for iter_idx, data in enumerate(test_data_loader, 0):
            in_text, text_lengths, target_vec, aux_info = data
            batch_size = target_vec.size(0)

            in_text = in_text.to(device)
            target = target_vec.to(device)

            out_poses = generator(in_text, text_lengths, target, None)
            loss = loss_fn(out_poses, target)
            losses.update(loss.item(), batch_size)

    # back to training mode
    generator.train(True)

    # print
    elapsed_time = time.time() - start
    logging.info('[VAL] loss: {:.3f} / {:.1f}s'.format(losses.avg, elapsed_time))

    return losses.avg


def main(config):
    args = config['args']

    trial_id = None

    # random seed
    if args.random_seed >= 0:
        torch.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)
        np.random.seed(args.random_seed)
        random.seed(args.random_seed)
        os.environ['PYTHONHASHSEED'] = str(args.random_seed)

    # set logger
    utils.train_utils.set_logger(args.model_save_path, os.path.basename(__file__).replace('.py', '.log'))
    utils.train_utils.set_logger(args.pretrain_model_save_path, os.path.basename(__file__).replace('.py', '.log'))

    logging.info("PyTorch version: {}".format(torch.__version__))
    logging.info("CUDA version: {}".format(torch.version.cuda))
    logging.info(pprint.pformat(vars(args)))

    # dataset
    train_dataset = TwhDataset(args.train_data_path[0],
                               n_poses=args.n_poses,
                               subdivision_stride=args.subdivision_stride,
                               pose_resampling_fps=args.motion_resampling_framerate,
                               data_mean=args.data_mean, data_std=args.data_std)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                              shuffle=True, drop_last=True, num_workers=args.loader_workers, pin_memory=True,
                              collate_fn=word_seq_collate_fn
                              )

    val_dataset = TwhDataset(args.val_data_path[0],
                             n_poses=args.n_poses,
                             subdivision_stride=args.subdivision_stride,
                             pose_resampling_fps=args.motion_resampling_framerate,
                             data_mean=args.data_mean, data_std=args.data_std)
    test_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size,
                             shuffle=False, drop_last=True, num_workers=args.loader_workers, pin_memory=True,
                             collate_fn=word_seq_collate_fn
                             )

    # build vocab
    vocab_cache_path = os.path.join(os.path.split(args.train_data_path[0])[0], 'vocab_cache.pkl')
    lang_model = build_vocab('words', [train_dataset, val_dataset], vocab_cache_path, args.wordembed_path,
                             args.wordembed_dim)
    train_dataset.set_lang_model(lang_model)
    val_dataset.set_lang_model(lang_model)
    print(lang_model.n_words)
    print(lang_model.word_embedding_weights.shape)
    # train
    #pretrain_epochs(args, train_loader, test_loader, lang_model,
    #             pose_dim=165, trial_id=trial_id)
    train_epochs(args, train_loader, test_loader, lang_model,
                 pose_dim=165, trial_id=trial_id)


if __name__ == '__main__':
    _args = parse_args(name='proposed')
    main({'args': _args})

