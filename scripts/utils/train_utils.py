import logging
import os
import random
from logging.handlers import RotatingFileHandler

import numpy as np
import time
import math
import torch

from train import init_model
from model.proposed_net import Embedder, Generator, Encoder, PoseDecoder, PoseDecoder2


def set_logger(log_path=None, log_filename='log'):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    handlers = [logging.StreamHandler()]
    if log_path is not None:
        os.makedirs(log_path, exist_ok=True)
        handlers.append(
            RotatingFileHandler(os.path.join(log_path, log_filename), maxBytes=10 * 1024 * 1024, backupCount=5))
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s: %(message)s', handlers=handlers)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def time_since(since):
    now = time.time()
    s = now - since
    return '%s' % as_minutes(s)


def save_checkpoint(state, filename):
    torch.save(state, filename)
    logging.info('Saved the checkpoint')


def load_checkpoint_and_model(checkpoint_path, _device='cpu'):
    print('loading checkpoint {}'.format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, map_location=_device)
    args = checkpoint['args']
    epoch = checkpoint['epoch']
    lang_model = checkpoint['lang_model']
    pose_dim = checkpoint['pose_dim']
    print('epoch {}'.format(epoch))

    generator, loss_fn = init_model(args, lang_model, pose_dim, _device)
    generator.load_state_dict(checkpoint['gen_dict'])

    # set to eval mode
    generator.train(False)

    return args, generator, loss_fn, lang_model, pose_dim

def load_pretrain_checkpoint(checkpoint_path, embedder, generator, _device='cpu'):
    print('loading checkpoint {}'.format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, map_location=_device)
    args = checkpoint['args']
    epoch = checkpoint['epoch']
    lang_model = checkpoint['lang_model']
    pose_dim = checkpoint['pose_dim']
    print('epoch {}'.format(epoch))

    embedder.load_state_dict(checkpoint['emb_dict'])
    generator.load_state_dict(checkpoint['gen_dict'])

    return args, embedder, generator, lang_model, pose_dim

def continue_proposed_checkpoint(checkpoint_path, embedder, generator, encoder, decoder, _device='cpu'):
    print('loading checkpoint {}'.format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, map_location=_device)
    args = checkpoint['args']
    epoch = checkpoint['epoch']
    lang_model = checkpoint['lang_model']
    pose_dim = checkpoint['pose_dim']
    print('epoch {}'.format(epoch))

    embedder.load_state_dict(checkpoint['emb_dict'])
    generator.load_state_dict(checkpoint['gen_dict'])
    encoder.load_state_dict(checkpoint['enc_dict'])
    decoder.load_state_dict(checkpoint['dec_dict'])

    return args, embedder, generator, encoder, decoder, lang_model, pose_dim

def load_proposed_checkpoint(checkpoint_path, _device='cpu'):
    print('loading checkpoint {}'.format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, map_location=_device)
    args = checkpoint['args']
    epoch = checkpoint['epoch']
    lang_model = checkpoint['lang_model']
    pose_dim = checkpoint['pose_dim']
    print('epoch {}'.format(epoch))

    embedder = Embedder(args, pose_dim, 128, lang_model.n_words, args.wordembed_dim, pre_trained_embedding=lang_model.word_embedding_weights).to(_device)
    generator = Generator(args, pose_dim, 128, lang_model.n_words).to(_device)
    encoder = Encoder(args, pose_len=40, audio_len=45).to(_device)
    decoder = PoseDecoder(args, pose_dim).to(_device)

    embedder.load_state_dict(checkpoint['emb_dict'])
    generator.load_state_dict(checkpoint['gen_dict'])
    encoder.load_state_dict(checkpoint['enc_dict'])
    decoder.load_state_dict(checkpoint['dec_dict'])

    return args, embedder, generator, encoder, decoder, lang_model, pose_dim
