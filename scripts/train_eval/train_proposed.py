import logging
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

loss_i = 0
def custom_loss(output, target, args):
    n_element = output.numel()

    # MSE
    l1_loss = F.l1_loss(output, target)
    l1_loss *= args.loss_l1_weight

    # continuous motion [batch, 40, 165]
    diff = [abs(output[:, n, :] - output[:, n-1, :]) for n in range(1, output.shape[1])]
    cont_loss = -torch.sum(torch.stack(diff)) / n_element
    cont_loss *= args.loss_cont_weight

    # motion variance
    norm = torch.norm(output, 2, 1)
    var_loss = -torch.sum(norm) / n_element
    var_loss *= args.loss_var_weight

    loss = l1_loss + cont_loss + var_loss

    # inspect loss terms
    global loss_i
    if loss_i == 100 :
        logging.debug('  (loss terms) l1 %.5f, cont %.5f, var %.5f' % (l1_loss.item(), cont_loss.item(), var_loss.item()))
        loss_i = 0
    loss_i += 1

    return loss


def train_iter_proposed(args, epoch, in_text, in_lengths, in_audio, in_pose, embedder, generator, encoder, optim, lang_model, device):
    # zero gradients
    optim.zero_grad()

    target_text = in_text
    target_audio = in_audio
    target_pose = in_pose

    # generation
    prob = np.random.rand(3)
    in_text = None if prob[0] < 0.1 else text_masking(args, in_text, lang_model)
    in_audio = None if prob[1] < 0.1 else audio_masking(args, in_audio)
    if in_text is None and in_audio is None :
        in_pose = pose_masking(args, in_pose)
    else :
        in_pose = None if prob[2] < 0.1 else pose_masking(args, in_pose)

    feat_text, feat_audio, feat_pose = embedder(in_text, in_audio, in_pose)
    enc_text, enc_audio, enc_pose = encoder(feat_text, feat_audio, feat_pose)
    out_text, out_audio, out_pose = generator(enc_text[..., 2:], enc_audio[..., 2:], enc_pose[..., 2:])

    # loss
    text_loss = 0 if target_text is None else F.cross_entropy(out_text.view(-1, out_text.shape[-1]), target_text.view(-1))
    audio_loss= 0 if target_audio is None else F.l1_loss(out_audio, target_audio)
    pose_loss = 0 if target_pose is None else custom_loss(out_pose, target_pose, args)
    loss = text_loss + audio_loss + 10 * pose_loss
    loss.backward()

    # optimize
    torch.nn.utils.clip_grad_norm_(encoder.parameters(), 5)
    optim.step()

    return {'loss': loss.item(), 'text_loss': text_loss.item(), 'audio_loss': audio_loss.item(), 'pose_loss': pose_loss.item(),}

def finetune_iter_proposed(args, epoch, in_text, in_lengths, in_audio, in_pose, embedder, generator, encoder, decoder, optim, lang_model, device):
    # zero gradients
    optim.zero_grad()

    target_text = in_text.clone()
    target_audio = in_audio.clone()
    target_pose = in_pose.clone()

    remover = np.random.rand()
    if remover < 0.3 :
        in_text = None
    elif remover < 0.6 :
        in_audio = None



    dec_in_pose = in_pose.clone()
    dec_in_pose[:, 10 : , :] = 0.
    _, _, dec_in_pose = embedder(None, None, dec_in_pose)

    # generation
    in_pose[:, 10 : , :] = 0.
    #in_pose = None

    feat_text, feat_audio, feat_pose = embedder(in_text, in_audio, in_pose)
    enc_text, enc_audio, enc_pose = encoder(feat_text, feat_audio, feat_pose)
    teacher_forcing = np.random.rand() < -1

    if teacher_forcing :
        dec_pose1 = decoder(dec_in_pose[...], torch.cat((enc_text, enc_audio, enc_pose), dim=1), True)
        dec_pose2 = decoder(dec_in_pose[...], torch.cat((enc_text, enc_audio, enc_pose), dim=1), False)
        _, _, out_pose2 = generator(None, None, dec_pose2[..., 2:])
    else :
        dec_pose1 = decoder(dec_in_pose[...], torch.cat((enc_text, enc_audio, enc_pose), dim=1), False)
        dec_pose2 = None
    out_text, out_audio, out_pose = generator(enc_text[..., 2:], enc_audio[..., 2:], dec_pose1[..., 2:])

    # loss
    text_loss = F.cross_entropy(out_text.view(-1, out_text.shape[-1]), target_text.view(-1))
    audio_loss= F.l1_loss(out_audio, target_audio)
    pose_loss1 = custom_loss(out_pose[:, :], target_pose[:, 1:], args)
    if teacher_forcing :
        pose_loss2 = custom_loss(out_pose2[:, :], target_pose[:, 1:], args)
        loss = pose_loss1 + pose_loss2 + text_loss + audio_loss #+ 0.001 * pose_loss2
    else :
        pose_loss2 = 0.
        loss = pose_loss1 + text_loss + audio_loss
    loss.backward()

    # optimize
    torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(decoder.parameters()), 5)
    optim.step()

    if teacher_forcing :
        return {'loss': loss.item(), 'text_loss': text_loss.item(), 'audio_loss': audio_loss.item(), 'pose_loss1': pose_loss1.item(), 'pose_loss2' : pose_loss2.item()}
        #return {'loss': loss.item(), 'text_loss': text_loss.item(), 'audio_loss': audio_loss.item(), 'pose_loss1': pose_loss1.item(), 'pose_loss2': 0}
    else :
        return {'loss': loss.item(), 'text_loss': text_loss.item(), 'audio_loss': audio_loss.item(), 'pose_loss1': pose_loss1.item(), 'pose_loss2': 0}
    #return {'loss': loss.item(), 'text_loss': text_loss.item(), 'audio_loss': audio_loss.item(), 'pose_loss': pose_loss.item(),}


def pretrain_iter_proposed(args, epoch, in_text, in_audio, in_pose, embedder, generator, optim):
    # zero gradients
    optim.zero_grad()

    # generation
    text, audio, pose = embedder(in_text, in_audio, in_pose)
    text, audio, pose = text[..., 2:], audio[..., 2:], pose[..., 2:]
    out_text, out_audio, out_pose = generator(text, audio, pose)

    # loss
    for i in range(text.shape[1]) :
        audio_idx = audio.shape[1] // text.shape[1]
        pose_idx = pose.shape[1] // text.shape[1]
        if i == 0 :
            embed_loss = F.l1_loss(text[:,i], torch.mean(audio[:,i*audio_idx:(i+1)*audio_idx], dim=1)) + F.l1_loss(text[:,i], torch.mean(pose[:,i*pose_idx:(i+1)*pose_idx], dim=1))
        else :
            embed_loss += F.l1_loss(text[:,i], torch.mean(audio[:,i*audio_idx:(i+1)*audio_idx], dim=1)) + F.l1_loss(text[:,i], torch.mean(pose[:,i*pose_idx:(i+1)*pose_idx], dim=1))

    text_loss = F.cross_entropy(out_text.view(-1, out_text.shape[-1]), in_text.view(-1))
    audio_loss = F.l1_loss(out_audio, in_audio)
    pose_loss = F.l1_loss(out_pose, in_pose)
    loss = 0.01 * embed_loss + text_loss + audio_loss + pose_loss
    loss.backward()

    # optimize
    #torch.nn.utils.clip_grad_norm_(net.parameters(), 5)
    optim.step()

    return {'loss': loss.item(), 'embed_loss' : embed_loss.item(), 'text_loss': text_loss.item(), 'audio_loss': audio_loss.item(), 'pose_loss': pose_loss.item(),}

def text_masking(args, in_text, lang_model) :
    masked_text = in_text
    random_number = np.random.rand()
    if random_number < 0.8 :
        # masking
        mask_idx = np.random.randint(1, in_text.shape[1]-1)
        masked_text[:, mask_idx] = 0
    elif 0.8 <= random_number < 0.9 :
        # random masking
        mask_idx = np.random.randint(0, in_text.shape[1])
        masked_text[:, mask_idx] = np.random.randint(lang_model.n_words)
    return masked_text

def audio_masking(args, in_audio) :
    masked_audio = in_audio
    mask_length = args.mask_length
    random_number = np.random.rand()
    if random_number < 0.8 :
        # masking
        mask_idx = np.random.randint(0, in_audio.shape[1]-mask_length-1)
        masked_audio[:, mask_idx : mask_idx + mask_length, :] = 0.
    elif 0.8 <= random_number < 0.9 :
        # random masking
        mask_idx = np.random.randint(0, in_audio.shape[1]-mask_length-1)
        masked_audio[:, mask_idx : mask_idx + mask_length, :] = torch.from_numpy(np.random.randn(mask_length, in_audio.shape[-1]))
    return masked_audio


def pose_masking(args, target_poses) :
    masked_poses = target_poses
    mask_length = args.mask_length
    random_number = np.random.rand()
    if random_number < 0.7 :
        masked_poses[:, 10 : , :] = 0.
    elif 0.7 <= random_number < 0.8 :
        # masking
        mask_idx = np.random.randint(0, target_poses.shape[1]-mask_length-1)
        masked_poses[:, mask_idx : mask_idx + mask_length, :] = 0.
    elif 0.8 <= random_number < 0.9 :
        # random masking
        mask_idx = np.random.randint(0, target_poses.shape[1]-mask_length-1)
        masked_poses[:, mask_idx : mask_idx + mask_length, :] = torch.from_numpy(np.random.randn(mask_length, target_poses.shape[-1]))
    return masked_poses
