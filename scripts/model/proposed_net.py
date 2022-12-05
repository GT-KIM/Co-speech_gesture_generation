import torch
import torch.nn.functional as F
from torch import nn

from scripts.model.Conformer import ConformerBlock, PositionalEncoding, ConformerAttentionBlock, AttentionBlock, SelfAttentionBlock
from scripts.model.seq2seq_net import Generator as RNNGenerator
class PoseEmbedding(nn.Module) :
    def __init__(self, args, pose_dim) :
        super(PoseEmbedding, self).__init__()
        self.embedding = nn.Sequential(nn.Linear(pose_dim, args.feature_dim), nn.GELU(),
                                      nn.Linear(args.feature_dim, args.feature_dim), nn.GELU(),
                                      nn.Linear(args.feature_dim, args.feature_dim))

    def forward(self, x) :
        return self.embedding(x)

class SpeechEmbedding(nn.Module) :
    def __init__(self, args, mel_dim) :
        super(SpeechEmbedding, self).__init__()
        self.embedding = nn.Sequential(nn.Linear(mel_dim, args.feature_dim), nn.GELU(),
                                      nn.Linear(args.feature_dim, args.feature_dim), nn.GELU(),
                                      nn.Linear(args.feature_dim, args.feature_dim))

    def forward(self, x) :
        return self.embedding(x)

class TextEmbedding(nn.Module) :
    def __init__(self, args, word_dim) :
        super(TextEmbedding, self).__init__()
        self.embedding = nn.Sequential(nn.Linear(word_dim, args.feature_dim), nn.GELU(),
                                      nn.Linear(args.feature_dim, args.feature_dim), nn.GELU(),
                                      nn.Linear(args.feature_dim, args.feature_dim))

    def forward(self, x) :
        return self.embedding(x)

class PoseGenerator(nn.Module) :
    def __init__(self, args, pose_dim) :
        super(PoseGenerator, self).__init__()
        self.embedding = nn.Sequential(nn.Linear(args.feature_dim, args.feature_dim), nn.GELU(),
                                      nn.Linear(args.feature_dim, args.feature_dim), nn.GELU(),
                                      nn.Linear(args.feature_dim, pose_dim))

    def forward(self, x) :
        return self.embedding(x)

class SpeechGenerator(nn.Module) :
    def __init__(self, args, mel_dim) :
        super(SpeechGenerator, self).__init__()
        self.embedding = nn.Sequential(nn.Linear(args.feature_dim, args.feature_dim), nn.GELU(),
                                      nn.Linear(args.feature_dim, args.feature_dim), nn.GELU(),
                                      nn.Linear(args.feature_dim, mel_dim))

    def forward(self, x) :
        return self.embedding(x)

class TextGenerator(nn.Module) :
    def __init__(self, args, vocab_dim) :
        super(TextGenerator, self).__init__()
        self.embedding = nn.Sequential(nn.Linear(args.feature_dim, args.feature_dim), nn.GELU(),
                                      nn.Linear(args.feature_dim, args.feature_dim), nn.GELU(),
                                      nn.Linear(args.feature_dim, vocab_dim))

    def forward(self, x) :
        return self.embedding(x)

class Embedder(nn.Module) :
    def __init__(self, args, pose_dim, mel_dim, text_dim, embed_size, pose_len=40, audio_len=45, device='cuda:0', pre_trained_embedding=None) :
        super(Embedder, self).__init__()
        self.device = device

        self.feature_dim = args.feature_dim
        self.text_max_len = args.text_max_len
        self.pose_len = pose_len
        self.audio_len = audio_len
        self.mel_dim = mel_dim
        self.pose_dim = pose_dim


        self.pose_embedding_net = PoseEmbedding(args, pose_dim)
        self.speech_embedding_net = SpeechEmbedding(args, mel_dim)
        self.text_embedding_net = TextEmbedding(args, embed_size)

        if pre_trained_embedding is not None:  # use pre-trained embedding (e.g., word2vec, glove)
            assert pre_trained_embedding.shape[0] == text_dim
            assert pre_trained_embedding.shape[1] == embed_size
            self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(pre_trained_embedding), freeze=False)
        else:
            self.embedding = nn.Embedding(text_dim, embed_size)   #(6884, 300)

    def forward(self, input_texts, input_speechs, input_poses) :
        # input_texts : (B, L) = (B, text_max_len)
        # input_poses : (B, L, P) = (B, 40, 165)

        batch_size = None
        if input_texts is not None:
            batch_size = input_texts.shape[0]
        if input_speechs is not None:
            batch_size = input_speechs.shape[0]
        if input_poses is not None:
            batch_size = input_poses.shape[0]
        if batch_size is None:
            assert ValueError("At least 1 modality must be existed")

        text = torch.zeros((batch_size, self.text_max_len), device=self.device).long() if input_texts is None else input_texts
        text = self.embedding(text)
        speech = torch.zeros((batch_size, self.audio_len, self.mel_dim), device=self.device) if input_speechs is None else input_speechs
        pose = torch.zeros((batch_size, self.pose_len, self.pose_dim), device=self.device) if input_poses is None else input_poses

        text = self.text_embedding_net(text)
        speech = self.speech_embedding_net(speech) # (B, speech_len, 1024)
        pose = self.pose_embedding_net(pose) # (B, 40, 1024)


        text = F.pad(text, (2, 0), value=0.)
        for i in range(text.shape[1]) :
            text[:, i, 0] = 1.
            text[:, i, 1] = i
        speech = F.pad(speech, (2, 0), value=0.)
        for i in range(speech.shape[1]) :
            speech[:, i, 0] = 2.
            speech[:, i, 1] = i
        pose = F.pad(pose, (2, 0), value=0.)
        for i in range(pose.shape[1]) :
            pose[:, i, 0] = 3.
            pose[:, i, 1] = i

        return text, speech, pose

class Generator(nn.Module) :
    def __init__(self, args, pose_dim, mel_dim, text_dim) :
        super(Generator, self).__init__()

        self.pose_generator_net = PoseGenerator(args, pose_dim)
        self.speech_generator_net = SpeechGenerator(args, mel_dim)
        self.text_generator_net = TextGenerator(args, text_dim)

    def forward(self, input_texts, input_speechs, input_poses) :
        # input_texts : (L, B) = (max_len, B)
        # input_poses : (B, L, P) = (B, 40, 165)
        text = None if input_texts is None else self.text_generator_net(input_texts) # (B, max_len, 6844)
        speech = None if input_speechs is None else self.speech_generator_net(input_speechs) # (B, speech_len, 1024)
        pose = None if input_poses is None else self.pose_generator_net(input_poses) # (B, 40, 1024)

        return text, speech, pose



class Encoder(nn.Module) :
    def __init__(self, args, pose_len=40, audio_len=45) :
        super(Encoder, self).__init__()
        self.feature_dim = args.feature_dim
        self.text_max_len = args.text_max_len
        self.pose_len = pose_len
        self.audio_len = audio_len

        num_attention_head = 1
        for i in range(1, 16) :
            if (args.feature_dim + 2) % i == 0 :
                num_attention_head = i

        self.conformer = nn.ModuleList([SelfAttentionBlock(
            encoder_dim=args.feature_dim+2,
            num_attention_heads=num_attention_head,
            feed_forward_expansion_factor=4,
            conv_expansion_factor=2,
            feed_forward_dropout_p=args.dropout_prob,
            attention_dropout_p=args.dropout_prob,
            conv_dropout_p=args.dropout_prob,
            conv_kernel_size=31,
            half_step_residual=True,
        ) for _ in range(args.encoder_n_layers
                         )])

        self.positional_encoding = PositionalEncoding(args.feature_dim+2)

    def forward(self, input_texts, input_audios, input_poses) :
        # input_texts : (B, L, F) = (B, max_len, 1024+1)
        # input_speech : (B, L, F) = (B, 45, 1024+1)
        # input_poses : (B, L, F) = (B, 40, 1024+1)

        inputs = torch.cat((input_texts, input_audios, input_poses), dim=1)
        #inputs = torch.cat((input_texts, input_audios), dim=1)
        batch_size, seq_length, _ = inputs.size()
        pos_embedding = self.positional_encoding(seq_length)
        pos_embedding = pos_embedding.repeat(batch_size, 1, 1)

        features = inputs + pos_embedding

        for layer in self.conformer :
            features = layer(features)

        #output_text = features[:, :self.text_max_len, 2:]
        #output_speech = features[:, self.text_max_len:, 2:]

        output_text = features[:, :self.text_max_len]
        #output_speech = features[:, self.text_max_len:]
        output_speech = features[:, self.text_max_len:-self.pose_len]
        output_pose = features[:, -self.pose_len:]

        return output_text, output_speech, output_pose

class PoseDecoder(nn.Module) :
    def __init__(self, args, pose_dim) :
        super(PoseDecoder, self).__init__()
        num_attention_head = 1
        for i in range(1, 16) :
            if (args.feature_dim+2) % i == 0 :
                num_attention_head = i
        """

        self.conformer1 = nn.ModuleList([SelfAttentionBlock(
            encoder_dim=(args.feature_dim+2),
            num_attention_heads=num_attention_head,
            feed_forward_expansion_factor=4,
            conv_expansion_factor=2,
            feed_forward_dropout_p=args.dropout_prob,
            attention_dropout_p=args.dropout_prob,
            conv_dropout_p=args.dropout_prob,
            conv_kernel_size=31,
            half_step_residual=True,
        ) for _ in range(1)])
        """

        self.conformer2 = AttentionBlock(
            encoder_dim=(args.feature_dim+2),
            num_attention_heads=num_attention_head,
            feed_forward_expansion_factor=4,
            conv_expansion_factor=2,
            feed_forward_dropout_p=args.dropout_prob,
            attention_dropout_p=args.dropout_prob,
            conv_dropout_p=args.dropout_prob,
            conv_kernel_size=31,
            half_step_residual=True,
        )
        """
        self.conformer3 = nn.ModuleList([SelfAttentionBlock(
            encoder_dim=(args.feature_dim+2),
            num_attention_heads=num_attention_head,
            feed_forward_expansion_factor=4,
            conv_expansion_factor=2,
            feed_forward_dropout_p=args.dropout_prob,
            attention_dropout_p=args.dropout_prob,
            conv_dropout_p=args.dropout_prob,
            conv_kernel_size=31,
            half_step_residual=True,
        ) for _ in range(1)])
        """

    def forward(self, x, encoder, teacher_forcing=True) : # (B, 40, feature_dim)
        if teacher_forcing :
            input_x = x[:, 0, :].unsqueeze(1)
            x = x[:, :-1, :]
            #for layer in self.conformer1:
            #    x = layer(x)
            x = self.conformer2(x, encoder)
            #for layer in self.conformer3 :
            #    x = layer(x)
            outputs= x
        else :
            # x = pre-seq
            outputs = list()
            input_x = x[:, :1, :]
            for i in range(1, 40) :
                #self_mask = torch.ones((39, i)).to(x.device)
                #mask = torch.ones((39, i)).to(x.device)
                #self_mask[:i, :] = 0
                #mask[:i, :] = 0
                #for layer in self.conformer1 :
                #    input_x = layer(input_x)
                output_x = self.conformer2(input_x, encoder)
                #for layer in self.conformer3 :
                #    output_x = layer(output_x)
                curr_output = output_x[:, i - 1]
                if i < 10 :
                    input_x = torch.cat([input_x, x[:, i:i+1, :]], dim=1)
                else :
                    input_x = torch.cat([input_x, curr_output.unsqueeze(1)], dim=1)
                outputs.append(curr_output)
            outputs = torch.stack(outputs, dim=1)
        return outputs

class PoseDecoder2(nn.Module) :
    def __init__(self, args, pose_dim, speaker_model=None):
        super(PoseDecoder2, self).__init__()
        self.decoder = RNNGenerator(args, 1026, hidden_size=1026, speaker_model=speaker_model)  # args, 165,

        self.n_pre_poses = args.n_pre_poses
        self.pose_dim = pose_dim

    def forward(self, x, encoder_outputs, teacher_forcing=True):  # ex([[   1, 1331,  861,    2]], 4, (1, 10, 165))
        # reshape to (seq x batch x dim)
        poses = x.transpose(0, 1)  # (40, 1, 165)
        encoder_outputs = encoder_outputs.transpose(0, 1)

        #outputs = torch.zeros(40, poses.size(1), 1026).to(poses.device)
        outputs = list()
        # 40, 64, 165)
        # run words through encoder
        decoder_hidden = torch.zeros(encoder_outputs[-40:-38, :, :].shape).to(poses.device).contiguous() # use last hidden state from encoder

        # run through decoder one time step at a time
        decoder_input = poses[0]  # initial pose from the dataset
        #outputs[0] = decoder_input

        for t in range(1, 40):  # 0번 빼고 39번 반복
            decoder_output, decoder_hidden, _ = self.decoder(None, decoder_input, decoder_hidden, encoder_outputs,
                                                             None)  # None, (1, 165), (2, 1, 200), (n, 1, 200), None
            # (batch, 165) (2, batch, 200)  (batch, 1, T)
            #outputs[t] = decoder_output  # (batch, 165)
            outputs.append(decoder_output)

            if teacher_forcing :
                decoder_input = poses[t]  # next input is current target
            else :
                if t < self.n_pre_poses:
                    decoder_input = poses[t]  # next input is current target
                else:
                    decoder_input = decoder_output  # next input is current prediction
        outputs = torch.stack(outputs, dim=0)

        return outputs.transpose(0, 1)