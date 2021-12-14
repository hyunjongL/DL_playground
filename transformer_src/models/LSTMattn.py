"""
LSTM Seq2Seq with Attention for translation.
Implemented using skeleton code provided from CS492 KAIST
"""

import torch

from ..utils import *
from ..config import *
from .LSTM import *

class Attention(nn.Module):
    def __init__(self):
        super().__init__()

        self.nhid_enc = args.nhid
        self.nhid_dec = args.nhid
        self.W1 = nn.Linear(self.nhid_enc, args.nhid_attn)
        self.W2 = nn.Linear(self.nhid_dec, args.nhid_attn)
        self.W3 = nn.Linear(args.nhid_attn, 1)

    def forward(self, x, enc_o, dec_h, length_enc=None):
        # WRITE YOUR CODE HERE
        L, B, _ = enc_o.shape
        t = self.W3(torch.tanh(self.W1(enc_o) + self.W2(dec_h).squeeze())) # 스퀴즈 확인 필요
        
        if length_enc is not None:
            for i in range(L):
                if torch.any(length_enc < i):
                    t[i].masked_fill_((length_enc < i).view(B, 1), float('-inf'))
        
        ct = torch.sum(F.softmax(t, dim=0) * enc_o, dim=0, keepdims=True)
        output = torch.cat((x, ct), dim=-1)
        
        return output

class LSTMAttnDecoder(nn.Module):
    def __init__(self):
        super(LSTMAttnDecoder, self).__init__()
        self.embed = nn.Embedding(trg_ntoken, args.ninp, padding_idx=pad_id)
        self.lstm = LSTM(args.ninp + args.nhid, args.nhid, args.nlayers, args.dropout)
        self.fc_out = nn.Linear(args.nhid, trg_ntoken)
        self.dropout = nn.Dropout(args.dropout)
        self.attn = Attention()
        self.fc_out.weight = self.embed.weight
        
    def forward(self, x, enc_o, states, length_enc=None):
        output = self.embed(x)
        output = self.dropout(output)
        output = self.attn(output, enc_o, states[-1][0], length_enc)
        output, output_states = self.lstm(output, states)
        output = self.fc_out(output)
        return output, output_states

class LSTMAttnSeq2Seq(nn.Module):
    def __init__(self):
        super(LSTMAttnSeq2Seq, self).__init__()
        self.encoder = LSTMEncoder()
        self.decoder = LSTMAttnDecoder()
    
    def _get_init_states(self, x):
        init_states = [
            (torch.zeros((x.size(1), args.nhid)).to(x.device),
            torch.zeros((x.size(1), args.nhid)).to(x.device))
            for _ in range(args.nlayers)
        ]
        return init_states

    def forward(self, x, y, length, max_len=None, teacher_forcing=True):
        if max_len is None:
            trg_len = y.size(0)
        else:
            trg_len = max_len
        
        init_states = self._get_init_states(x)
        enc_output, enc_context_vectors = self.encoder(x, init_states, length)
        output_states = self._get_init_states(x)
        outputs = []
        for i in range(trg_len - 1):
            if i == 0:
                dec_output, output_states = self.decoder(y[0:1], enc_output, output_states, length)
                outputs.append(dec_output)
            elif teacher_forcing:
                dec_output, output_states = self.decoder(y[i:i+1], enc_output, output_states, length)
                outputs.append(dec_output)
            else:
                dec_output, output_states = self.decoder(dec_output.argmax(-1), enc_output, output_states, length)
                outputs.append(dec_output)
        return torch.cat(outputs)
