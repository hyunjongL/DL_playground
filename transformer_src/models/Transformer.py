import torch

from ..utils import *
from ..config import *

MAX_LEN = 100
class MaskedMultiheadAttention(nn.Module):
    """
    A vanilla multi-head masked attention layer with a projection at the end.
    """
    def __init__(self, mask=False):
        super(MaskedMultiheadAttention, self).__init__()
        assert args.nhid_tran % args.nhead == 0
        # mask : whether to use 
        # key, query, value projections for all heads
        self.key = nn.Linear(args.nhid_tran, args.nhid_tran)
        self.query = nn.Linear(args.nhid_tran, args.nhid_tran)
        self.value = nn.Linear(args.nhid_tran, args.nhid_tran)
        # regularization
        self.attn_drop = nn.Dropout(args.attn_pdrop)
        # output projection
        self.proj = nn.Linear(args.nhid_tran, args.nhid_tran)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        if mask:
            self.register_buffer("mask", torch.tril(torch.ones(MAX_LEN, MAX_LEN)))
        self.nhead = args.nhead
        self.d_k = args.nhid_tran // args.nhead

    def forward(self, q, k, v, mask=None):
        B, T_q, nhid = q.shape
        B, T, nhid = k.shape # and v.shape
        
        Q = self.query(q).view(B, T_q, self.nhead, self.d_k).transpose(1, 2)
        K = self.key(k).view(B, T, self.nhead, self.d_k).transpose(1, 2)
        V = self.value(v).view(B, T, self.nhead, self.d_k).transpose(1, 2)
        
        output = torch.matmul(Q, K.transpose(2, 3))
        output /= math.sqrt(self.d_k)
        
        if mask is None:
            output.masked_fill_(self.mask[:T_q, :T]==0, float('-inf'))
        else:
            mask = mask.unsqueeze(1).unsqueeze(1).repeat(1, self.nhead, T_q, 1).to(device)
            output.masked_fill_(mask==0, float('-inf'))
            
        output = torch.softmax(output, dim=-1)
        output = self.attn_drop(output)
        output = torch.matmul(output, V).transpose(1, 2).reshape(B, T_q, self.nhead*self.d_k) # concat 이렇게 하면 안될거같은데...
        output = self.proj(output)
        return output
        
        
class TransformerEncLayer(nn.Module):
    def __init__(self):
        super(TransformerEncLayer, self).__init__()
        self.ln1 = nn.LayerNorm(args.nhid_tran)
        self.ln2 = nn.LayerNorm(args.nhid_tran)
        self.attn = MaskedMultiheadAttention()
        self.dropout1 = nn.Dropout(args.resid_pdrop)
        self.dropout2 = nn.Dropout(args.resid_pdrop)
        self.ff = nn.Sequential(
            nn.Linear(args.nhid_tran, args.nff),
            nn.ReLU(), 
            nn.Linear(args.nff, args.nhid_tran)
        )

    def forward(self, x, mask=None):
        emb = self.ln1(x)
        output = self.dropout1(self.attn(emb, emb, emb, mask)) + emb
        output = self.ln2(output)
        output = self.dropout2(self.ff(output)) + output
        return output
        
        
class TransformerDecLayer(nn.Module):
    def __init__(self):
        super(TransformerDecLayer, self).__init__()
        self.ln1 = nn.LayerNorm(args.nhid_tran)
        self.ln2 = nn.LayerNorm(args.nhid_tran)
        self.ln3 = nn.LayerNorm(args.nhid_tran)
        self.dropout1 = nn.Dropout(args.resid_pdrop)
        self.dropout2 = nn.Dropout(args.resid_pdrop)
        self.dropout3 = nn.Dropout(args.resid_pdrop)
        self.attn1 = MaskedMultiheadAttention(mask=True) # self-attention 
        self.attn2 = MaskedMultiheadAttention() # tgt to src attention
        self.ff = nn.Sequential(
            nn.Linear(args.nhid_tran, args.nff),
            nn.ReLU(), 
            nn.Linear(args.nff, args.nhid_tran)
        )
        
    def forward(self, x, enc_o, enc_mask=None):
        emb = self.ln1(x)
        output = self.dropout1(self.attn1(emb, emb, emb)) + emb
        output = self.ln2(output)
        output = self.dropout2(self.attn2(output, enc_o, enc_o, enc_mask)) + output
        output = self.ln3(output)
        output = self.dropout3(self.ff(output)) + output
        return output
class PositionalEncoding(nn.Module):
    def __init__(self, max_len=4096):
        super().__init__()
        dim = args.nhid_tran
        pos = np.arange(0, max_len)[:, None]
        i = np.arange(0, dim // 2)
        denom = 10000 ** (2 * i / dim)

        pe = np.zeros([max_len, dim])
        pe[:, 0::2] = np.sin(pos / denom)
        pe[:, 1::2] = np.cos(pos / denom)
        pe = torch.from_numpy(pe).float()

        self.register_buffer('pe', pe)

    def forward(self, x):
        # DO NOT MODIFY
        return x + self.pe[:x.shape[1]]


class TransformerEncoder(nn.Module):

    def __init__(self):
        super(TransformerEncoder, self).__init__()
        # input embedding stem
        self.tok_emb = nn.Embedding(src_ntoken, args.nhid_tran)
        self.pos_enc = PositionalEncoding()
        self.dropout = nn.Dropout(args.embd_pdrop)
        # transformer
        self.transform = nn.ModuleList([TransformerEncLayer() for _ in range(args.nlayers_transformer)])
        # decoder head
        self.ln_f = nn.LayerNorm(args.nhid_tran)
        

    def forward(self, x, mask):
        output = self.dropout(self.pos_enc(self.tok_emb(x)))
        for layer in self.transform:
            output = layer(output, mask)
        output = self.ln_f(output)
        return output
        
class TransformerDecoder(nn.Module):
    def __init__(self):
        super(TransformerDecoder, self).__init__()
        self.tok_emb = nn.Embedding(trg_ntoken, args.nhid_tran)
        self.pos_enc = PositionalEncoding()
        self.dropout = nn.Dropout(args.embd_pdrop)
        self.transform = nn.ModuleList([TransformerDecLayer() for _ in range(args.nlayers_transformer)])
        self.ln_f = nn.LayerNorm(args.nhid_tran)
        self.lin_out = nn.Linear(args.nhid_tran, trg_ntoken)
        self.lin_out.weight = self.tok_emb.weight


    def forward(self, x, enc_o, enc_mask):
        # WRITE YOUR CODE HERE
        logits = self.dropout(self.pos_enc(self.tok_emb(x)))
        
        for layer in self.transform:
            logits = layer(logits, enc_o, enc_mask)
        logits = self.lin_out(self.ln_f(logits))
        
        logits /= args.nhid_tran ** 0.5 # Scaling logits. Do not modify this
        return logits
    
    
class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = TransformerEncoder()
        self.decoder = TransformerDecoder()
        
    def forward(self, x, y, length_x, max_len=None, teacher_forcing=True):
        B, T = x.shape
        if max_len is None:
            T_q = y.shape[1]
            max_len = T_q

        enc_mask = None
        if length_x is not None:
            enc_mask = torch.zeros(B, T)
            for j in range(T):
                if torch.any(length_x > j):
                    enc_mask[length_x > j, j] += 1            

        enc_o = self.encoder(x, enc_mask)
        
        if teacher_forcing or self.training:
            dec_output = self.decoder(y[:,:-1], enc_o, enc_mask)
        else:
            for i in range(max_len - 1):
                if i == 0:
                    dec_input = y[:,:1]
                    dec_output = self.decoder(dec_input, enc_o, enc_mask)
                else:
                    dec_input = torch.concat((dec_input, dec_output[:,-1:].argmax(-1)), dim=1)
                    dec_output = self.decoder(dec_input, enc_o, enc_mask)
        return dec_output