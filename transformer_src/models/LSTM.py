"""
LSTM Seq2Seq for translation.
Implemented using skeleton code provided from CS492 KAIST
"""
import torch

from ..utils import *
from ..config import *

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.linear_input = nn.Linear(input_size, 4 * hidden_size)
        self.linear_hidden = nn.Linear(hidden_size, 4 * hidden_size)

    def forward(self, x, state):
#         (hx, cx) = state
        hx, cx = state
        # cx = state[1]
        t = self.linear_input(x) + self.linear_hidden(hx)
        chunk_forgetgate, chunk_ingate, chunk_cellgate, chunk_outgate = t.chunk(chunks=4, dim=1)
        ox = torch.sigmoid(chunk_outgate)
        
        cy = (cx * torch.sigmoid(chunk_forgetgate)) + torch.sigmoid(chunk_ingate) * torch.tanh(chunk_cellgate)
        hy = ox * torch.tanh(cy)
        
        return hy, (hy, cy)
    
    
class LSTMLayer(nn.Module):
    def __init__(self,*cell_args):
        super(LSTMLayer, self).__init__()
        self.cell = LSTMCell(*cell_args)

    def forward(self, x, state, length_x=None):
        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        inputs = x.unbind(0)
        assert (length_x is None) or torch.all(length_x == length_x.sort(descending=True)[0])
        outputs = [] 
        out_hidden_state = []
        out_cell_state = []
        for i in range(len(inputs)):
            out, state = self.cell(inputs[i] , state)
            outputs += [out] 
            if length_x is not None:
                if torch.any(i+1 == length_x):
                    out_hidden_state = [state[0][i+1==length_x]] + out_hidden_state
                    out_cell_state = [state[1][i+1==length_x]] + out_cell_state
        if length_x is not None:
            state = (torch.cat(out_hidden_state, dim=0), torch.cat(out_cell_state, dim=0))
        return torch.stack(outputs), state 
    

class LSTM(nn.Module):
    def __init__(self, ninp, nhid, num_layers, dropout):
        super(LSTM, self).__init__()
        self.layers = []
        self.dropout = nn.Dropout(dropout)
        for i in range(num_layers):
            if i == 0:
                self.layers.append(LSTMLayer(ninp, nhid))
            else:
                self.layers.append(LSTMLayer(nhid, nhid))
        self.layers = nn.ModuleList(self.layers) 

    def forward(self, x, states, length_x=None):
        # WRITE YOUR CODE HERE
        num_layers = len(self.layers)
        output = x # 20 x 32 x 256
        output_states = []
            
        for i in range(num_layers):
            if i == 0:
                output, output_state = self.layers[0](output, states[0], length_x)
            else:
                output = self.dropout(output)
                output, output_state = self.layers[i](output, states[i], length_x)
            output_states.append(output_state)
        return output, output_states

class LSTMEncoder(nn.Module):
    def __init__(self):
        super(LSTMEncoder, self).__init__()
        ninp = args.ninp
        nhid = args.nhid
        nlayers = args.nlayers
        dropout = args.dropout
        self.embed = nn.Embedding(src_ntoken, ninp, padding_idx=pad_id)
        self.dropout = nn.Dropout(dropout)
        self.lstm = LSTM(ninp, nhid, nlayers, dropout)
        
    def forward(self, x, states, length_x=None):
        embedding = self.embed(x)
        lstm_input = self.dropout(embedding)
        output, context_vectors = self.lstm(lstm_input, states, length_x)
        return output, context_vectors

class LSTMDecoder(nn.Module):
    def __init__(self):
        super(LSTMDecoder, self).__init__()
        self.embed = nn.Embedding(trg_ntoken, args.ninp, padding_idx=pad_id)
        self.lstm = LSTM(args.ninp, args.nhid, args.nlayers, args.dropout)
        self.fc_out = nn.Linear(args.nhid, trg_ntoken)
        self.dropout = nn.Dropout(args.dropout)
        self.fc_out.weight = self.embed.weight
        
    def forward(self, x, states):
        embedding = self.embed(x)
        lstm_input = self.dropout(embedding)
        output, output_states = self.lstm(lstm_input, states)
        output = self.fc_out(output)
        return output, output_states

class LSTMSeq2Seq(nn.Module):
    def __init__(self):
        super(LSTMSeq2Seq, self).__init__()
        self.encoder = LSTMEncoder()
        self.decoder = LSTMDecoder()
    
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
        output, output_states = self.encoder(x, init_states, length)
        outputs = []

        for i in range(trg_len-1):
            if i == 0:
                dec_output, output_states = self.decoder(y[0:1], output_states)
                outputs.append(dec_output)
            elif teacher_forcing:
                dec_output, output_states = self.decoder(y[i:i+1], output_states)
                outputs.append(dec_output)
            else:
                dec_output, output_states = self.decoder(dec_output.argmax(-1), output_states)
                outputs.append(dec_output)
        
        return torch.cat(outputs, dim=0)