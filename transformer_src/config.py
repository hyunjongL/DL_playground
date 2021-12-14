"""
Settings and config from CS492 KAIST
"""
# Basic settings
torch.manual_seed(470)
torch.cuda.manual_seed(470)

#!pip install easydict
from easydict import EasyDict as edict

args = edict()
args.batch_size = 32
args.nlayers = 4
args.ninp = 256
args.nhid = 512


args.clip = 1
args.lr_lstm = 0.001
args.dropout = 0.2
args.nhid_attn = 256
args.epochs = 20

##### Transformer
args.nhid_tran = 256
args.nhead = 8
args.nlayers_transformer = 8
args.attn_pdrop = 0.1
args.resid_pdrop = 0.1
args.embd_pdrop = 0.1
args.nff = 4 * args.nhid_tran


args.lr_transformer = 0.0001 #1.0
args.betas = (0.9, 0.98)

args.gpu = True


device = 'cuda:0' if torch.cuda.is_available() and args.gpu else 'cpu'
# Create directory name.
result_dir = Path(root) / 'results'
result_dir.mkdir(parents=True, exist_ok=True)

"""

"""


SRC = Field(tokenize = "spacy",
            tokenizer_language="de_core_news_sm",
            init_token = '<sos>',
            eos_token = '<eos>',
            lower = True)

TRG = Field(tokenize = "spacy",
            tokenizer_language="en_core_web_sm",
            init_token = '<sos>',
            eos_token = '<eos>',
            lower = True)

train_data, valid_data, test_data = Multi30k.splits(exts = ('.de', '.en'),
                                                    fields = (SRC, TRG))
SRC.build_vocab(train_data, min_freq = 2)
TRG.build_vocab(train_data, min_freq = 2)

src_ntoken = len(SRC.vocab.stoi) 
trg_ntoken = len(TRG.vocab.stoi)

train_iter, valid_iter, test_iter = BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size = args.batch_size,
    device = device)

pad_id_trg = TRG.vocab.stoi[TRG.pad_token]
pad_id_src = SRC.vocab.stoi[SRC.pad_token]
pad_id = pad_id_src
eos_id = TRG.vocab.stoi[TRG.eos_token]
criterion = nn.CrossEntropyLoss(ignore_index=pad_id) 


if __name__ == "__main__":
    for batch in train_iter:
        src, trg, length_src = sort_batch(batch.src, batch.trg)
        print(length_src)
        print(src, src.shape)
        print(trg, trg.shape)
        break

    print("##### EXAMPLE #####")
    print("SRC: ", word_ids_to_sentence(src[:, 1:2].long().cpu(), SRC.vocab))
    print("TRG: ", word_ids_to_sentence(trg[:, 1:2].long().cpu(), TRG.vocab))

    print("SRC vocab size", len(SRC.vocab.stoi))
    print("TRG vocab size", len(TRG.vocab.stoi))
    print("Vocab", list(SRC.vocab.stoi.items())[:10])
