import torch

from .utils import *
from .config import *
from models.LSTM import *
from models.LSTMattn import *
from models.Transformer import *


def run_experiment(model):
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
  
    optimizer = optim.Adam(model.parameters(), lr=args.lr_lstm if not isinstance(model, Transformer) else args.lr_transformer)
  
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
            factor=0.25, patience=1, threshold=0.0001, threshold_mode='rel', 
            cooldown=0, min_lr=0, eps=1e-08, verbose=False)

    best_val_loss = np.inf
        
    for epoch in tq.tqdm(range(4)): #args.epochs
        run_epoch(epoch, model, optimizer, is_train=True)
        with torch.no_grad():
            val_loss = run_epoch(epoch, model, None, is_train=False)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, 'best')
        save_model(model)
        scheduler.step(val_loss)
    

lstm_model = LSTMSeq2Seq().to(device)
lstm_model.apply(init_weights)
run_experiment(lstm_model)
run_translation(lstm_model, test_iter, max_len=100)
#  0.24306756258010864
print('')

attn_model = LSTMAttnSeq2Seq().to(device)
attn_model.apply(init_weights)
run_experiment(attn_model)
run_translation(attn_model, test_iter, max_len=100)
# 0.31881673121358267
print('')

transformer_model = Transformer().to(device)
run_experiment(transformer_model)
run_translation(transformer_model, test_iter, max_len=100)
# 0.37427328016167555
print('')

