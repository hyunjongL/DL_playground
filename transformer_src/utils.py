"""
Util functions provided from CS492 KAIST
"""

def word_ids_to_sentence(id_tensor, vocab, join=' '):
    """Converts a sequence of word ids to a sentence"""
    if isinstance(id_tensor, torch.LongTensor):
        ids = id_tensor.transpose(0, 1).contiguous().view(-1)
    elif isinstance(id_tensor, np.ndarray):
        ids = id_tensor.transpose().reshape(-1)
    batch = [vocab.itos[ind] for ind in ids] # denumericalize
    if join is None:
        return batch
    else:
        return join.join(batch)
    
# Extracts bias and non-bias parameters from a model.
def get_parameters(model, bias=False):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            if bias:
                yield m.bias
            else:
                yield m.weight
        else:
            if not bias:
                yield m.parameters()
                
def run_epoch(epoch, model, optimizer, is_train=True, data_iter=None):
    total_loss = 0
    n_correct = 0
    n_total = 0
    if data_iter is None:
        data_iter = train_iter if is_train else valid_iter
    if is_train:
        model.train()
    else:
        model.eval()
    for batch in data_iter:
        x, y, length = sort_batch(batch.src.to(device), batch.trg.to(device))
        target = y[1:]
        if isinstance(model, Transformer):
            x, y = x.transpose(0, 1), y.transpose(0, 1)
            target = target.transpose(0, 1) #y[:, 1:]
        pred = model(x, y, length)
        loss = criterion(pred.reshape(-1, trg_ntoken), target.reshape(-1))
        n_targets = (target != pad_id).long().sum().item() 
        n_total += n_targets 
        n_correct += (pred.argmax(-1) == target)[target != pad_id].long().sum().item()
        if is_train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
           
            
        total_loss += loss.item() * n_targets
    total_loss /= n_total
    print("Epoch", epoch, 'Train' if is_train else 'Valid', 
          "Loss", np.mean(total_loss), 
          "Acc", n_correct / n_total, 
          "PPL", np.exp(total_loss))
    return total_loss

def word_ids_to_sentence_(ids, vocab):
    sentence = []
    for ind in ids:
        if ind == eos_id:
            break
        sentence.append(vocab.itos[ind])      
    return sentence

def run_translation(model, data_iter, max_len=100, mode='best'):
    with torch.no_grad():
        model.eval()
        load_model(model, mode)
        src_list = []
        gt_list = []
        pred_list = []
        for batch in data_iter:
            x, y, length = sort_batch(batch.src.to(device), batch.trg.to(device))
            target = y[1:]
            if isinstance(model, Transformer):
                x, y = x.transpose(0, 1), y.transpose(0, 1)
                target = target.transpose(0, 1)
            pred = model(x, y, length, max_len=max_len, teacher_forcing=False)
            pred_token = pred.argmax(-1)
            if not isinstance(model, Transformer):
                pred_token = pred_token.transpose(0, 1).cpu().numpy() 
                y = y.transpose(0, 1).cpu().numpy()
                x = x.transpose(0, 1).cpu().numpy()
            # pred_token : batch_size x max_len
            for x_, y_, pred_ in zip(x, y, pred_token):
                src_list.append(word_ids_to_sentence_(x_[1:], SRC.vocab))
                gt_list.append([word_ids_to_sentence_(y_[1:], TRG.vocab)])
                pred_list.append(word_ids_to_sentence_(pred_, TRG.vocab))
        
        for i in range(5):
            print(f"--------- Translation Example {i+1} ---------")
            print("SRC :", ' '.join(src_list[i]))
            print("TRG :", ' '.join(gt_list[i][0]))
            print("PRED:", ' '.join(pred_list[i]))
        print()
        print("BLEU:", bleu_score(pred_list, gt_list))



def save_model(model, mode="last"):
    torch.save(model.state_dict(),  result_dir / f'{type(model).__name__}_{mode}.ckpt')

def load_model(model, mode="last"):
    if os.path.exists(result_dir / f'{type(model).__name__}_{mode}.ckpt'):
        model.load_state_dict(torch.load(result_dir / f'{type(model).__name__}_{mode}.ckpt'))
        print("loaded")
    
def sort_batch(X, y, lengths=None):
    if lengths is None:
        lengths = (X != pad_id_src).long().sum(0)
    lengths, indx = lengths.sort(dim=0, descending=True)
    X = torch.index_select(X, 1, indx) 
    y = torch.index_select(y, 1, indx) 
    return X, y, lengths 

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)