def train_net(net, optimizer, scheduler, block_type, writer, startstep=0, resume_dir=''):
    
    global_step = startstep
    if global_step != 0:
      checkpoint = torch.load(resume_dir)
      net.load_state_dict(checkpoint)
    best_accuracy = 0


    for epoch in range(args.epoch):
        # Here starts the train loop.
        net.train()
        for batch_idx, (x, y) in enumerate(train_dataloader):

            global_step += 1

            #  Send `x` and `y` to either cpu or gpu using `device` variable. 
            x = x.to(device=device)
            y = y.to(device=device)
            
            # Feed `x` into the network, get an output, and keep it in a variable called `logit`. 
            logit = net(x)

            # Compute accuracy of this batch using `logit`, and keep it in a variable called 'accuracy'.
            accuracy = (logit.argmax(1) == y).float().mean()

            # Compute loss using `logit` and `y`, and keep it in a variable called `loss`.
            loss = nn.CrossEntropyLoss()(logit, y)

            # flush out the previously computed gradient.
            optimizer.zero_grad()

            # backward the computed loss. 
            loss.backward()

            # update the network weights. 
            optimizer.step()

            if global_step % args.log_iter == 0 and writer is not None:
                # Log loss and accuracy values using `writer`. Use `global_step` as a timestamp for the log. 
                writer.add_scalar('train_loss', loss, global_step)
                writer.add_scalar('train_accuracy', accuracy, global_step)

            if global_step % args.ckpt_iter == 0: 
                # Save network weights in the directory specified by `ckpt_dir` directory. 
                # torch.save(net.state_dict(), f'{ckpt_dir}/{global_step}.pt')
                torch.save({
                            'epoch': epoch,
                            'model_state_dict': net.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': loss}, f'{ckpt_dir}/{global_step}.pt')

        # Here starts the test loop.
        net.eval()
        with torch.no_grad():
            test_loss = 0.
            test_accuracy = 0.
            test_num_data = 0.
            for batch_idx, (x, y) in enumerate(test_dataloader):
                # Send `x` and `y` to either cpu or gpu using `device` variable..
                x = x.to(device=device)
                y = y.to(device=device)

                # Feed `x` into the network, get an output, and keep it in a variable called `logit`.
                logit = net(x)

                # Compute loss using `logit` and `y`, and keep it in a variable called `loss`.
                loss = nn.CrossEntropyLoss()(logit, y)

                # Compute accuracy of this batch using `logit`, and keep it in a variable called 'accuracy'.
                accuracy = (logit.argmax(dim=1) == y).float().mean()

                test_loss += loss.item()*x.shape[0]
                test_accuracy += accuracy.item()*x.shape[0]
                test_num_data += x.shape[0]

            test_loss /= test_num_data
            test_accuracy /= test_num_data

            if writer is not None: 
                # Log loss and accuracy values using `writer`. Use `global_step` as a timestamp for the log. 
                writer.add_scalar('test_loss', test_loss, global_step)
                writer.add_scalar('test_accuracy', test_accuracy, global_step)

                # Just for checking progress
                print(f'Test result of epoch {epoch}/{args.epoch} || loss : {test_loss:.3f} acc : {test_accuracy:.3f} ')

                writer.flush()

            # Whenever `test_accuracy` is greater than `best_accuracy`, save network weights with the filename 'best.pt' in the directory specified by `ckpt_dir`.
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                torch.save(net.state_dict(), f'{ckpt_dir}/{block_type}_best.pt')
    
        scheduler.step()
    return best_accuracy
