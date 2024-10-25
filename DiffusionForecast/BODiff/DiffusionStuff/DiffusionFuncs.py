def generate(output_directory,
             num_samples,
             ckpt_path,
             data_path,
             ckpt_iter,
             use_model,
             missing_k,
             only_generate_missing,
             iter_num,
             path, 
             forecast_window, 
             forecast_cols, 
             dummy_columns_for_forecast_window,
             gpu_number
             ):
    """
    Generate data based on ground truth

    Parameters:
    output_directory (str):           save generated speeches to this path
    num_samples (int):                number of samples to generate, default is 4
    ckpt_path (str):                  checkpoint path
    ckpt_iter (int or 'max'):         the pretrained checkpoint to be loaded; 
                                      automitically selects the maximum iteration if 'max' is selected
    data_path (str):                  path to dataset, numpy array.
    use_model (int):                  0:DiffWave. 1:SSSDSA. 2:SSSDS4.
    masking (str):                    'mnr': missing not at random, 'bm': black-out, 'rm': random missing
    only_generate_missing (int):      0:all sample diffusion.  1:only apply diffusion to missing portions of the signal
    missing_k (int)                   k missing time points for each channel across the length.
    gpu_number (int)                  GPU number associated with cuda
    """

    # generate experiment (local) path
    local_path = "T{}_beta0{}_betaT{}".format(diffusion_config["T"],
                                              diffusion_config["beta_0"],
                                              diffusion_config["beta_T"])

    # Get shared output_directory ready
    output_directory = os.path.join(output_directory, local_path)
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)
        os.chmod(output_directory, 0o775)
    print("output directory", output_directory, flush=True)

    # map diffusion hyperparameters to gpu
    for key in diffusion_hyperparams:
        if key != "T":
            diffusion_hyperparams[key] = diffusion_hyperparams[key].cuda()

    net = SSSDS4Imputer(**model_config).cuda()

    # load checkpoint
    ckpt_path = os.path.join(ckpt_path, local_path)
    if ckpt_iter == 'max':
        ckpt_iter = find_max_epoch(ckpt_path)
        ckpt_iter = iter_num
    model_path = os.path.join(path, '{}.pkl'.format(ckpt_iter))

    print(model_path)
   
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        net.load_state_dict(checkpoint['model_state_dict'])
        print('Successfully loaded model at iteration {}'.format(iter))
    except:
        raise Exception('No valid model found')

    # Custom data loading and reshaping

    testing_data = np.split(validation_data, 1, 0)


    # testing_data = test_scaled.copy()
    testing_data = np.array(testing_data)
    print(testing_data.shape)
    testing_data = torch.from_numpy(testing_data).float().cuda()
    print(testing_data.shape)
    all_mse = []

    for i, batch in enumerate(testing_data):

        # batch = batch.permute(0, 2, 1)


        batch = batch.to(f'cuda:{gpu_number}').float()
        transposed_mask = get_mask_bm_forecasting(batch[0,:,:], forecast_window, forecast_cols)
        

        for col in dummy_columns_for_forecast_window:
            batch[:, -forecast_window:, col] = 0


        
        
        mask = transposed_mask.permute(1, 0)
        mask = mask.repeat(batch.size()[0], 1, 1).float().cuda()
        
        
        
        loss_mask = ~mask.bool()
        
        batch = batch.permute(0, 2, 1)   
        

        # batch = batch.permute(0, 2, 1)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()


        
        print(mask.size())

        print(batch.shape)
        
        sample_length = batch.size(2)
        sample_channels = batch.size(1)
        generated_audio = sampling(net, (num_samples, sample_channels, sample_length),
                                   diffusion_hyperparams,
                                   cond=batch,
                                   mask=mask,
                                   only_generate_missing=only_generate_missing)

        end.record()
        torch.cuda.synchronize()

        print('generated {} utterances of random_digit at iteration {} in {} seconds'.format(num_samples,
                                                                                             ckpt_iter,
                                                                                             int(start.elapsed_time(
                                                                                                 end) / 1000)))

        generated_audio = generated_audio.detach().cpu().numpy()
        batch = batch.detach().cpu().numpy()
        mask = mask.detach().cpu().numpy()
        loss_mask = loss_mask.detach().cpu().numpy()


        print(r'\n\n\n\n\n\n\n\n\n ADD AN IF STATEMENT HERE TO CHECK IF SAVING THESE OR NOT\n\n\n\n\n\n\n\n\n')


        outfile = f'imputation{i}.npy'
        new_out = os.path.join(path, outfile)
        np.save(new_out, generated_audio)

        outfile = f'original{i}.npy'
        new_out = os.path.join(path, outfile)
        np.save(new_out, batch)

        outfile = f'mask{i}.npy'
        new_out = os.path.join(path, outfile)
        np.save(new_out, mask)


        print('saved generated samples at iteration %s' % ckpt_iter)

        mse = mean_squared_error(generated_audio[loss_mask.astype(bool)], batch[loss_mask.astype(bool)])
            
        all_mse.append(mse)
 

    print('Total MSE:', mean(all_mse))
    return mean(all_mse)

# def calculate_score(gen_config, current_iteration, save_bool):
#     score_pmom = generate(**gen_config,
#         ckpt_iter="max",
#         num_samples=1,
#         use_model=train_config["use_model"],
#         data_path=trainset_config["test_data_path"],
#         missing_k=train_config["missing_k"],
#         only_generate_missing=1, iter_num=current_iteration,forecast_window=168, forecast_cols=[0,1], dummy_columns_for_forecast_window=[2,3,4,5,6,7,8,9,10,11,12,13,14,15],
#         path = r"results/forecastPrevDayDataV2/T200_beta00.0001_betaT0.02/", 
#         save_bool 
#         )
    
    #path to the correct model weights - get path from train #"results/forecastPrevDayDataV2/T200_beta00.0001_betaT0.02/",
    
    return score_pmom


def training(output_directory,
          ckpt_iter,
          n_iters,
          iters_per_ckpt,
          iters_per_logging,
          learning_rate, # <optimise>
          use_model,
          only_generate_missing, # N/A
          missing_k, 
          batch_size, 
          forecast_window, # points to forecast
          forecast_cols, # idx of columns we want to forecast - defauly [0,1] 
          dummy_columns_for_forecast_window, # idx of conditional cols, hidden during forecast - i.e day of week CAN be seen - sets to zero
          gpu_number
          ):

    """
    To optimise: 
    - learning_rate : O(1e-4)
    - "T" : number of diffusion steps O(10)
    - "beta_0" : amount of noise at start of linear scheduel O(0.01)
    - "beta_T" : amount of noise at end of linear scheduel O(0.01)
    """

    # generate experiment (local) path
    local_path = "T{}_beta0{}_betaT{}".format(diffusion_config["T"],
                                              diffusion_config["beta_0"],
                                              diffusion_config["beta_T"]
                                              )
    # Get shared output_directory ready
    output_directory = os.path.join(output_directory, local_path)
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)
        os.chmod(output_directory, 0o775)
    print("output directory", output_directory, flush=True)

    # map diffusion hyperparameters to gpu
    for key in diffusion_hyperparams:
        if key != "T":
            diffusion_hyperparams[key] = diffusion_hyperparams[key].cuda()


    net = SSSDS4Imputer(**model_config).cuda()
    optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate)

    # load checkpoint
    if ckpt_iter == 'max':
        ckpt_iter = find_max_epoch(output_directory)
    if ckpt_iter >= 0:
        try:
            # load checkpoint file
            model_path = os.path.join(output_directory, '{}.pkl'.format(ckpt_iter))
            checkpoint = torch.load(model_path, map_location='cpu')

            # feed model dict and optimizer state
            net.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            print('Successfully loaded model at iteration {}'.format(ckpt_iter))
        except:
            ckpt_iter = -1
            print('No valid checkpoint model found, start training from initialization try.')
    else:
        ckpt_iter = -1
        print('No valid checkpoint model found, start training from initialization.')

    training_data = train_scaled
    
    num_groups = training_data.shape[0] // batch_size
    training_data = np.split(training_data, num_groups, 0)
    
    training_data = np.array(training_data)
    print(training_data.shape)
    training_data = torch.from_numpy(training_data).float().cuda()
    print('Data loaded')

    
    
    # training
    n_iter = ckpt_iter + 1
    while n_iter < n_iters + 1: # TODO: can alter for early stopping
        for batch in training_data: # 40 N, 65 K, 200 L  (batch into the model)
                # batch = torch.from_numpy(batch)
                
                #have one mask for generation and one mask for loss so we optimise on the correct data but dont leak answers to the model
                batch = batch.to(f'cuda:{gpu_number}').float()
                transposed_mask = get_mask_bm_forecasting(batch[0,:,:], forecast_window, forecast_cols)
                
   
                # Replacing conditionals with dummy values to block forecasting on conditionals
                for col in dummy_columns_for_forecast_window:
                    batch[:, -forecast_window:, col] = 0

                # Repeating mask over each sample
                mask = transposed_mask.permute(1, 0)
                mask = mask.repeat(batch.size()[0], 1, 1).float().cuda()
                
                # Invert mask for loss function calculation
                loss_mask = ~mask.bool()
        
                batch = batch.permute(0, 2, 1)   
    
                assert batch.size() == mask.size() == loss_mask.size()

                # back-propagation
                optimizer.zero_grad()
                X = batch, batch, mask, loss_mask
                loss = training_loss_replace(net, nn.MSELoss(), X, diffusion_hyperparams,
                                    only_generate_missing=only_generate_missing)
                
                loss.backward()
                optimizer.step()

                # TODO: add generation check here every nth iteration
                # TODO: pass an extra param to generate to save or not

                if n_iter % iters_per_logging == 0:
                    print("iteration: {} \tloss: {}".format(n_iter, loss.item()))

                # save checkpoint
                if n_iter > 0 and n_iter % iters_per_ckpt == 0:
                    checkpoint_name = '{}.pkl'.format(n_iter)
                    torch.save({'model_state_dict': net.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict()},
                               os.path.join(output_directory, checkpoint_name))
                    print('model at iteration %s is saved' % n_iter)

                n_iter += 1

