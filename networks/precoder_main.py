# coding=utf8
import torch
import torch.nn as nn
import numpy as np
from random import shuffle
import time
import sys
import math
from utils import cal_sumrate, PrecoderGNN
import h5py

# this file trains the precoder module for simulating the scenes of different SNR and NRF

# to avoid falling into local minima: If the performance after the first epoch does not achieve half of that of conventional numerical methods, please restart the training

# one can fine-tune a well-trained model of a certain SNR, to obtain the results at different SNR level, instead of training from scratch

# each training epoch may take around ten minutes, depending on specific device

if __name__ == '__main__':
    time_start = time.time()
    NRF = 4

    M = 16
    Kin = 10 # Kin means K
    NT = 16
    NR = 1

    # Mtest = [4,8,16,32,64,128]
    Mtest = [16]
    Kin2 = Kin
    NT2 = NT
    NR2 = NR

    train_number = 200000
    test_number = min(train_number,2000)

    # read training set
    Kread = 4000000
    RBread = 16
    NTread = NT
    NRread = NR
    dict_data = h5py.File('../data/ChannelUMaNLOSRB' + str(RBread) + 'NT' + str(NTread) + 'NR' + str(NRread) + 'K' + str(Kread) + '.mat')
    data = dict_data['CHRB']  # number,NR,NT,M

    data = data[0:train_number * Kin, :, :, 0:M].reshape([train_number, Kin, NR, NT, M])
    data = data.transpose(0, 4, 1, 2, 3)
    dataTrain = np.stack([data['real'], data['imag']], 1)
    del data
    dict_data.close()

    # read test set
    Kread = 40000
    RBread = 16 # >=max(Mtest)
    NTread = NT2
    NRread = NR2
    
    dict_data = h5py.File('../data/ChannelUMaNLOSRB' + str(RBread) + 'NT' + str(NTread) + 'NR' + str(NRread) + 'K' + str(Kread) + '.mat')
    data = dict_data['CHRB']  # number,NR,NT,M2

    data = data[0:test_number * Kin2, :, :, 0:RBread].reshape([test_number, Kin2, NR2, NT2, RBread])
    data = data.transpose(0, 4, 1, 2, 3)
    dataTest = np.stack([data['real'], data['imag']], 1)
    del data
    dict_data.close()

    ## SNR
    power_dBm = 46  # BSpower Ptot
    noiseFigure = 7
    noise_dBm_per_Hz = -174  # noise density
    Pn = noise_dBm_per_Hz + 10 * np.log10(400 * 1e6) + noiseFigure
    PnoiseM = Pn - 10 * np.log10(264) + 10 * np.log10(M)
    SNR = power_dBm - PnoiseM # Ptot/(M*sigma^2)
    SNR = SNR - 140  # offset, since PLdB is reduced by 140 when generating channels

    SNRtest = []
    for m in range(len(Mtest)):
        PnoiseM2 = Pn - 10 * np.log10(264) + 10 * np.log10(Mtest[m])
        SNR2 = power_dBm - PnoiseM2
        SNR2 = SNR2 - 140
        SNRtest.append(SNR2)

    # Strongest schedule
    K = NRF

    power = np.sum(dataTrain ** 2, axis=(1, 4, 5))
    index = np.argsort(power, -1)[..., ::-1]
    index = np.expand_dims(np.expand_dims(np.expand_dims(index, 1), 4), 5)
    dataTrain = np.take_along_axis(dataTrain, index, axis=3)
    dataTrain = dataTrain[:, :, :, 0:K]

    power = np.sum(dataTest ** 2, axis=(1, 4, 5))
    index = np.argsort(power, -1)[..., ::-1]
    index = np.expand_dims(np.expand_dims(np.expand_dims(index, 1), 4), 5)
    dataTest = np.take_along_axis(dataTest, index, axis=3)
    dataTest = dataTest[:, :, :, 0:K]

    # make data set
    Xtrain = torch.from_numpy(np.float32(dataTrain[0:train_number, :, 0:M]))
    Xtest = []
    for m in range(len(Mtest)):
        Xtest.append(torch.from_numpy(np.float32(dataTest[0:test_number, :, 0:Mtest[m]])))

    # hyper parameters
    BATCH_SIZE = 50 # GNN uses 50 is good, but large batchsize (~1000) may help avoid local minima (allocate power to one person only) in FNN, CNN, and GAT
    LEARNING_RATE = 1e-3
    layer = [128] * 6 #or 128*8
    MAX_EPOCH = 300 # 90 is enough for training precoder
    is_attention = True # if False, that is ``SGNN w/o A'' version (when using sequence GNNs in scheduler)
    test_equal = False # always False

    device = torch.device("cuda:0")
    # device = torch.device("cpu")

    print('M', M, 'K', Kin, 'K\'', K, 'NRF', NRF, 'NT', NT, 'NR', NR, 'SNR', SNR, 'Ptot',power_dBm, 'train_number', train_number, 'testnumber', test_number)
    print('BatchSize', BATCH_SIZE, 'ini_LEARNING_RATE', LEARNING_RATE, 'layer', layer, 'atten', is_attention, 'Mtest',Mtest)

    model = PrecoderGNN(input_dim=2, hidden_dim=layer, output_dim=2 + 4 * NRF, is_attention=is_attention, NRF=NRF)
    model.to(device)

    epoch_equal = 10 # allocate equal power to each RB or user, during the first 10 epochs to avoid local minima
    print("weights", sum(x.numel() for x in model.parameters()),epoch_equal)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [30, 70, 110, 150], gamma=0.3, last_epoch=-1)

    # # load a half-trained precoder
    # model.load_state_dict(torch.load('model_M16_NRF4_K10_Kp4_NT16_NR1_P46_128_6_200k.ckp')['model_state_dict'])
    # optimizer.load_state_dict(torch.load('model_M16_NRF4_K10_Kp4_NT16_NR1_P46_128_6_200k.ckp')['optimizer_state_dict'])
    # scheduler.load_state_dict(torch.load('model_M16_NRF4_K10_Kp4_NT16_NR1_P46_128_6_200k.ckp')['scheduler_state_dict'])
    # for p in model.parameters():
    #     p.requires_grad = True
    # epoch_equal = 0
    # print('load model')

    sys.stdout.flush()
    for epoch in range(MAX_EPOCH):

        index = [i for i in range(train_number)] # shuffle training set
        shuffle(index)
        Xtrain = Xtrain[index]

        model.train() #train the model
        for b in range(int(train_number / BATCH_SIZE)):
            batch_x = Xtrain[b * BATCH_SIZE:(b + 1) * BATCH_SIZE].to(device)
            if epoch < epoch_equal:
                x_pred, y_pred, z_pred = model(batch_x, equal_user=True, equal_RB=True) # allocate power evenly for each user and each RB, helping avoid local minima
            else:
                x_pred, y_pred, z_pred = model(batch_x, equal_user=test_equal, equal_RB=test_equal)
            loss = -cal_sumrate(H=batch_x, WRF=z_pred, WBB=y_pred, VRFH=x_pred, SNR_dB=SNR, device=device)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        if epoch % 1 == 0: #test the model
            model.eval()
            with torch.no_grad():
                sum_train_loss = 0
                sum_test_loss = [0]*len(Mtest)
                
                for b in range(int(test_number / BATCH_SIZE)):
                    batch_x = Xtrain[b * BATCH_SIZE:(b + 1) * BATCH_SIZE].to(device)
                    x_pred, y_pred, z_pred = model(batch_x, equal_user=test_equal, equal_RB=test_equal)
                    loss = -cal_sumrate(H=batch_x, WRF=z_pred, WBB=y_pred, VRFH=x_pred, SNR_dB=SNR, device=device)
                    sum_train_loss = sum_train_loss + loss.detach().cpu().numpy()

                    for m in range(len(Mtest)):
                        batch_x = Xtest[m][b * BATCH_SIZE:(b + 1) * BATCH_SIZE].to(device)
                        x_pred, y_pred, z_pred = model(batch_x, equal_user=test_equal, equal_RB=test_equal)
                        loss = -cal_sumrate(H=batch_x, WRF=z_pred, WBB=y_pred, VRFH=x_pred, SNR_dB=SNRtest[m], device=device)
                        sum_test_loss[m] = sum_test_loss[m] + loss.detach().cpu().numpy()

                time_end = time.time()
                print(epoch, time_end - time_start, -sum_train_loss / (test_number / BATCH_SIZE), end=' ')
                for m in range(len(Mtest)):
                    print(-sum_test_loss[m] / (test_number / BATCH_SIZE), end=' ')
                print()
                sys.stdout.flush()
                time_start = time.time()

        if epoch % 10 == 0 and epoch != 0:
            torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict()}, 'model_M'+str(M)+'_NRF'+str(NRF)+'_K'+str(Kin)+'_Kp'+str(K)+'_NT'+str(NT)+'_NR'+str(NR)+'_P'+str(power_dBm)+'_E'+str(epoch)+'_128_6_200k.ckp')


