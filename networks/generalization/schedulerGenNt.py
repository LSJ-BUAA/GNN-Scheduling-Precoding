# coding=utf8
import torch
import torch.nn as nn
import numpy as np
from random import shuffle
import time
import sys
import math
from utils import complex_matmul, ampli2, cal_sumrate, layer_3d_dep, stre_orth,PrecoderGNN,artificial_feature
import h5py

# trains scheduler and precoder for generalizing the number of BS-ANs (NT)


class GNN3D(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNN3D, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.dim = [input_dim] + list(hidden_dim) + [output_dim]
        for i in range(len(self.dim) - 1):
            if i != len(self.dim) - 2:
                self.layers.append(layer_3d_dep(self.dim[i], self.dim[i + 1], transfer_function=nn.ReLU(), is_BN=True, is_transfer=True, is_attention=is_attention))
            else:
                self.layers.append(layer_3d_dep(self.dim[i], self.dim[i + 1], transfer_function=nn.ReLU(), is_BN=False, is_transfer=False, is_attention=is_attention))

    def forward(self, A):
        # A(BS,2,M,K,NR,NT)
        # BATCH_SIZE,_,M,K,NR,NT = A.shape
        #consider the permutations of RBs, user antennas, and BS antennas
        for i in range(len(self.dim) - 1):
            A = self.layers[i](A)

        return A


class Scheduler(nn.Module):
    def __init__(self):
        super(Scheduler, self).__init__()

        self.gnnM = torch.nn.ModuleList()
        for i in range(NRF):
            self.gnnM.append(GNN3D(input_dim=2+2+1, hidden_dim=layerM, output_dim=1))

    def forward(self, A, check_repeated=False):
        #A(BS,2,M,K,NR,NT)
        BATCH_SIZE, _, M, K, NR, NT = A.shape
        H = A.clone()

        stre, orth = artificial_feature(H)

        A =torch.cat([A,w1*stre,w2*orth],1) # w1 and w2 are the weights for balancing different features

        sumb = torch.zeros([BATCH_SIZE, 1, M, K]).to(device) # scheduled result, to be passed sub-scheduler by sub-scheduler
        b = torch.zeros([BATCH_SIZE, NRF, M, K]).to(device) # this is tensor A' in paper

        for i in range(NRF):
            temp = self.gnnM[i](torch.cat([A, sumb.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1,1,NR, NT)], dim=1))
            Y = temp.mean([-1,-2])

            if self.training:
                Y = torch.nn.functional.softmax(Y / tau, dim=-1)
            else: # get onehot result
                index = Y.max(dim=-1, keepdim=True)[1]
                Y = torch.zeros_like(Y).scatter_(-1, index, 1.0)


            b[:, i] = Y.view(BATCH_SIZE,M, K)
            sumb = sumb + Y

        b = b.transpose(-2,-3).unsqueeze(1).unsqueeze(3)
        H = H.transpose(-2,-3)

        Hnew = torch.matmul(b, H).transpose(-2,-3) # channels of scheduled users
        x_pred, y_pred, z_pred = precoder(Hnew, equal_user=False, equal_RB=False) # well-trained precoder GNN

        if check_repeated:
            # to varify that no user is scheduled twice
            repeat_number = torch.mean(torch.sum(nn.ReLU()(sumb - 1), -1))  # count the number of users that are selected repeatedly, averaged over RBs and batchsize
            # it will be very close to zero after several epochs

            return Hnew, x_pred, y_pred, z_pred, repeat_number
        else:
            return Hnew, x_pred, y_pred, z_pred


class Top_k(torch.nn.Module):
    def __init__(self, k):
        super(Top_k, self).__init__()
        self.k = k #the number of users to be selected
        EPSILON = np.finfo(np.float32).tiny
        self.eps = torch.tensor([EPSILON]).to(device)

    def forward(self, scores, hard=False):
        # m = torch.distributions.gumbel.Gumbel(torch.zeros_like(scores), torch.ones_like(scores)) # do not use Gumbel
        # g = m.sample()
        # scores = scores + g

        khot = torch.zeros_like(scores)
        onehot_approx = torch.zeros_like(scores)

        khot_M = torch.zeros_like(scores).unsqueeze(-2).repeat([1,1,1,self.k,1])

        for i in range(self.k):
            khot_mask = torch.max(1.0 - onehot_approx, self.eps)
            scores = scores + torch.log(khot_mask)
            onehot_approx = torch.nn.functional.softmax(scores / tau, dim=-1)
            khot = khot + onehot_approx

            khot_M[..., i, :] = onehot_approx

        if hard: # used in test, so does not need detach the gradient
            khot_hard = torch.zeros_like(khot)
            _, ind = torch.topk(khot, self.k, dim=-1)
            khot_hard = khot_hard.scatter_(-1, ind, 1)
            res = khot_hard# - khot.detach() + khot # do not use STE

            khot_M_hard = torch.zeros_like(khot_M)
            _, ind = torch.topk(khot_M, 1, dim=-1)
            khot_M_hard = khot_M_hard.scatter_(-1, ind, 1)
            res_M = khot_M_hard# - khot_M.detach() + khot_M

        else:
            res = khot
            res_M = khot_M


        return res, res_M


class NonSeqScheduler(nn.Module):
    def __init__(self):
        super(NonSeqScheduler, self).__init__()

        self.gnnM = GNN3D(input_dim=2+2, hidden_dim=layerM, output_dim=1)
        self.select = Top_k(NRF)

    def forward(self, A, check_repeated=False):
        #A(BS,2,M,K,NR,NT)
        BATCH_SIZE, _, M, K, NR, NT = A.shape
        H = A.clone()

        stre, orth = artificial_feature(H)

        A =torch.cat([A,w1*stre,w2*orth],1) # w1 and w2 are the weights for balancing different features, whose variances are different

        # sumb = torch.zeros([BATCH_SIZE, 1, M, K]).to(device) # scheduled result, to be passed sub-scheduler by sub-scheduler
        # b = torch.zeros([BATCH_SIZE, NRF, M, K]).to(device) # this is tensor A' in paper


        temp = self.gnnM(A)
        Y = temp.mean([-1,-2])

        _,b = self.select(Y, not self.training)
        sumb = torch.sum(b, dim=1, keepdim=True)

        b = b.unsqueeze(3)
        H = H.transpose(-2,-3)

        Hnew = torch.matmul(b, H).transpose(-2,-3) # channels of scheduled users
        x_pred, y_pred, z_pred = precoder(Hnew, equal_user=False, equal_RB=False) # well-trained precoder GNN

        if check_repeated:
            # to varify that no user is scheduled twice

            repeat_number = torch.mean(torch.sum(nn.ReLU()(sumb - 1), -1))  # count the number of users that are selected repeatedly, averaged over RBs and batchsize
            # it will be very close to zero after several epochs

            return Hnew, x_pred, y_pred, z_pred, repeat_number
        else:
            return Hnew, x_pred, y_pred, z_pred



if __name__ == '__main__':
    time_start = time.time()
    M = 4
    K = Kin = 20
    NRF = 4
    NT = 16
    NR = 2

    M2 = M
    Kin2 = Kin
    NTtest = [8,16,32,64,96,128]
    NR2 = NR

    train_number = 200000
    test_number = 2000

    # read training set
    Kread = 6000000
    RBread = 16
    NTread = NT
    NRread = NR
    dict_data = h5py.File('../data/ChannelUMaNLOSRB' + str(RBread) + 'NT' + str(NTread) + 'NR' + str(NRread) + 'K' + str(Kread) + '.mat')
    data = dict_data['CHRB']  # number,NR,NT,M

    data = data[0:train_number  * Kin, :, :, 0:M].reshape([train_number , Kin, NR, NT, M])
    data = data.transpose(0, 4, 1, 2, 3)
    dataTrain = np.stack([data['real'], data['imag']], 1)
    del data
    dict_data.close()

    # read test set
    Kread = 40000
    RBread = 16
    NRread = NR2
    dataTest = []
    for n in range(len(NTtest)):
        NTread = NTtest[n]
        dict_data = h5py.File('../data/ChannelUMaNLOSRB' + str(RBread) + 'NT' + str(NTread) + 'NR' + str(NRread) + 'K' + str(Kread) + '.mat')
        data = dict_data['CHRB']  # number,NR,NT,M2

        data = data[0:test_number * Kin2, :, :, 0:RBread].reshape([test_number, Kin2, NR2, NTread, RBread])
        data = data.transpose(0, 4, 1, 2, 3)
        dataTest.append(np.stack([data['real'], data['imag']], 1))
        del data
        dict_data.close()



    ## SNR
    power_dBm = 46  # BSpower
    noiseFigure = 7
    noise_dBm_per_Hz = -174  # noise density
    Pn = noise_dBm_per_Hz + 10 * np.log10(400 * 1e6) + noiseFigure
    PnoiseM = Pn - 10 * np.log10(264) + 10 * np.log10(M)
    SNR = power_dBm - PnoiseM
    SNR = SNR - 140  # offset

    # make data set
    Xtrain = torch.from_numpy(np.float32(dataTrain[0:train_number,:,0:M]))
    Xtest = []
    for n in range(len(NTtest)):
        Xtest.append(torch.from_numpy(np.float32(dataTest[n][0:test_number,:,0:M2])))

    # hyper parameters
    BATCH_SIZE = 50

    LEARNING_RATE = 3e-4
    LR2 = 3e-4 # or 1e-4 3e-5

    layerM = [32]*2

    tau0 = 0.5
    tauinf = 0.1
    # tau0 = 0.05 # or 0.1 if NRF is large
    # tauinf = 0.02

    MAX_EPOCH = 300
    is_attention = False
    test_equal = False
    is_sequential = True

    device = torch.device("cuda:0")
    # device = torch.device("cpu")

    epoch_train_precoder = 10 # 20 if NRF is large
    w1 = 8
    w2 = 4

    if is_sequential:
        model = Scheduler()
    else:
        layerM = [64] * 4
        tau0 = 0.1
        tauinf = 0.1
        model = NonSeqScheduler()
    model.to(device)

    print('M',M,'K',K,'NRF',NRF,'NT',NT,'NR',NR,'SNR',SNR,'train_number',train_number,'test_number',test_number,'tau',tau0,tauinf)
    print('Sequential',is_sequential,'BatchSize',BATCH_SIZE,'ini_LEARNING_RATE',LEARNING_RATE,LR2,'layerM',layerM,'atten',is_attention, 'epoch_train_precoder',epoch_train_precoder, 'feature_weight',w1,w2,'NTtest',NTtest)
    print("weights", sum(x.numel() for x in model.parameters()))

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [10,30, 70, 110], gamma=0.3, last_epoch=-1)

    # load precoder
    layer = [128] * 10
    precoder = PrecoderGNN(input_dim=2, hidden_dim=layer, output_dim=2 + 4 * NRF, is_attention=True, NRF=NRF)
    precoder.to(device)
    precoder.load_state_dict(torch.load('model_M'+str(M)+'_NRF'+str(NRF)+'_K'+str(K)+'_Kp'+str(NRF)+'_NT'+str(NT)+'_NR'+str(NR)+'_P'+str(power_dBm)+'_E'+str(90)+'_128_10_200k.ckp')['model_state_dict'])
    for p in precoder.parameters():
        p.requires_grad = True
    optimizer1 = torch.optim.Adam(precoder.parameters(), lr=LR2, weight_decay=0)
    scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer1, [30-epoch_train_precoder, 70-epoch_train_precoder, 110-epoch_train_precoder], gamma=0.3, last_epoch=-1)

    # # load a half-trained scheduler,
    # model.load_state_dict(torch.load('Smodel_M'+str(M)+'_NRF'+str(NRF)+'_K'+str(K)+'_Kp'+str(NRF)+'_NT'+str(NT)+'_NR'+str(NR)+'_P'+str(power_dBm)+'_E'+str(10)+'_32_2_200k1.ckp')['model_state_dict'])
    # for p in model.parameters():
    #     p.requires_grad = True
    # optimizer.load_state_dict(torch.load('Smodel_M'+str(M)+'_NRF'+str(NRF)+'_K'+str(K)+'_Kp'+str(NRF)+'_NT'+str(NT)+'_NR'+str(NR)+'_P'+str(power_dBm)+'_E'+str(10)+'_32_2_200k1.ckp')['optimizer_state_dict'])
    # scheduler.load_state_dict(torch.load('Smodel_M'+str(M)+'_NRF'+str(NRF)+'_K'+str(K)+'_Kp'+str(NRF)+'_NT'+str(NT)+'_NR'+str(NR)+'_P'+str(power_dBm)+'_E'+str(10)+'_32_2_200k1.ckp')['scheduler_state_dict'])
    # precoder.load_state_dict(torch.load('Smodel_M' + str(M) + '_NRF' + str(NRF) + '_K' + str(K) + '_Kp' + str(NRF) + '_NT' + str(NT) + '_NR' + str(NR) + '_P' + str(power_dBm) +'_E'+str(10)+ '_32_2_200k1.ckp')['Pmodel'])
    # for p in precoder.parameters():
    #     p.requires_grad = True
    # optimizer1.load_state_dict(torch.load('Smodel_M' + str(M) + '_NRF' + str(NRF) + '_K' + str(K) + '_Kp' + str(NRF) + '_NT' + str(NT) + '_NR' + str(NR) + '_P' + str(power_dBm) + '_E' + str(10) + '_32_2_200k1.ckp')['Pop'])
    # scheduler1.load_state_dict(torch.load('Smodel_M' + str(M) + '_NRF' + str(NRF) + '_K' + str(K) + '_Kp' + str(NRF) + '_NT' + str(NT) + '_NR' + str(NR) + '_P' + str(power_dBm) + '_E' + str(10) + '_32_2_200k1.ckp')['Psc'])


    sys.stdout.flush()
    for epoch in range(MAX_EPOCH):
        tau = tauinf + (tau0-tauinf) * math.exp(-epoch * 0.02)

        index = [i for i in range(train_number)]
        shuffle(index)
        Xtrain = Xtrain[index]

        model.train()
        for b in range(int(train_number/BATCH_SIZE)):
            batch_x = Xtrain[b*BATCH_SIZE:(b+1)*BATCH_SIZE].to(device)
            h_pred, x_pred, y_pred, z_pred = model(batch_x)
            loss = -cal_sumrate(H=h_pred, WRF=z_pred, WBB=y_pred, VRFH=x_pred, SNR_dB=SNR, device=device)

            optimizer.zero_grad()

            if epoch>epoch_train_precoder:
                optimizer1.zero_grad()

            loss.backward()
            optimizer.step()

            if epoch > epoch_train_precoder:
                optimizer1.step()

        scheduler.step()
        if epoch > epoch_train_precoder:
            scheduler1.step()


        if epoch % 1 == 0:
            model.eval()
            with torch.no_grad():
                sum_train_loss = 0
                sum_test_loss = [0]*len(NTtest)

                stre_train = 0
                orth_train = 0
                num_repeat_train = 0

                for b in range(int(test_number / BATCH_SIZE)):
                    batch_x = Xtrain[b * BATCH_SIZE:(b + 1) * BATCH_SIZE].to(device)
                    h_pred, x_pred, y_pred, z_pred, numRepeat = model(batch_x, check_repeated=True)
                    loss = -cal_sumrate(H=h_pred, WRF=z_pred, WBB=y_pred, VRFH=x_pred, SNR_dB=SNR, device=device)
                    sum_train_loss = sum_train_loss + loss.detach().cpu().numpy()
                    t1,t2 = stre_orth(h_pred)
                    stre_train = stre_train + t1.detach().cpu().numpy()
                    orth_train = orth_train + t2.detach().cpu().numpy()
                    num_repeat_train = num_repeat_train + numRepeat.detach().cpu().numpy()

                    for m in range(len(NTtest)):
                        batch_x = Xtest[m][b * BATCH_SIZE:(b + 1) * BATCH_SIZE].to(device)
                        h_pred, x_pred, y_pred, z_pred = model(batch_x)
                        loss = -cal_sumrate(H=h_pred, WRF=z_pred, WBB=y_pred, VRFH=x_pred, SNR_dB=SNR, device=device)
                        sum_test_loss[m] = sum_test_loss[m] + loss.detach().cpu().numpy()


                time_end = time.time()
                print(epoch, time_end - time_start, -sum_train_loss / (test_number / BATCH_SIZE), end=' ')
                print(stre_train/ (test_number / BATCH_SIZE), orth_train/ (test_number / BATCH_SIZE), num_repeat_train/ (test_number / BATCH_SIZE), end=' ')
                for m in range(len(NTtest)):
                    print(-sum_test_loss[m] / (test_number / BATCH_SIZE), end=' ')
                print()
                sys.stdout.flush()
                time_start = time.time()

        if epoch % 10 == 0 and epoch != 0:
            torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(),'Pmodel':precoder.state_dict(),'Pop':optimizer1.state_dict(),'Psc':scheduler1.state_dict()}, 'Smodel_M'+str(M)+'_NRF'+str(NRF)+'_K'+str(K)+'_Kp'+str(NRF)+'_NT'+str(NT)+'_NR'+str(NR)+'_P'+str(power_dBm)+'_E'+str(epoch)+'_32_2_200k.ckp')
