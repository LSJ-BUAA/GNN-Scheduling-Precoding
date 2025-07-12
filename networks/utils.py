# coding=utf8
import torch
import torch.nn as nn
import math


def complex_matmul(A,B):
    # Multiplication of two complex matrices
    b1, _, m1, k1, h1, _ = A.size()
    _, _, m2, k2, _, w2 = B.size()
    return torch.stack([torch.matmul(A[:, 0, :, :, :, :],B[:, 0, :, :, :, :])-torch.matmul(A[:, 1, :, :, :, :],B[:, 1, :, :, :, :]),
                      torch.matmul(A[:, 0, :, :, :, :],B[:, 1, :, :, :, :])+torch.matmul(A[:, 1, :, :, :, :],B[:, 0, :, :, :, :])],dim=1).view(b1,2,max(m1,m2),max(k1,k2),h1,w2)


def ampli2(A):
    # Squared Frobenius norm
    return A[:,0,:,:,:,:]**2+A[:,1,:,:,:,:]**2


def cal_sumrate(H, WRF, WBB, VRFH, SNR_dB, device):
    # Calculate the averaged spectral efficiency over a batch
    # SIZE: H(BS,2,M,K,NR,NT), WRF(BS,2,1,1,NT,NRF), WBB(BS,2,M,K,NRF,1), VRFH(BS,2,1,K,1,NR)

    BS,_,M,K,NR,_ = H.shape

    sigma2 = 10.0 ** (-SNR_dB / 10.0) / M
    Heff = complex_matmul(complex_matmul(VRFH,H),WRF)#BS,2,M,K,1,NRF

    Heff = Heff.transpose(-2,-3)
    WBB = WBB.transpose(-1,-3)

    Q = complex_matmul(Heff, WBB)
    Q2 = ampli2(Q)

    D = torch.eye(K).to(device) * Q2
    sumD = torch.sum(D, dim=-1, keepdim=True)
    sumQ = torch.sum(Q2, dim=-1, keepdim=True)
    sinr = sumD/(sumQ-sumD+NR*sigma2)
    rate = torch.sum(torch.log2(1.0 + sinr))
    return rate/BS/M



def stre_orth(H):
    # Calculate the averaged Strength and Orthogonality of scheduled channels
    H = H.contiguous()
    BATCH_SIZE, _, M, K, NR, NT = H.shape
    stre = torch.mean(torch.sqrt(torch.sum(H**2,[-1,-2,-5])))

    H = H.view(BATCH_SIZE, 2, M, K * NR, NT)
    H = H / torch.norm(H, dim=[1, 4], keepdim=True)
    Hp = H.transpose(-1, -2).clone()
    Hp[:, 1] = -Hp[:, 1]#conjugate
    U = torch.sqrt(ampli2(complex_matmul(H.unsqueeze(2), Hp.unsqueeze(2))))
    U = U.view(BATCH_SIZE,1,M,K,NR,K,NR).sum([-1,-3])
    orth = torch.mean(torch.sum(torch.triu(U, 1), dim=[-1, -2]))
    return stre, orth



def artificial_feature(H):
    # Calculate artificial features F_S and F_O
    BATCH_SIZE, _, M, K, NR, NT = H.shape

    stre = torch.sqrt(torch.sum(H ** 2, [-1, -2, -5], keepdim=True).repeat(1, 1, 1, 1, NR, NT))
    stre = (stre - torch.mean(stre, -3, keepdim=True)) / torch.std(stre, -3, keepdim=True)


    Hv = H / torch.norm(H, dim=[1, 5], keepdim=True)
    Hv = Hv.view(BATCH_SIZE, 2, M, K * NR, NT)
    Hp = Hv.transpose(-1, -2).clone()
    Hp[:, 1] = -Hp[:, 1]#conjugate
    U = torch.sqrt(ampli2(complex_matmul(Hv.unsqueeze(2), Hp.unsqueeze(2))))
    orth = torch.mean(U, -1, keepdim=True).view(BATCH_SIZE,1,M,K,NR,1).repeat(1, 1, 1, 1, 1, NT)
    orth = (orth - torch.mean(orth, -3, keepdim=True)) / torch.std(orth, -3, keepdim=True)
    return stre, orth


class layer_3d_dep(nn.Module):
    # one layer of 3D-GNN
    def __init__(self, input_dim, output_dim, transfer_function=nn.ReLU(), is_BN=True, is_transfer=True, is_attention=True):
        super(layer_3d_dep, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.is_BN = is_BN
        self.is_transfer = is_transfer
        self.is_attention = is_attention

        if is_BN:
            self.batch_norms = nn.BatchNorm1d(output_dim)
        if is_transfer:
            self.activation = transfer_function

        ini = torch.sqrt(torch.FloatTensor([3.0/output_dim/input_dim]))
        self.P1 = nn.Parameter(torch.rand([output_dim, input_dim], requires_grad=True) * 2 * ini - ini)
        self.P2 = nn.Parameter(torch.rand([output_dim, input_dim], requires_grad=True) * 2 * ini - ini)
        self.P3 = nn.Parameter(torch.rand([output_dim, input_dim], requires_grad=True) * 2 * ini - ini)
        self.P4 = nn.Parameter(torch.rand([output_dim, input_dim], requires_grad=True) * 2 * ini - ini)
        self.P5 = nn.Parameter(torch.rand([output_dim, input_dim], requires_grad=True) * 2 * ini - ini)

        if is_attention:
            self.Q = nn.Parameter(torch.rand([output_dim, input_dim], requires_grad=True) * 2 * ini - ini) #P6
            self.K = nn.Parameter(torch.rand([output_dim, input_dim], requires_grad=True) * 2 * ini - ini) #P7

    def forward(self, A, aggr_func=torch.mean):
        BATCH_SIZE,_,M,K,NR,NT = A.shape
        A = A.contiguous()

        A1 = torch.matmul(self.P1, A.view([BATCH_SIZE, self.input_dim, -1])).view(BATCH_SIZE, self.output_dim, M, K, NR, NT)
        A2 = torch.matmul(self.P2, aggr_func(A, 2).view(BATCH_SIZE, self.input_dim, -1)).view(BATCH_SIZE, self.output_dim, 1, K, NR, NT)
        A3 = torch.matmul(self.P3, aggr_func(A, 5).view(BATCH_SIZE, self.input_dim, -1)).view(BATCH_SIZE, self.output_dim, M, K, NR, 1)
        A4 = torch.matmul(self.P4, aggr_func(A, 4).view(BATCH_SIZE, self.input_dim, -1)).view(BATCH_SIZE, self.output_dim, M, K, 1, NT)

        if self.is_attention:
            temp = torch.mean(A, 4).view([BATCH_SIZE, self.input_dim, -1])
            q = torch.matmul(self.Q, temp).view(BATCH_SIZE, self.output_dim, M, K, NT)
            k = torch.matmul(self.K, temp).view(BATCH_SIZE, self.output_dim, M, K, NT)
            alpha = nn.Tanh()(torch.matmul(q, k.transpose(-1,-2))/NT)

            v = torch.matmul(self.P5, temp).view(BATCH_SIZE, self.output_dim, M, K,  NT)
            A5 = torch.matmul(alpha, v).view([BATCH_SIZE, self.output_dim, M, K, 1, NT]) / K

            A = A1 + 0.1 * A2 + 0.1 * A3 + 0.5*A4 +2*A5 # coefficients for converging faster
            # A = A1 + 0.1 * A2 + 0.1 * A3 + 0.5 * A4 + A5  #
            # A = A1 + 0.1 * A2 + 0.1 * A3 + 0.1 * A4 + A5  # maybe more effective with large NR
            # when NR=1, alpha = 2*nn.Tanh()(torch.matmul(q, k.transpose(-1,-2)) / NT) and A = A1 + 0.1 * A2 + 0.1 * A3 + 0.5*A4 + A5 may be helpful

        else:
            A5 = torch.matmul(self.P5, aggr_func(A, [3,4]).view(BATCH_SIZE, self.input_dim, -1)).view(BATCH_SIZE, self.output_dim, M, 1, 1, NT)
            # A = A1 + A2 + A3 + A4 + A5
            A = A1 + 0.1*A2 + 0.1*A3 + 0.1*A4 + 0.1*A5

        if self.is_transfer:
            A = self.activation(A)
        if self.is_BN:
            A = self.batch_norms(A.view(BATCH_SIZE, self.output_dim, -1)).view(BATCH_SIZE, self.output_dim, M,K,NR,NT)
        return A


class PrecoderGNN(nn.Module):
    # the precoder module in NGNN and SGNN
    def __init__(self, input_dim, hidden_dim, output_dim, is_attention, NRF):
        super(PrecoderGNN, self).__init__()
        self.NRF = NRF
        self.layers = torch.nn.ModuleList()
        self.dim = [input_dim] + list(hidden_dim) + [output_dim]
        for i in range(len(self.dim) - 1):
            if i != len(self.dim) - 2:
                self.layers.append(layer_3d_dep(self.dim[i], self.dim[i + 1], transfer_function=nn.ReLU(), is_BN=True, is_transfer=True, is_attention=is_attention))
            else:
                self.layers.append(layer_3d_dep(self.dim[i], self.dim[i + 1], transfer_function=nn.ReLU(), is_BN=False, is_transfer=False, is_attention=is_attention))

    def forward(self, A, equal_user, equal_RB):
        # consider the permutations of RBs, user antennas, and BS antennas
        # A(BS,2,M,K,NR,NT) K means K' and Kp
        BATCH_SIZE, _, M, K, NR, NT = A.shape
        NRF = self.NRF

        # update layers
        for i in range(len(self.dim) - 1):
            A = self.layers[i](A)

        # A(BS,2+4NRF,M,K,NR,NT)
        # output layer: y for W_BB, z for W_RF, x for V_RF^H
        # should output WRF(BS,2,1,1,NT,NRF), WBB(BS,2,M,K,NRF,1), VRFH(BS,2,1,K,1,NR)

        # WBB
        y1 = torch.mean(A[:, 2:2 + NRF], dim=[4, 5], keepdim=True).transpose(1, 4)
        y2 = torch.mean(A[:, 2 + NRF:2 + 2 * NRF], dim=[4, 5], keepdim=True).transpose(1, 4)

        y = torch.cat([y1, y2], dim=1)

        # to satisfy VRF constraint
        x1 = torch.mean(A[:, 0], dim=[1, 4], keepdim=True)
        x2 = torch.mean(A[:, 1], dim=[1, 4], keepdim=True)

        mo = torch.sqrt(x1 ** 2 + x2 ** 2)
        x1 = x1 / mo
        x2 = x2 / mo
        x = torch.stack([x1, x2], dim=1).transpose(-1, -2)

        # to satisfy WRF constraint
        z1 = torch.mean(A[:, 2 + 2 * NRF:2 + 3 * NRF], dim=[2, 3, 4]).transpose(-1, -2).view([BATCH_SIZE, 1, 1, 1, NT, NRF])
        z2 = torch.mean(A[:, 2 + 3 * NRF:2 + 4 * NRF], dim=[2, 3, 4]).transpose(-1, -2).view([BATCH_SIZE, 1, 1, 1, NT, NRF])

        mo = torch.sqrt(z1 ** 2 + z2 ** 2)
        z1 = z1 / mo
        z2 = z2 / mo
        z = torch.cat([z1, z2], dim=1)

        # to satisfy power constraints
        W = complex_matmul(z, y) #W(BS,2,M,K,NT,1)
        # ampli2(W) (BS,M,K,NT,1)

        # allocate equal power to each RB or user
        # during the first several epochs to avoid local minima
        if equal_RB:
            if equal_user:
                temp = math.sqrt(M * K) * torch.sqrt(torch.sum(ampli2(W), dim=3)).view(BATCH_SIZE, 1, M, K, 1, 1)
            else:
                temp = math.sqrt(M) * torch.sqrt(torch.sum(ampli2(W), dim=[2, 3])).view(BATCH_SIZE, 1, M, 1, 1, 1)
        else:
            if equal_user:
                temp = math.sqrt(K) * torch.sqrt(torch.sum(ampli2(W), dim=[1, 3])).view(BATCH_SIZE, 1, 1, K, 1, 1)
            else:
                temp = torch.sqrt(torch.sum(ampli2(W), dim=[1, 2, 3])).view(BATCH_SIZE, 1, 1, 1, 1, 1)

        y = y / temp

        return x, y, z


