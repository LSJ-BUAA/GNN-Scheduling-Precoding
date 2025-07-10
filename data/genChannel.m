clear
clc
close all
% output MU-MIMO-OFDM channel tensor: CHRB, whose size is [RB,NT,NR,K*samples]
% some parameters are vaild only for fc=28GHz,BW=400MHz,SCS=120kHz, should be carefully modified

fc = 28e9;                    % carrier frequency in Hz
c = physconst('lightspeed'); % speed of light in m/s
v = 3;                   % UE velocity in km/h
fd = (v*1000/3600)/c*fc;     % UE max Doppler frequency in Hz

RBdemand = 16; %number of RBs enabled

TX = [4 8 1 1 1]; %BS antennas, such as [1 8 1 1 1] [2 8 1 1 1] [4 8 1 1 1] [4 16 1 1 1] [6 16 1 1 1] [8 16 1 1 1]
RX = [1 2 1 1 1]; %UE antennas, such as [1 1 1 1 1] [1 2 1 1 1] [2 2 1 1 1] [2 4 1 1 1] [2 4 1 1 1] [4 4 1 1 1]

NT = prod(TX);
NR = prod(RX);

% Nk = 20*200000; % K*number of samples, for training
% Nk = 20*2000; % K*number of samples, for test
Nk = 100; % for test this file

CHRB = zeros(RBdemand,NT,NR,Nk,'single');

for nk = 1:Nk
    if mod(nk,1000)==0
        disp(nk);
    end
    
    %% Large scale parameter
    % all users are outdoors
    hBS = 25;%BS high in meter
    hUT = unifrnd(1.5,2.5);%UE high in meter
    shadow = 6; %shadow fading in dB 
    distance2d = sqrt(rand*(250^2-35^2)+35^2);%horizontal distance in meter, cell radius 250m, minimum distance 35m
    distance3d = sqrt(distance2d^2 + (hBS-hUT)^2);%3D distance in meter
    
    %% Channel Parameter Configurations
    cdl = nrCDLChannel;  
    cdl.DelayProfile = 'Custom';  
    
    %% Custom Delay Profile: UMa NLOS in TR38.901 Table 7.5-6 Part-1    
    N = 20; %number of clusters
    M = 20; %number of rays in each cluster
    
    r_tau = 2.3; %STEP5 in Sec. 7.5 of TR38.901
    DS = 10.^(-6.28 - 0.204*log10(fc/1e9)+ 0.39*randn());%Table 7.5-6
    tau_np = -r_tau*DS*log(rand(1,N));
    tau_n = sort(tau_np-min(tau_np));
    cdl.PathDelays = tau_n;
    
    zeta = 3;%STEP6
    Zn = randn(1,N)*zeta;
    Pnp = exp(-tau_n*(r_tau-1)/(r_tau*DS)).*10.^(-Zn/10);
    Pn = Pnp/sum(Pnp);
    PndB =  10*log10(Pn/max(Pn));
    cdl.AveragePathGains = PndB;
    
    %STEP 7
    ASA = 10.^(2.08 - 0.27*log10(fc/1e9)+ 0.11*randn()); %Table 7.5-6
    ASD = 10.^(1.5 - 0.1144*log10(fc/1e9)+ 0.28*randn());
    ZSA = 10.^(-0.3236*log10(fc/1e9) + 1.512+ 0.16*randn());
    mulgZSD = -2.1*distance2d/1000-0.01*(hUT-1.5)+0.9;%Table 7.5-7
    ZSD = 10.^(mulgZSD+ 0.49*randn());
    
    ASA= min(ASA, 104);%STEP 4
    ASD = min(ASD, 104);
    ZSA = min(ZSA, 52);
    ZSD = min(ZSD, 52);
    
    AoAp = 2/1.289*ASA/1.4*sqrt(-log(Pn/max(Pn)));%eq7.5-9
    AoA = (round(rand(1,N))*2-1).*AoAp + randn(1,N)*ASA/7;%eq7.5-11
    
    AoDp = 2/1.289*ASD/1.4*sqrt(-log(Pn/max(Pn)));
    AoD = (round(rand(1,N))*2-1).*AoDp + randn(1,N)*ASD/7;
    
    ZoAp = -ZSA/1.178*log(Pn/max(Pn));%eq7.5-14
    ZoA = (round(rand(1,N))*2-1).*ZoAp + randn(1,N)*ZSA/7;%eq7.5-16
    
    ZoDp = -ZSD/1.178*log(Pn/max(Pn));
    ZoDoffset =  7.66*log10(fc/1e9)-5.96-10^((0.208*log10(fc/1e9)- 0.782)*log10(distance2d) -0.13*log10(fc/1e9)+2.03-0.07*(hUT-1.5));
    ZoD =  (round(rand(1,N))*2-1).*ZoDp + randn(1,N)*ZSD/7 + ZoDoffset;%eq7.5-19 
    
    cdl.AnglesAoA = AoA + unifrnd(0, 360);
    cdl.AnglesAoD = AoD + unifrnd(-60, 60); %in a sector
    cdl.AnglesZoA = ZoA + unifrnd(0, 180);
    cdl.AnglesZoD = ZoD + 180 - atand(distance2d/(hBS-hUT));
    
    cdl.AngleSpreads = [2 15 3/8*10^mulgZSD 7]; %Table 7.5-6/7
   
    cdl.XPR = 7+3*randn(); %Table 7.5-6
    % cdl.XPR = 7+3*randn(N,M);% NOTE: new Matlab version can use this one, but it causes errors in R2021a
    
%     cdl.NumStrongestClusters = 2; %STEP11 % NOTE: may should be used, but we use the default
%     cdl.ClusterDelaySpread = (6.5622-3.4084*log10(fc/1e9))/1e9; % NOTE: may should be used, but we use the default

    %% other Parameter Configurations
    cdl.CarrierFrequency = fc;
    cdl.MaximumDopplerShift = fd;

    cdl.TransmitAntennaArray.Size = TX; 
    cdl.ReceiveAntennaArray.Size = RX;
    
    carrierspacing = 120e3; %SCS 120kHz
    BW = 400e6; %BW 400MHz
    Nfft = 2^ceil(log2(BW/carrierspacing));
    SR = carrierspacing * Nfft;
    cdl.SampleRate = SR;
    cdlinfo = cdl.info;
    
    % set the duration of fading channel realizations
    cdl.ChannelFiltering = false;
    Duration = 0.125e-3; % Duration of fading channel realizations, in second. This value is for >= 14 ofdm sysbols
    cdl.NumTimeSamples = SR * Duration;
    
%     cdl.NormalizeChannelOutputs=False; % NOTE: may should be False, but we use the default and normalize it in python, see precoderGenNr.py
    
    % Sample channels: once per OFDM symbol 
    % including CP, an OFDM symbol takes 8.92us, for SCS=120kHz
    cdl.SampleDensity = (1/8.92e-6)/2/cdl.MaximumDopplerShift; % channel sampling times per second, normlized by 2fd as defined by cdl.SampleDensity. 
    
    %% generate channel coefficients
    [pathGains,sampleTimes] = cdl();
    
    subpathDelays = cdlinfo.PathDelays * cdl.SampleRate; % quantize to discrete sample index
    Lpath = ceil(subpathDelays(end))+1;
%     Lpath = ceil(max(subpathDelays))+1; %NOTE: use this when enabling cdl.NumStrongestClusters
    chGains = zeros(size(pathGains,1), Lpath, size(pathGains,3), size(pathGains,4));
    
    % generate time-domain channel for each OFDM symbol
    for i4 = 1:size(pathGains,4)% UE AN
        for i3 = 1:size(pathGains,3)% BS AN
            for i1 = 1:size(pathGains,1)% number of OFDM symbols   
                for i2 = 1:size(pathGains,2)%time 
                    index = ceil(subpathDelays(i2))+1;
                    chGains(i1,index,i3,i4) = chGains(i1,index,i3,i4) + pathGains(i1,i2,i3,i4);
                end
            end
        end
    end
    
    % generate frequency-domain channel, which are averaged for each RB and each time slot
    % an RB includes 12 subcarriers and 14 OFDM symbols
    Nslot = 14;
    for i4 = 1:size(pathGains,4)    % UE AN
        for i3 = 1:size(pathGains,3) % BS AN
            CHtime = zeros(1, Lpath);
            islot = 0;
            for i1 = 1:size(pathGains,1) % number of OFDM symbols            
                CHtime = CHtime + chGains(i1,:,i3,i4); 
                if mod(i1, Nslot) == 0 % per time slot
                    islot = islot + 1;
                    CHtime = CHtime / Nslot; % averge the channels in a time slot
                    CHfreq = fftshift(fft(CHtime, Nfft)); % channels in frequency domain
                    
                    starting = floor((Nfft - 12*264)/2); % starting number of data subcarriers, Mmax=264 RBs  for 400MHz
                    for iRB = 1:RBdemand
%                         CHRB(islot, iRB, i3, i4, nk) = single( mean( CHfreq(starting+(iRB-1)*12 : starting+iRB*12-1) )); % if simulate many time slots
                            % averge the channels over 12 subcarriers for one RB
                            
                            CHRB(iRB, i3, i4, nk) = single( mean( CHfreq(starting+(iRB-1)*12 : starting+iRB*12-1) )); %consider only one time slot
                    end
                    CHtime = zeros(1, Lpath);
                end
            end
        end
    end
    
    %% Large scale Fading
    PLdB = 13.54+39.08*log10(distance3d)+20*log10(fc/1e9)-0.6*(hUT-1.5) + shadow*randn; %NLOS UMa
    PLdB = PLdB - 140; %To improve the precision of data saved, will be offset in calculating SNR
    PL = 10^(-PLdB/10);
    CHRB(:, :, :,  nk) = CHRB(:, :,  :, nk) * sqrt(PL);
    
end

%% save dataset

size(CHRB)
save(['ChannelUMaNLOS' 'RB' num2str(size(CHRB,1)) 'NT' num2str(size(CHRB,2)) 'NR' num2str(size(CHRB,3)) 'K' num2str(Nk) '.mat'], 'CHRB');




