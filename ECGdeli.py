# pyECGdeli - ECG delineation algorithms for python (ECGdeli python port)
# (c) Nicolas Pilia, 2022
# Licensed under GPL 3.0
# Please see the project page for further information: https://github.com/NPilia/pyECGdeli

import scipy as sc
import numpy as np
import pywt


def Check_Small_RR(FPT, fs):
    RR = np.abs(np.diff(FPT[:, 5])) / fs * 1000
    RR250pos = np.where(RR <= 250)[0]
    remove = list()
    FPT_Checked = FPT
    while RR250pos.shape[0] > 0:
        for i in range(0, len(RR250pos)):
            if 3 <= RR250pos[i] <= FPT_Checked.shape[0] - 3:
                Rloc = FPT_Checked[RR250pos[i] - 2:RR250pos[i] + 4:1, 5]
                RR1 = np.diff([Rloc[0:1], Rloc[3:5]])
                RR2 = np.diff([Rloc[0:2], Rloc[4:5]])
                d1 = np.sum(np.abs(np.diff(RR1)))
                d2 = np.sum(np.abs(np.diff(RR2)))
                if d1 > d2:
                    remove.append(RR250pos[i])
                else:
                    remove.append(RR250pos[i] + 1)
            elif RR250pos[i] < 3:
                remove.append(RR250pos[i] + 1)
            elif RR250pos[i] > FPT_Checked.shape[0] - 3:
                remove.append(RR250pos[i])

        FPT_Checked = np.delete(FPT_Checked, np.array(remove), 0)
        remove = list()
        RR = np.abs(np.diff(FPT_Checked[:, 5])) / fs * 1000
        RR250pos = np.where(RR <= 250)[0]

    keep_index = np.where(np.isin(FPT[:, 5], FPT_Checked[:, 5]))[0]
    return FPT_Checked, keep_index


def Sync_R_Peaks(FPT_Cell, fs):
    N_Channels = len(FPT_Cell)

    if N_Channels == 1:
        print('Only one channel present in structure. No synchronization of channels needed.')
        FPT_Synced = FPT_Cell[0]
        FPT_Cell_Synced = FPT_Cell
        return

    M = np.empty((0, 13))
    FPT_Cell_Synced = np.empty([N_Channels], dtype=object)
    for n in range(0, N_Channels):
        FPT_Cell_Synced[n] = np.empty((0, 13))

    Time_Limit = 100
    Voting_Threshhold = 1 / 2


    for n in range(0, N_Channels):
        comp_mat = np.zeros([FPT_Cell[n].shape[0], N_Channels])
        pos_mat = np.zeros([FPT_Cell[n].shape[0], N_Channels], dtype=int)
        samp_point_R_mat = np.zeros([FPT_Cell[n].shape[0], N_Channels])
        for i in range(0, FPT_Cell[n].shape[0]):
            for j in range(0, N_Channels):
                if (FPT_Cell[j].shape[0] == 0):
                    continue
                else:
                    Time_Dif = np.min(1000 / fs * np.abs(FPT_Cell[n][i, 5] - FPT_Cell[j][:, 5]))
                    pos = np.argmin(1000 / fs * np.abs(FPT_Cell[n][i, 5] - FPT_Cell[j][:, 5]))
                    if Time_Dif <= Time_Limit:
                        comp_mat[i, j] = 1
                        pos_mat[i, j] = int(pos)
                        samp_point_R_mat[i, j] = FPT_Cell[j][pos, 5]

            V = np.sum(comp_mat[i, :]) / N_Channels
            if V >= Voting_Threshhold:
                RPOS = np.round(np.median(samp_point_R_mat[i, comp_mat[i, :] == 1]))
                M = np.vstack((M, np.zeros([1, 13])))
                M[-1, 5] = RPOS
                for l in range(0, N_Channels):
                    if comp_mat[i, l]:
                        FPT_Cell_Synced[l] = np.vstack((FPT_Cell_Synced[l], np.zeros([1, 13])))
                        FPT_Cell_Synced[l][-1, 5] = FPT_Cell[l][pos_mat[i, l], 5]
                    else:
                        FPT_Cell_Synced[l] = np.vstack((FPT_Cell_Synced[l], M[-1, :]))
        for j in range(0, N_Channels):
            if FPT_Cell[j].shape[0]>0:
                FPT_Cell[j] = FPT_Cell[j][pos_mat[pos_mat[:, j] <= 0, j], :]

        #print('Elapesed time at iteration end: ' + str(time.time() - t))

    FPT_Synced = np.zeros((M.shape[0], FPT_Cell[0].shape[1]))
    ind = np.argsort(M[:, 5])
    FPT_Synced[:, 5] = M[ind, 5]
    for i in range(0, N_Channels):
        FPT_Cell_Synced[i] = FPT_Cell_Synced[i][ind, :]

    out, ind = np.unique(FPT_Synced[:, 5], return_index=True)
    FPT_Synced = FPT_Synced[ind, :]
    for i in range(0, N_Channels):
        FPT_Cell_Synced[i] = FPT_Cell_Synced[i][ind, :]

    FPT_Synced, keep_index = Check_Small_RR(FPT_Synced, fs)
    for i in range(0, N_Channels):
        FPT_Cell_Synced[i] = FPT_Cell_Synced[i][keep_index, :]

    return FPT_Synced, FPT_Cell_Synced

def Sync_Beats(FPT_Cell, fs):
    N_Channels = len(FPT_Cell)
    if N_Channels == 1:
        print('Only one channel present in structure. No synchronization of channels needed.')
        FPT_Synced = FPT_Cell[0]
        return FPT_Synced, FPT_Cell
    M = np.empty([0, 13])
    FPT_Cell_Synced = np.empty(N_Channels, dtype=object)

    Time_Limit = 100
    Voting_Threshhold = 1 / 2

    for n in range(0, N_Channels):
        FPT_Cell_Synced[n] = np.empty([0, 13])

    for n in range(0, N_Channels):
        comp_mat = np.zeros([FPT_Cell[n].shape[0], N_Channels])
        pos_mat = np.copy(comp_mat).astype(int)
        samp_point_Pon_mat = np.copy(pos_mat)
        samp_point_Ppeak_mat = np.copy(pos_mat)
        samp_point_Poff_mat = np.copy(pos_mat)
        samp_point_QRSon_mat = np.copy(pos_mat)
        samp_point_R_mat = np.copy(pos_mat)
        samp_point_Q_mat = np.copy(pos_mat)
        samp_point_S_mat = np.copy(pos_mat)
        samp_point_QRSoff_mat = np.copy(pos_mat)
        samp_point_J_mat = np.copy(pos_mat)
        samp_point_Ton_mat = np.copy(pos_mat)
        samp_point_Tpeak_mat = np.copy(pos_mat)
        samp_point_Toff_mat = np.copy(pos_mat)

        for i in range(0, FPT_Cell[n].shape[0]):
            for j in range(0, N_Channels):
                if FPT_Cell[j].shape[0] == 0:
                    continue
                else:
                    Timedif = np.min(1000 / fs * abs(FPT_Cell[n][i, 5] - FPT_Cell[j][:, 5]))
                    pos = np.argmin(1000 / fs * abs(FPT_Cell[n][i, 5] - FPT_Cell[j][:, 5]))

                    if Timedif <= Time_Limit:
                        comp_mat[i, j] = 1
                        pos_mat[i, j] = pos
                        samp_point_Pon_mat[i, j] = FPT_Cell[j][pos, 0]
                        samp_point_Ppeak_mat[i, j] = FPT_Cell[j][pos, 1]
                        samp_point_Poff_mat[i, j] = FPT_Cell[j][pos, 2]
                        samp_point_QRSon_mat[i, j] = FPT_Cell[j][pos, 3]
                        samp_point_Q_mat[i, j] = FPT_Cell[j][pos, 4]
                        samp_point_R_mat[i, j] = FPT_Cell[j][pos, 5]
                        samp_point_S_mat[i, j] = FPT_Cell[j][pos, 6]
                        samp_point_QRSoff_mat[i, j] = FPT_Cell[j][pos, 7]
                        samp_point_J_mat[i, j] = FPT_Cell[j][pos, 8]
                        samp_point_Ton_mat[i, j] = FPT_Cell[j][pos, 9]
                        samp_point_Tpeak_mat[i, j] = FPT_Cell[j][pos, 10]
                        samp_point_Toff_mat[i, j] = FPT_Cell[j][pos, 11]

            V = sum(comp_mat[i, :]) / N_Channels

            if V >= Voting_Threshhold:
                PonPOS = np.round(np.median(samp_point_Pon_mat[i, comp_mat[i, :] == 1]))
                PpeakPOS = np.round(np.median(samp_point_Ppeak_mat[i, comp_mat[i, :] == 1]))
                PoffPOS = np.round(np.median(samp_point_Poff_mat[i, comp_mat[i, :] == 1]))
                QRSonPOS = np.round(np.median(samp_point_QRSon_mat[i, comp_mat[i, :] == 1]))
                RPOS = np.round(np.median(samp_point_R_mat[i, comp_mat[i, :] == 1]))
                QPOS = np.round(np.median(samp_point_Q_mat[i, comp_mat[i, :] == 1]))
                SPOS = np.round(np.median(samp_point_S_mat[i, comp_mat[i, :] == 1]))
                QRSoffPOS = np.round(np.median(samp_point_QRSoff_mat[i, comp_mat[i, :] == 1]))
                JPOS = np.round(np.median(samp_point_J_mat[i, comp_mat[i, :] == 1]))
                TonPOS = np.round(np.median(samp_point_Ton_mat[i, comp_mat[i, :] == 1]))
                TpeakPOS = np.round(np.median(samp_point_Tpeak_mat[i, comp_mat[i, :] == 1]))
                ToffPOS = np.round(np.median(samp_point_Toff_mat[i, comp_mat[i, :] == 1]))
                M = np.vstack([M, np.array([PonPOS, PpeakPOS, PoffPOS, QRSonPOS, QPOS, RPOS, SPOS, QRSoffPOS, JPOS,
                                                TonPOS, TpeakPOS, ToffPOS, 0])])
                for l in range(0, N_Channels):
                    if comp_mat[i, l]:
                        FPT_Cell_Synced[l] = np.vstack([FPT_Cell_Synced[l], np.hstack([FPT_Cell[l][pos_mat[i, l], 0:12], 0])])
                    else:
                        FPT_Cell_Synced[l] = np.vstack([FPT_Cell_Synced[l], M[-1, :]])
        for j in range(0, N_Channels):
            FPT_Cell[j] = FPT_Cell[j][pos_mat[pos_mat[:, j] <= 0, j], :]

    if M.shape[0] == 0:
        print('Warning: Too little QRS complexes were detected. Returning an empty FPT table')
        FPT_Synced = []
        return FPT_Synced, FPT_Cell_Synced

    FPT_Synced = np.copy(M)
    ind = np.argsort(M[:, 5])

    FPT_Synced[:, 0:12] = M[ind, 0:12]

    for i in range(0, N_Channels):
        FPT_Cell_Synced[i] = FPT_Cell_Synced[i][ind, :]

    ind = np.unique(FPT_Synced[:, 5], return_index=True)[1]

    FPT_Synced = FPT_Synced[ind, :]

    for i in range(0, N_Channels):
        FPT_Cell_Synced[i] = FPT_Cell_Synced[i][ind, :]

    FPT_Synced, keep_index = Check_Small_RR(FPT_Synced, fs)
    for i in range(0, N_Channels):
        FPT_Cell_Synced[i] = FPT_Cell_Synced[i][keep_index, :]

    FPT_Synced, keep_index = Check_Position_ECG_Waves(FPT_Synced)
    for i in range(0, N_Channels):
        FPT_Cell_Synced[i] = FPT_Cell_Synced[i][keep_index, :]

    if FPT_Synced.shape[0] < 3:
        print('Warning: Too little QRS complexes were detected. Returning an empty FPT table')
        FPT_Synced = []
        for i in range(0, N_Channels):
            FPT_Cell_Synced[i] = []

    return FPT_Synced, FPT_Cell_Synced


def Check_Position_ECG_Waves(FPT):
    arr = np.zeros([13], dtype=bool)
    arr[1] = True
    arr[5] = True
    arr[10] = True
    ind_waves = np.logical_and(np.sum(FPT, axis=0) > 0, arr)
    FPT_Checked = FPT
    keep_index = np.empty(0)
    if np.any(ind_waves):
        M = FPT[:, ind_waves]
        A = np.diff(M)
        remove1 = np.logical_not(np.all(A > 0, axis=1))

        i = np.arange(0, M.shape[0] - 1)
        remove2 = np.logical_not(M[i, -1] < M[i + 1, 0])
        remove2 = np.hstack([remove2, False])
        keep = np.logical_not(np.logical_or(remove1, remove2))
        FPT_Checked = FPT_Checked[keep, :]
        keep_index = np.where(keep)[0]

    return FPT_Checked, keep_index


def QRS_detection(raw_signal, fs):
    # Pre-filter the signal
    highpass_frequency = 0.5
    lowpass_frequency = 30
    bandpass = sc.signal.butter(2, [highpass_frequency, lowpass_frequency], btype='bandpass', fs=fs, output='sos')
    padlen = np.round(np.min([0.9*raw_signal.shape[0],10*fs])).astype(int)
    signal = sc.signal.sosfiltfilt(bandpass, raw_signal, axis=0, padtype='odd', padlen=padlen)
    flag_posR = False
    # downsample the signal to increase speed
    # fdownsample = 400;
    # flagdownsample = false
    # if samplerate > fdownsample:
    #    oldsamplerate = samplerate
    #    oldsignal = signal

    #    r = np.floor(samplerate / fdownsample);
    #    signal = decimate(signal, r); % downsampling
    #    samplerate = samplerate / r; % new
    #    flagdownsample = true;

    # Wavelet level
    x = int(np.ceil(np.log2(fs / 2 / 30)))
    if x < 1:
        x = 1

    # extend signal
    if np.log2(signal.shape[0]) == np.ceil(np.log2(signal.shape[0])):
        l = 2 ** (np.ceil(np.log2(signal.shape[0])) + 1)
    else:
        l = 2 ** np.ceil(np.log2(signal.shape[0]))

    l1 = int(np.floor((l - signal.shape[0]) / 2))
    l2 = int(l - signal.shape[0] - l1)
    ecg_w = np.pad(signal, [(l1, l2), (0, 0)])
    for ld in range(0, signal.shape[1]):
        left = np.kron(np.ones(l1), signal[0, ld])
        right = np.kron(np.ones(l2), signal[-1, ld])
        ecg_w[:, ld] = np.concatenate([left.transpose(), signal[:, ld], right.transpose()])
    # Wavelet transform
    sig_wt = pywt.swt(ecg_w, 'haar', level=x, axis=0, trim_approx=False, norm=False)
    Dx1 = sig_wt[0][1]
    Dx1 = Dx1[l2:-l1, :]

    sig_wt = pywt.swt(ecg_w[::-1, :], 'haar', level=x, axis=0, trim_approx=False, norm=False)
    Dx2 = sig_wt[0][1]
    Dx2 = Dx2[::-1, :]
    Dx2 = Dx2[l2:-l1, :]

    Dx = np.abs(Dx1 + Dx2)
    Dx = Dx / np.std(Dx, axis=0)
    saturation = np.quantile(Dx, 0.99, axis=0)
    for ld in range(0, Dx.shape[1]):
        Dx[Dx[:, ld] > saturation[ld], ld] = saturation[ld]

    Thbegin = 1
    Thend = np.quantile(Dx, 0.95, axis=0) / saturation
    threshold = np.linspace(Thbegin, Thend, 20)

    Tl = 4
    nrep = 3
    QRS_pos = list()
    R_Cell = list()

    for ld in range(0, Dx.shape[1]):
        R_Cell.append(list())
        for j in range(0, nrep):
            NR_vec = np.zeros(threshold.shape[0])

            n1 = int(np.floor(fs * Tl))
            n2 = int(np.floor(signal.shape[0] / n1) - 1)
            rms_Dx_base = np.zeros(Dx.shape[0])
            for i in range(0, n2 + 1):
                if n2 == 0:
                    rms_Dx_base = np.quantile(Dx[round(0.1 * fs):Dx.shape[0] - round(0.1 * fs), ld], 0.95,
                                              axis=0) * np.ones(Dx.shape[0])
                else:
                    if i == 0:
                        rms_Dx_base[0:n1] = np.quantile(Dx[round(0.1 * fs):Dx.shape[0] - round(0.1 * fs), ld], 0.95,
                                                        axis=0) * np.ones(Dx[i * n1:(i + 1) * n1].shape[0])
                    elif i == n2:
                        rms_Dx_base[i * n1:-1] = np.quantile(Dx[i * n1:-1, ld], 0.95, axis=0) * np.ones(
                            Dx[i * n1:-1, :].shape[0])
                    else:
                        rms_Dx_base[i * n1:(i + 1) * n1] = np.quantile(Dx[i * n1:(i + 1) * n1, ld], 0.95,
                                                                       axis=0) * np.ones(
                            Dx[i * n1:(i + 1) * n1, :].shape[0])

            for H in range(0, threshold.shape[0]):
                if H == threshold.shape[0] - 1:
                    mt = np.argmin(np.diff(NR_vec[0:H - 2]), axis=0)
                    rms_Dx = threshold[mt, ld] * rms_Dx_base
                else:
                    rms_Dx = threshold[H, ld] * rms_Dx_base
                candidates_Dx = Dx[:, ld] > rms_Dx

                Can_Sig_Dx = np.zeros(Dx.shape[0])
                Can_Sig_Dx[candidates_Dx] = 1
                Can_Sig_Dx[0] = 0
                Can_Sig_Dx[-1] = 0

                i = np.arange(0, Can_Sig_Dx.shape[0])
                i = i[0:-2]
                Bound_A = np.argwhere(np.logical_and(Can_Sig_Dx[i] == 0, Can_Sig_Dx[i + 1] > 0))
                Bound_B = np.argwhere(np.logical_and(Can_Sig_Dx[i] > 0, Can_Sig_Dx[i + 1] == 0))
                Bound_AB = np.zeros([np.max([Bound_B.shape[0], Bound_A.shape[0]]), 2])

                for k in range(0, Bound_A.shape[0]):
                    idx = np.where(Bound_B > Bound_A[k])[0]
                    if idx.shape[0]:
                        idx = idx[0]
                        if (Bound_B[idx]-Bound_A[k]) / fs <= 0.1 and Bound_B.shape[0]-1 >= idx and Bound_A.shape[0]-1 >= idx:
                            Bound_AB[k, 0] = Bound_A[k]
                            Bound_AB[k, 1] = Bound_B[idx]
                Bound_A = Bound_AB[Bound_AB.any(axis=1), 0]
                Bound_B = Bound_AB[Bound_AB.any(axis=1), 1]


                ind = np.argwhere(np.logical_or(Bound_B - Bound_A / fs < 5e-3, Bound_B - Bound_A / fs > 0.25))
                Bound_B = Bound_B[ind[:, 0]]
                Bound_A = Bound_A[ind[:, 0]]

                NR_vec[H] = Bound_A.shape[0]
                if H > 1:
                    dNR = NR_vec[H] - NR_vec[H - 1]
                    if dNR <= 0 or H == threshold.shape[0]:
                        if Bound_A.shape[0] <= 1 or Bound_B.shape[0] <= 1:
                            continue
                        else:
                            Tl = np.quantile(np.diff(Bound_A) / fs, 0.98) * 4
                            break

            QRS_pos.append(1 / 2 * (Bound_A + Bound_B))
            R_Cell[ld].append(np.round(QRS_pos[j]))

    FPT_Cell = list()
    for ld in range(0, Dx.shape[1]):
        FPT_Cell.append(list())
        for i in range(0, len(R_Cell[ld])):
            FPT_Cell[ld].append(np.zeros([R_Cell[ld][i].shape[0], 13]))
            FPT_Cell[ld][i][:, 5] = R_Cell[ld][i][:]

    R_Synced = list()
    for ld in range(0, Dx.shape[1]):
        FPT_Cell[ld] = Sync_R_Peaks(FPT_Cell[ld], fs)[0]  # Check why slow
        R_Synced.append(FPT_Cell[ld][:, 5].astype(int))


    # if flagdownsample:
    #     samplerate = oldsamplerate
    #     signal = oldsignal
    #     R_Synced = R_Synced * r
    for ld in range(0, Dx.shape[1]):
        if not R_Synced[ld].all:
            print('No QRS complexes were found. Returning an empty FPT table')
            FPT = []
            return
        else:
            WB = int(np.round(0.05 * fs))
            QRS_region = np.array([R_Synced[ld] - WB, R_Synced[ld] + WB]).transpose()

            if QRS_region[0, 0] < 0:
                ind = np.where(QRS_region[0] >= 0)[0][0]
                R_Synced[ld] = R_Synced[ld][ind:-1]

            if QRS_region[0, 1] >= signal.shape[0]:
                ind = np.where(QRS_region[0] < signal.shape[0])[0][-1]
                R_Synced[ld] = R_Synced[ld][ind:-1]

            FPT = np.zeros([R_Synced[ld].shape[0], 13])

        if R_Synced[ld].shape[0] < 3:
            print('Too little QRS complexes were detected. Returning an empty FPT table')
            FPT = []
            return

        RPOS_vector = np.zeros([FPT.shape[0]], dtype=int)
        QPOS_vector = np.zeros([FPT.shape[0]], dtype=int)
        SPOS_vector = np.zeros([FPT.shape[0]], dtype=int)

        if not flag_posR:
            dsignal = np.diff(signal[:, ld])
            i = np.arange(0, dsignal.shape[0] - 1)
            I_ext = np.where(np.logical_or(np.logical_and(dsignal[i] >= 0, dsignal[i + 1] < 0),
                                           np.logical_and(dsignal[i] < 0, dsignal[i + 1] >= 0)))[0]

            RR = np.diff(R_Synced[ld])
            X = np.vstack((RR[0:-2], RR[1:-1])).transpose()
            index = np.arange(0, X.shape[0])
            SCORE = np.matmul((X - np.hstack(
                (np.mean(X[index, 0]) * np.ones([X.shape[0], 1]), np.mean(X[index, 1]) * np.ones([X.shape[0], 1])))) * (
                                          1 / np.sqrt(2)), np.array([[1, -1], [1, 1]]))
            D1 = np.abs(SCORE[:, 0])
            Thl1 = 2.5 * np.std(D1)
            index = np.logical_and(SCORE[:, 0] >= -Thl1, SCORE[:, 1] <= 0)
            Ind_QRS_normal = np.where(index)[0]+1
            Ind_QRS_normal = Ind_QRS_normal[1:-2].astype(int)
            QRS_Matrix = np.zeros((2 * WB, Ind_QRS_normal.shape[0]))
            MP = np.zeros((Ind_QRS_normal.shape[0], 2))
            for k in range(0, Ind_QRS_normal.shape[0]):
                QRS_Matrix[:, k] = signal[R_Synced[ld][Ind_QRS_normal[k]] - WB:R_Synced[ld][Ind_QRS_normal[k]] + WB, ld]
                MP[k, :] = np.array([np.max(QRS_Matrix[:, k]), np.min(QRS_Matrix[:, k])])
            if MP.shape[0] > 0:
                Th11 = np.quantile(MP[:, 0], 0.25)
                Th12 = np.quantile(MP[:, 0], 0.75)
                Th21 = np.quantile(MP[:, 1], 0.25)
                Th22 = np.quantile(MP[:, 1], 0.75)
                QRS_Matrix_selected = QRS_Matrix[:, np.logical_and(np.logical_and(MP[:, 0] >= Th11, MP[:, 0] <= Th12),
                                                                   np.logical_and(MP[:, 1] >= Th21, MP[:, 1] <= Th22))]
                Template = np.mean(QRS_Matrix_selected, axis=1)
            else:
                Template = np.mean(QRS_Matrix, axis=1)


            R_type = np.sign(np.max(Template) + np.min(Template))

            biph_crit = 2 / 5
            w_crit = 9 / 10
            for i in range(0, RPOS_vector.shape[0]):
                tmp_ZC = np.where(np.logical_and(I_ext >= QRS_region[i, 0] - WB, I_ext <= QRS_region[i, 1] + WB))[0]

                if tmp_ZC.shape[0] <= 0:
                    RPOS_vector[i] = round((QRS_region[i, 0] + QRS_region[i, 1]) / 2)
                    QPOS_vector[i] = QRS_region[i, 0]
                    SPOS_vector[i] = QRS_region[i, 1]
                elif tmp_ZC.shape[0] == 1:
                    RPOS_vector[i] = I_ext[tmp_ZC[0]]
                    QPOS_vector[i] = QRS_region[i, 0]
                    SPOS_vector[i] = QRS_region[i, 1]
                else:
                    amplitude = np.sort(signal[I_ext[tmp_ZC], ld])
                    index = np.argsort(signal[I_ext[tmp_ZC], ld])
                    if np.min([np.abs(amplitude[0] / amplitude[-1]), np.abs(amplitude[-1] / amplitude[1])]) > biph_crit:
                        if R_type >= 0:
                            if np.abs(amplitude[-2] / amplitude[-1]) < w_crit:
                                RPOS_vector[i] = I_ext[tmp_ZC[index[-1]]]
                                Qpeak = index[-1] - 1
                                Speak = index[-1] + 1
                            else:
                                RPOS_vector[i] = np.min([I_ext[tmp_ZC[index[-1]]], I_ext[tmp_ZC[index[-2]]]])
                                Qpeak = np.min([index[-2], index[-1]]) - 1
                                Speak = np.max([index[-2], index[-1]]) + 1

                        else:
                            if np.abs(amplitude[1] / amplitude[0]) < w_crit:
                                RPOS_vector[i] = I_ext[tmp_ZC[index[0]]]
                                Qpeak = index[0] - 1
                                Speak = index[0] + 1
                            else:
                                RPOS_vector[i] = np.min([I_ext[tmp_ZC[index[0]]], I_ext[tmp_ZC[index[1]]]])
                                Qpeak = np.min([index[1], index[0]]) - 1
                                Speak = np.max([index[1], index[0]]) + 1

                    elif np.abs(amplitude[-1]) > np.abs(amplitude[0]):
                        if np.abs(amplitude[-2] / amplitude[-1]) < w_crit:
                            RPOS_vector[i] = I_ext[tmp_ZC[index[-1]]]
                            Qpeak = index[-1] - 1
                            Speak = index[-1] + 1
                        else:
                            RPOS_vector[i] = np.min([I_ext[tmp_ZC[index[-1]]], I_ext[tmp_ZC[index[-2]]]])
                            Qpeak = np.min([index[-2], index[-1]]) - 1
                            Speak = np.max([index[-2], index[-1]]) + 1

                    else:
                        if np.abs(amplitude[1] / amplitude[0]) < w_crit:
                            RPOS_vector[i] = I_ext[tmp_ZC[index[0]]]
                            Qpeak = index[0] - 1
                            Speak = index[0] + 1
                        else:
                            RPOS_vector[i] = np.min([I_ext[tmp_ZC[index[0]]], I_ext[tmp_ZC[index[1]]]])
                            Qpeak = np.min([index[1], index[0]]) - 1
                            Speak = np.max([index[1], index[0]]) + 1

                    if Qpeak > 0:
                        QPOS_vector[i] = I_ext[tmp_ZC[Qpeak]]
                    else:
                        QPOS_vector[i] = RPOS_vector[i] - WB
                    if Speak < tmp_ZC.shape[0]:
                        SPOS_vector[i] = I_ext[tmp_ZC[Speak]]
                    else:
                        SPOS_vector[i] = RPOS_vector[i] + WB

        else:
            QRS_region[QRS_region <= 0] = 1
            QRS_region[QRS_region > signal.shape[0]] = signal.shape[0]
            for i in range(0, RPOS_vector.shape[0]):
                rpeak = np.argmax(signal[QRS_region[i, 0]:QRS_region[i, 1], ld])
                RPOS_vector[i] = rpeak + QRS_region[i, 0]
                speak = np.argmin(signal[RPOS_vector[i]:QRS_region[i, 1], ld])
                SPOS_vector[i] = RPOS_vector[i] + speak
                qpeak = np.argmin(signal[QRS_region[i, 0]:RPOS_vector[i], ld])
                QPOS_vector[i] = QRS_region[i, 0] + qpeak

            if QPOS_vector[0] < 2:
                QPOS_vector[0] = 2

            if SPOS_vector[-1] > signal.shape[0] - 1:
                SPOS_vector[-1] = signal.shape[0] - 1

        donoff = np.round(25e-3 * fs).astype(int)
        QRSonPOS_vector = QPOS_vector - donoff
        QRSonPOS_vector[0] = np.max([0, QRSonPOS_vector[0]])
        QRSoffPOS_vector = SPOS_vector + donoff
        QRSoffPOS_vector[-1] = np.min([QRSoffPOS_vector[-1], signal.shape[0]])

        FPT_Cell[ld] = np.zeros([RPOS_vector.shape[0], 13])
        FPT_Cell[ld][:, 3] = QRSonPOS_vector
        FPT_Cell[ld][:, 4] = QPOS_vector
        FPT_Cell[ld][:, 5] = RPOS_vector
        FPT_Cell[ld][:, 6] = SPOS_vector
        FPT_Cell[ld][:, 7] = QRSoffPOS_vector

    FPT_Multichannel, FPT_Cell = Sync_Beats(FPT_Cell, fs)

        # remove = find(diff(FPT(:, 6)) / samplerate < 0.25);
        # FPT(remove + 1,:)=[];

    return FPT_Multichannel, FPT_Cell
