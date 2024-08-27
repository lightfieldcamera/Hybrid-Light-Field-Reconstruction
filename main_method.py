import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import cv2
import lightfieldpackage
import os
from shutil import copyfile
import sys

class Net(nn.Module):
    def __init__(self, data_source_torch, is_1D):
        super(Net, self).__init__()
        data_torch = torch.tensor(data_source_torch, requires_grad=True, device="cuda", dtype=torch.float)
        self.image_source = torch.nn.Parameter(data_torch)
        self.dim = 0 if is_1D else (0,1)

    def forward(self):
        self.image = torch.sum(self.image_source,
                               # self.image = torch.sum(self.image_source,
                               dim=self.dim)  # die Projection muss hier und nicht in der init stehen, weil sie muss ja nach jedem Trainingsschritt neu berechnet werden
        return self.image

    def get_lf(self):
        return self.image_source

    def criterion(self, prediction, label):
        return nn.MSELoss()(prediction, label) # MSE macht hier deutlich mehr Sinn als Cross Entropy. Weil CE wird bei Classification und Segmentation verwendet und Segmentation ist ja auch Classification je Pixel.
        #return nn.CrossEntropyLoss()(prediction, label)

def main():
    fontsize_label = 18
    fontsize_legend = 12
    fontsize_ticks = 12

    # usaf
    path_USAF_conv = r'log\log_20220621_122209_conv_2D_paraxial_no-phasemask_my_paper_usaf_wie_broxton\irradiance.npy'
    path_USAF_plen = r'log\log_20220621_115800_plen10_2D_paraxial_no-phasemask_my_paper_usaf_wie_broxton\irradiance.npy'
    output_folder = r'results_princeton'

    cmap = 'gray'
    flip_axis = (0, 1)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    copyfile(os.path.basename(__file__), os.path.join(output_folder, os.path.basename(__file__)))

    sys.path.append(os.path.dirname(os.path.abspath(path_USAF_plen)))
    import generate_usaf_4f_system_2D as config

    method = "fourier800"  # "spatial700", "fourier800", "just_refocusing"
    is_1D = False
    n_epochs = 5
    threshold = 1e-20
    lr = 5000
    Network = Net  #
    comment = "fourier800"  # "Non Neg"#"Avg LF" #"Clamp"#"Hybrid" #
    N_MLA = config.N_MLA  # Anzahl Mikrolinsen
    N_MP = config.N_MP  # Anzahl Mikropixel
    tile_size = lightfieldpackage.utils.round_up_to_odd(N_MP)

    downsample_factor = tile_size  # was ist das?

    D_ML1 = config.D_ML1  # Linsendurchmesser ML
    D_S = config.D_S  # Sensorgröße
    F = config.B_ML_MLA

    USAF_conv_Z = np.load(path_USAF_conv)
    USAF_plen_Z = np.load(path_USAF_plen)

    inds_of_interest = [0, 1, 2, 3]

    USAF_conv_Z = USAF_conv_Z[inds_of_interest, :, :]
    USAF_plen_Z = USAF_plen_Z[inds_of_interest, :, :]

    # zu viel RAM und GPU, deswegen croppe ich das Lichtfeld und das ConvCam Bild
    x1_original = 648
    x2_original = 1408
    y1_original = 648
    y2_original = 1408

    x1 = N_MP*(x1_original//N_MP) # 648
    x2 = N_MP*(x2_original//N_MP+1) # 1408
    y1 = N_MP*(y1_original//N_MP) # 648
    y2 = N_MP*(y2_original//N_MP+1) # 1408

    USAF_conv_Z = USAF_conv_Z[:,y1:y2,x1:x2]
    USAF_plen_Z = USAF_plen_Z[:,y1:y2,x1:x2]
    N_MLA_new = (y2_original//N_MP+1)-(y1_original//N_MP)

    for i in range(USAF_plen_Z.shape[0]):
        #F2 = config.B_ML_MLA + M*(config.g_ML_focus - config.g_ML[g_ind])
        #F2 = config.B_ML_MLA + (1/(1/config.f_ML2-1/config.g_ML_focus)+config.g_ML_focus)*M
        F2 = 1/(1/config.f_ML2+1/config.b_ML[inds_of_interest][i])
        # F2 = 1/(1/config.f_ML2-1/config.g_MLA[g_ind]) #

        alpha = F2 / F
        alphas = np.array([alpha])

        # load data
        irradiance_np_conv = USAF_conv_Z[i,:,:]
        if is_1D:
            irradiance_np_interp_conv = np.squeeze(
                irradiance_np_conv)  # cv2.resize(irradiance_np_conv, (500, 800), interpolation=cv2.INTER_CUBIC)
            irradiance_np_conv = irradiance_np_interp_conv

        irradiance_np_plen = USAF_plen_Z[i,:,:]
        # irradiance_np_interp_plen = cv2.resize(irradiance_np_plen, (500, 800), interpolation=cv2.INTER_CUBIC)
        if is_1D:
            irradiance_np_interp_plen = np.squeeze(irradiance_np_plen)
            irradiance_np_plen = irradiance_np_interp_plen

        if not is_1D:
            irradiance_np_conv = np.array([irradiance_np_conv], dtype=np.float32)
            irradiance_np_plen = np.array([irradiance_np_plen], dtype=np.float32)

        if method == "fourier800":
            refocus_before_optim_array, refocus_after_optim_array, lf_after_optim_array, refocus_before_optim_no_interp_array = lightfieldpackage.utils.optimize_with_princeton_method_800(
                irradiance_np_conv=irradiance_np_conv,
                irradiance_np_plen=irradiance_np_plen, alphas=alphas,
                downsample_factor=downsample_factor, N_MLA=N_MLA_new,
                N_MP=tile_size, D_S=D_S*N_MLA_new/N_MLA, D_ML=D_ML1, is_1D=is_1D, n_epochs=n_epochs, threshold=threshold, with_scaling = False,interp_method=cv2.INTER_NEAREST)#, x1=x1, x2=x2, y1=y1, y2=y2)

        elif method == "spatial700":
            refocus_before_optim_array, refocus_after_optim_array, lf_after_optim_array, refocus_before_optim_no_interp_array = lightfieldpackage.utils.optimize_with_princeton_method_700(irradiance_np_conv=irradiance_np_conv,
                                                                         irradiance_np_plen=irradiance_np_plen, alphas=alphas,
                                                                         downsample_factor=downsample_factor, N_MLA=N_MLA_new,
                                                                         N_MP=tile_size, D_S=D_S, D_ML=D_ML1, Net=Network, lr=lr, is_1D=is_1D, n_epochs=n_epochs, threshold=threshold,interp_method=cv2.INTER_NEAREST)

        # _, refocus_after_optim_array_non_neg, lf_after_optim_array_non_neg = optimize_with_princeton_method(irradiance_np_conv=irradiance_np_conv,
        #                                                 irradiance_np_plen=irradiance_np_plen, g_ML=g_ML,
        #                                                 B_ML_MLA=B_ML_MLA, b_ML=b_ML,
        #                                                 downsample_factor=downsample_factor, N_MLA=N_MLA,
        #                                                 N_MP=N_MP, D_S=D_S, D_ML=D_ML, Net=Net_NonNeg, lr=lr2, is_1D=is_1D, n_epochs=n_epochs, threshold=threshold)

        elif method == "just_refocusing":
            refocus_before_optim_array, refocus_after_optim_array, lf_after_optim_array, refocus_before_optim_no_interp_array = lightfieldpackage.utils.no_optim_just_refocus(irradiance_np_conv=irradiance_np_conv,
                                                                         irradiance_np_plen=irradiance_np_plen, alphas=alphas,
                                                                         downsample_factor=downsample_factor, N_MLA=N_MLA_new,
                                                                         N_MP=tile_size, D_S=D_S, D_ML=D_ML1, Net=Network, lr=lr, is_1D=is_1D, n_epochs=n_epochs, threshold=threshold)

        refocus_before_optim_array = np.squeeze(refocus_before_optim_array)[y1_original-y1:y1_original-y1+y2_original-y1_original,x1_original-x1:x1_original-x1+x2_original-x1_original]
        refocus_after_optim_array = np.squeeze(refocus_after_optim_array)[y1_original-y1:y1_original-y1+y2_original-y1_original,x1_original-x1:x1_original-x1+x2_original-x1_original]
        refocus_before_optim_no_interp_array = np.squeeze(refocus_before_optim_no_interp_array)[y1_original-y1:y1_original-y1+y2_original-y1_original,x1_original-x1:x1_original-x1+x2_original-x1_original]

        plt.imsave(os.path.join(output_folder, "refocus_before_optim_array_"+str(i).zfill(2)+".png"), np.flip(refocus_before_optim_array, axis=flip_axis), cmap=cmap)
        plt.imsave(os.path.join(output_folder, "refocus_after_optim_array_"+str(i).zfill(2)+".png"), np.flip(refocus_after_optim_array, axis=flip_axis), cmap=cmap)
        plt.imsave(os.path.join(output_folder, "refocus_before_optim_no_interp_array_"+str(i).zfill(2)+".png"), np.flip(cv2.resize(refocus_before_optim_no_interp_array, refocus_after_optim_array.shape, 0, 0, interpolation=cv2.INTER_NEAREST), axis=flip_axis), cmap=cmap)

if __name__ == '__main__':
    main()
