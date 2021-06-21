# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 20:04:36 2020

@author: John
"""

import quantities as pq
import numpy as np
import matplotlib.pyplot as plt
import pylgn
import pylgn.kernels as kernel
import my_kernels as mk
from scipy.io import loadmat
from dataclasses import dataclass
import csv


@dataclass
class cell:
    name: str = None
    Spont_mean: float = None
    Spont_SEM: float = None
    rf_sta_lat: float = None
    rf_xdeg: float = None
    rf_ydeg: float = None
    rf_center: float = None
    rf_sigma: float = None
    sf_tuning: float = None
    sf_preferred: float = None
    sf_cutoff: float = None
    sf_latency: float = None
    sf_spikes: float = None
    tf_tuning: float = None
    tf_spikes: float = None
    c_tuning: float = None
    c_gain: float = None
    c_c50: float = None
    c_fit: float = None
    c_spikes: float = None
    d_tuning: float = None
    d_preferred: float = None
    d_DSI: float = None
    d_OSI: float = None
    d_spikes: float = None
    f_transient_infex: float = None
    f_spikes: float = None
    sf_curvefit: float = None
    c_curvefit: float = None


@dataclass
class mu_cell:
    mouse: str = None
    ID: str = None
    include: bool = False
    g1_xloc: float = None  # x1
    g1_yloc: float = None  # x2
    g2_xloc: float = None  # x3
    g2_yloc: float = None  # x4
    g1_xvar: float = None  # x5
    g1_rot: float = None  # x6 (clockwise)
    g1_yvar: float = None  # x7
    g2_xvar: float = None  # x8
    g2_rot: float = None  # x9 (clockwise)
    g2_yvar: float = None  # x10
    g1_tcen: float = None  # x11 ("ie centre at x11 ms before")
    g2_tcen: float = None  # x12 ("ie centre at x11 ms before")
    g1_tvar: float = None  # x13
    g2_tvar: float = None  # x14
    g1_tamp: float = None  # x15
    g2_tamp: float = None  # x16


# Load the matlab file into x, type dict
# x = loadmat('C:\\Users\\John\\Documents\\UNI\\Yr_3_Neuro-Project\\Dryad_Data\\doi_10.5061_dryad.b2t22__v1\\cells.mat')
mu_x = loadmat('D:\\Uni\\Year_3\\Neuro\\Data\\qmi_rfvs_gauss_fit-extracted.mat')
# print((mu_x["rfv_d2g"][0].tolist()[7]))
# mu_xlist = mu_x["rfv_d2g"][0,3].tolist()

# Initialise the empty list to load cell data into
# cells = []
mu_cells = []
'''
#1. Iterate through the list of cells in x
for tempCellMat in range(x["cells"].size):
    x_tolist = x["cells"][0,tempCellMat].tolist()
    x_tolist[0]

    #2. Form a temporary cell from the fields
    #TODO: simplify this?
    tempCellPy = cell()
    tempCellPy.name = x_tolist[0]
    tempCellPy.Spont_mean = x_tolist[1]
    tempCellPy.Spont_SEM = x_tolist[2]
    tempCellPy.rf_sta_lat = x_tolist[3]
    tempCellPy.rf_xdeg = x_tolist[4]
    tempCellPy.rf_ydeg = x_tolist[5]
    tempCellPy.rf_center = x_tolist[6]
    tempCellPy.rf_sigma = x_tolist[7]
    tempCellPy.sf_tuning = x_tolist[8]
    tempCellPy.sf_preferred = x_tolist[9]
    tempCellPy.sf_cutoff =  x_tolist[10]
    tempCellPy.sf_latency = x_tolist[11]
    tempCellPy.sf_spikes = x_tolist[12]
    tempCellPy.tf_tuning = x_tolist[13]
    tempCellPy.tf_spikes = x_tolist[14]
    tempCellPy.c_tuning = x_tolist[15]
    tempCellPy.c_gain = x_tolist[16]
    tempCellPy.c_c50 = x_tolist[17]
    tempCellPy.c_fit = x_tolist[18]
    tempCellPy.c_spikes = x_tolist[19]
    tempCellPy.d_tuning = x_tolist[20]
    tempCellPy.d_preferred = x_tolist[21]
    tempCellPy.d_DSI = x_tolist[22]
    tempCellPy.d_OSI = x_tolist[23]
    tempCellPy.d_spikes = x_tolist[24]
    tempCellPy.f_transient_index = x_tolist[25]
    tempCellPy.f_spikes = x_tolist[26]
    tempCellPy.sf_curvefit = x_tolist[27]
    tempCellPy.c_curvefit = x_tolist[28]

    #Add the temporary cell to the full list of cells
    cells.append(tempCellPy)'''

# print((mu_x["rfv_d2g"][0].tolist()[7]))

#for mu_cell_no in range(1297):
for mu_cell_no in range(1):
    mu_x_tolist = mu_x["rfv_d2g"][mu_cell_no].tolist()

    # 2. Form a temporary cell from the fields
    # TODO: simplify this?
    if (mu_x_tolist[2] == 1):  # checks whether the cell was fitted with dog
        tempMuCellPy = mu_cell()
        tempMuCellPy.mouse = mu_x_tolist[0]
        tempMuCellPy.ID = mu_x_tolist[1]
        tempMuCellPy.g1_xloc = mu_x_tolist[4]  # x1
        tempMuCellPy.g1_yloc = mu_x_tolist[5]  # x2
        tempMuCellPy.g2_xloc = mu_x_tolist[6]  # x3
        tempMuCellPy.g2_yloc = mu_x_tolist[7]  # x4
        tempMuCellPy.g1_xvar = mu_x_tolist[8]  # x5
        tempMuCellPy.g1_rot = mu_x_tolist[9]  # x6 (clockwise)
        tempMuCellPy.g1_yvar = mu_x_tolist[10]  # x7
        tempMuCellPy.g2_xvar = mu_x_tolist[11]  # x8
        tempMuCellPy.g2_rot = mu_x_tolist[12]  # x9 (clockwise)
        tempMuCellPy.g2_yvar = mu_x_tolist[13]  # x10
        tempMuCellPy.g1_tcen = mu_x_tolist[14]  # x11 ("ie centre at x11 ms before")
        tempMuCellPy.g2_tcen = mu_x_tolist[15]  # x12 ("ie centre at x11 ms before")
        tempMuCellPy.g1_tvar = mu_x_tolist[16]  # x13
        tempMuCellPy.g2_tvar = mu_x_tolist[17]  # x14
        tempMuCellPy.g1_tamp = mu_x_tolist[18]  # x15
        tempMuCellPy.g2_tamp = mu_x_tolist[19]  # x16

        # Add the temporary cell to the full list of cells
        mu_cells.append(tempMuCellPy)

# PyLGN MODEL

# cell_responses = []
w = csv.writer(open("cell_responses_cut.csv", "w"))  # file location to save responses to

for cell_no in range(len(mu_cells)):
    # select a cell
    selected_cell_no = cell_no
    # if mu_cells[selected_cell_no].ID == ['ID:2. ch1-3']: #search for a particular cell ID - need to indent code bellow
    X = mu_cells[selected_cell_no]

    k_max_id = 50
    feedback_strength = np.array([0.0])  # , 0.5, 0.8 - varies cortical feedback strength values if non-zero
    response = np.zeros([k_max_id, len(feedback_strength)]) / pq.s

    for j, d in enumerate(feedback_strength):
        # create network
        network = pylgn.Network()

        # create integrator
        integrator = network.create_integrator(nt=5, nr=7, dt=1 * pq.ms,
                                               dr=1 * pq.deg)  # nt=8, nr=9, dt=1*pq.ms, dr=0.1*pq.deg
        spatial_angular_freqs = integrator.spatial_angular_freqs[:k_max_id]

        # create kernels
        G = mk.create_gauss_ft(A=1, a=1 * pq.deg, dx=0 * pq.deg, dy=0 * pq.deg, xscale=X.g1_xvar, yscale=X.g1_yvar,
                               theta=X.g1_rot)  # A=1, a=1*pq.deg,dx=X.g1_xloc*pq.deg, dy=X.g1_yloc*pq.deg,xscale=X.g1_xvar,yscale=X.g1_yvar, theta=X.g1_rot
        G1 = mk.create_gauss_ft(A=1, a=1 * pq.deg, dx=(X.g2_xloc - X.g1_xloc) * pq.deg,
                                dy=(X.g2_yloc - X.g1_yloc) * pq.deg, xscale=X.g2_xvar, yscale=X.g2_yvar,
                                theta=X.g2_rot)  # A=1, a=1*pq.deg,dx=X.g2_xloc*pq.deg, dy=X.g2_yloc*pq.deg,xscale=X.g2_xvar,yscale=X.g2_yvar, theta=X.g2_rot
        T = mk.create_temp_gauss_ft(A=X.g1_tamp, a=X.g1_tvar * pq.ms, delay=X.g1_tcen * pq.ms)
        T1 = mk.create_temp_gauss_ft(A=X.g2_tamp, a=X.g2_tvar * pq.ms, delay=X.g2_tcen * pq.ms)
        Kd_t = kernel.temporal.create_delta_ft()
        Kd_s = kernel.spatial.create_delta_ft()
        # Kdg_s = kernel.spatial.create_dog_ft(A=1, a=0.62*pq.deg, B=0.85, b=1.26*pq.deg)
        Kg_s = kernel.spatial.create_gauss_ft(A=1, a=0.1 * pq.deg)
        Kg_s1 = kernel.spatial.create_gauss_ft(A=1, a=0.3 * pq.deg)
        Kg_s2 = kernel.spatial.create_gauss_ft(A=1, a=0.83 * pq.deg)

        # create neuron
        ganglion = network.create_ganglion_cell(kernel=(Kd_s, Kd_t))
        relay = network.create_relay_cell()
        cortical = network.create_cortical_cell()

        # connect neurons
        network.connect(ganglion, relay, (G, T), weight=1)  # KRG
        network.connect(ganglion, relay, (G1, T1), weight=-1)  # KRG
        # network.connect(ganglion, relay, (G1, Kd_t),weight=-1) #KRG
        # network.connect(ganglion, relay, (Kg_s1, Kd_t),weight=-0.5) #KRIG
        network.connect(relay, cortical, (Kd_s, Kd_t), weight=1)  #
        network.connect(cortical, relay, (Kg_s2, Kd_t), weight=d)  # KexRCR

        for i, k_d in enumerate(spatial_angular_freqs):
            # create stimulus
            # stimulus = pylgn.stimulus.create_natural_image('C:/Users/John/Documents/UNI/Yr_3_Neuro-Project/circle.jpg', delay=20, duration=50)
            stimulus = pylgn.stimulus.create_fullfield_grating_ft(wavenumber=k_d,
                                                                  contrast=1)  # create_fullfield_grating_ft(wavenumber=k_d, contrast=1)
            network.set_stimulus(stimulus)  # ,compute_fft=True

            # compute
            network.compute_response(relay, recompute_ft=True)
            response[i, j] = relay.center_response[0]
            # response[i,j] = np.sum(relay.response[1])
            # response[i,j] = np.sum(relay.response)#-(256*256*relay.center_response[i])

            # imgplot = plt.imshow(relay.response[i])
            # plt.savefig('fig_nr=10.jpg')
            # plt.show()

    # visualize Relay cell response
    print(X.mouse)
    print(X.ID)
    for d, R in zip(feedback_strength, response.T):
        plt.plot(spatial_angular_freqs, R, '-o', label="feedback strength={}".format(d))
    plt.xlabel("Wavenumber (1/deg)")
    plt.ylabel("Relay Response")
    plt.legend()
    plt.show()
    cell_dict = {'Mouse': X.mouse, 'ID': X.ID, 'response': response}
    # cell_responses.append(cell_dict)
    for key, val in cell_dict.items():
        w.writerow([key, val])
    '''

    # visulize
    print(X.mouse)
    print(X.ID)
    pylgn.plot.animate_cube(relay.response,
                            title="Relay cell responses",
                            dt=integrator.dt.rescale("ms"),save_anim=True, filename="anim.gif")'''