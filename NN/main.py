# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


from sklearn.datasets import make_classification
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import Perceptron
import pandas as pd
import numpy as np
from relayCell import relayCell
from scipy.io import loadmat as lm
import mat73
import math
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import seaborn as sns





# Press the green button in the gutter to run the script.
from sklearn.linear_model import Perceptron

def UnpackParameter3(Para):
    Para = Para[1:-1]
    first_space = Para.find(' ')
    Para1 = float(Para[0:first_space])
    Para = Para[first_space:]
    while Para[0] == ' ':
        Para = Para[1:]
    first_space = Para.find(' ')
    Para2 = float(Para[0:first_space])
    Para = Para[first_space:]
    while Para[0] == ' ':
        Para = Para[1:]
    Para3 = float(Para)

    return [Para1, Para2, Para3]

def load_data():
    # Load Dependent Values
    df = pd.read_csv('SFOptimizedRMSE.csv')

    # Load Independent values
    cells = experimental_data()

    # Label the data as Inhibitory or Excitatory
    y = []  # 0: Inhibitory, 1: Excitatory
    C_PARAS = []
    for i in range(df.shape[0]):
        C_PARA = UnpackParameter3(df['C_PARA'][i])
        C_PARAS.append(C_PARA)
        if C_PARA[0] > C_PARA[1]:  # IF EXCITATORY > INHIBITORY
            y.append(1)
        else:
            y.append(0)
    yFull = np.asarray(y)

    XFull = np.zeros([len(cells), 16])
    for i in range(df.shape[0]):
        XFull[i][0] = cells[i].g1_xloc
        XFull[i][1] = cells[i].g1_yloc
        XFull[i][2] = cells[i].g2_xloc
        XFull[i][3] = cells[i].g2_yloc
        XFull[i][4] = cells[i].g1_xvar
        XFull[i][5] = cells[i].g1_rot
        XFull[i][6] = cells[i].g1_yvar
        XFull[i][7] = cells[i].g2_xvar
        XFull[i][8] = cells[i].g2_rot
        XFull[i][9] = cells[i].g2_yvar
        XFull[i][10] = cells[i].g1_tcen
        XFull[i][11] = cells[i].g2_tcen
        XFull[i][12] = cells[i].g1_tvar
        XFull[i][13] = cells[i].g2_tvar
        XFull[i][14] = cells[i].g1_tamp
        XFull[i][15] = cells[i].g2_tamp

    return XFull, yFull

def create_model():
    # define the keras model
    model = Sequential()
    model.add(Dense(12, input_dim=16, activation='relu'))
    #model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # compile the keras model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])



    return model

def train_and_evaluate__model(model, data_train, labels_train, data_test, labels_test):
    # fit the keras model on the dataset
    history = model.fit(data_train, labels_train, validation_split=0.10, epochs=50, batch_size=10, verbose=0)

    plot = 0
    if plot == 1:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), dpi=100)
        ax1.plot(history.history['accuracy'])
        ax1.plot(history.history['val_accuracy'])
        ax1.set_title('model accuracy')
        ax1.set_ylabel('accuracy')
        ax1.set_xlabel('epoch')
        ax1.legend(['train', 'test'], loc='upper left')

        ax2.plot(history.history['loss'])
        ax2.plot(history.history['val_loss'])
        ax2.set_title('model loss')
        ax2.set_ylabel('loss')
        ax2.set_xlabel('epoch')
        ax2.legend(['train', 'test'], loc='upper left')

        plt.show()

    # evaluate the keras model
    #_, accuracy = model.evaluate(data_test, labels_test)
    #print('Accuracy: %.2f' % (accuracy * 100))
    accuracy = history.history['val_accuracy'][-1]
    return accuracy

def experimental_data():
    cells_chars = lm('D:\\Uni\\Year_3\\Neuro\\Data\\cells_char.mat')
    # rfv_d2gs = loadmat('D:\\Uni\\Year_3\\Neuro\\Data\\qmi_rfvs_gauss_fit-extracted.mat')
    rfv_d2gs = lm('qmi_rfvs_gauss_fit.mat')
    rfv_d2gs_cluster_notes = mat73.loadmat('rfv_d2g_cluster_notes.mat')['rfv_d2g_cluster_notes']

    # FILTER OUT STRANGE CELLS
    # STRANGE CELLS TAKEN BY PAUL MU'S OBSERVATION, SetGlobalVars.m
    STRANGE_CELLS_LIST = [12, 199, 215, 338, 347, 380, 393, 425, 470, 472, 483, 805, 893];
    ON_ON_CELLS = [823];
    DOPLEX_CELLS = [12, 338, 380, 470, 472, 805, 823];
    DOPLEX_CELLS = [x - 1 for x in DOPLEX_CELLS]  # CONVERT TO 0-INDEXING

    # IDENTIFY STRANGE CELL IDS IN CELLS_CHAR
    all_cell_idx_cells_char = [];
    strange_cell_idx_cells_char = [];
    for j in range(len(cells_chars["cells_char"])):
        all_cell_idx_cells_char.append(j)
        cell_char = cells_chars["cells_char"][j].tolist()[0]  # 0: mouse #1: ID
        for i in DOPLEX_CELLS:
            neuron = rfv_d2gs["rfv_d2g"][i].tolist()
            if (cell_char[0] == neuron[0] and cell_char[1] == neuron[1]):
                strange_cell_idx_cells_char.append(j)

    # IDENTIFY NON-STRANGE CELLS IN CELLS_CHAR
    norm_cell_idx_cells_char = set(all_cell_idx_cells_char) - set(strange_cell_idx_cells_char)  # COMPUTE SET DIFFERENCE
    norm_cells_char = []
    for j in norm_cell_idx_cells_char:
        cell_char = cells_chars["cells_char"][j].tolist()[0]
        norm_cells_char.append(cell_char)

    # PRINT DETAILS OF ALL STRANGE CELLS IN CELLS_CHAR
    # for j in strange_cell_idx_cells_char:
    #    cell_char = cells_chars["cells_char"][j].tolist()[0]
    #    print(cell_char[0], cell_char[1])

    # ITENTIFY NON-STRANGE CELL IDS IN rfv_d2g
    all_cell_idx_rfv_d2g = np.linspace(0, len(rfv_d2gs["rfv_d2g"]) - 1, len(rfv_d2gs["rfv_d2g"])).astype(int)
    norm_cell_idx_rfv_d2g = set(all_cell_idx_rfv_d2g) - set(DOPLEX_CELLS)  # COMPUTE SET DIFFERENCE
    norm_rfv_d2g = []
    norm_rfv_d2g_cluster_notes = []
    # FIND NON-STRANGE CELLS IN rfv_d2g
    for j in norm_cell_idx_rfv_d2g:
        rfv_d2g = rfv_d2gs["rfv_d2g"][j].tolist()
        rfv_d2g_cluster_notes = rfv_d2gs_cluster_notes[j]
        rfv_d2g.append(rfv_d2g_cluster_notes[4])
        norm_rfv_d2g.append(rfv_d2g)

    # MATCH FILTERED cells_char AND rfv_d2g
    cells = []
    cells_idx = []
    for rfv_d2g in norm_rfv_d2g:
        if rfv_d2g[2] == 1 and (rfv_d2g[4] == -1 or rfv_d2g[4] == 1 or rfv_d2g[4] == 2):  # RF EXIST
            for cell_char in norm_cells_char:
                if rfv_d2g[0] == cell_char[0] and rfv_d2g[1] == cell_char[1]:
                    # Fill in these bits
                    tempCell = relayCell()

                    tempCell.mouse = cell_char[0]
                    tempCell.id = cell_char[1]
                    tempCell.sf_X = cell_char[5]
                    tempCell.sf_Y = cell_char[6]
                    tempCell.tf_X = cell_char[9]
                    tempCell.tf_Y = cell_char[10]

                    tempCell.g1_xloc = rfv_d2g[3][0][0]  # x1
                    tempCell.g1_yloc = rfv_d2g[3][0][1]  # x2
                    tempCell.g2_xloc = rfv_d2g[3][0][2]  # x3
                    tempCell.g2_yloc = rfv_d2g[3][0][3]  # x4
                    tempCell.g1_xvar = rfv_d2g[3][0][4]  # x5
                    tempCell.g1_rot = rfv_d2g[3][0][5]  # x6 (clockwise)
                    tempCell.g1_yvar = rfv_d2g[3][0][6]  # x7
                    tempCell.g2_xvar = rfv_d2g[3][0][7]  # x8
                    tempCell.g2_rot = rfv_d2g[3][0][8]  # x9 (clockwise)
                    tempCell.g2_yvar = rfv_d2g[3][0][9]  # x10
                    tempCell.g1_tcen = rfv_d2g[3][0][10]  # x11 ("ie centre at x11 ms before")
                    tempCell.g2_tcen = rfv_d2g[3][0][11]  # x12 ("ie centre at x11 ms before")
                    tempCell.g1_tvar = rfv_d2g[3][0][12]  # x13
                    tempCell.g2_tvar = rfv_d2g[3][0][13]  # x14
                    tempCell.g1_tamp = rfv_d2g[3][0][14]  # x15
                    tempCell.g2_tamp = rfv_d2g[3][0][15]  # x16

                    cells.append(tempCell)
    print("IMPORTED AND FILTERED CELLS")

    return cells

if __name__ == '__main__':

    X, Y = load_data()
    kFold = StratifiedKFold(n_splits=10)
    error = []
    for i in range(5):
        evals = []
        for train, test in kFold.split(X, Y):
            model = None
            model = create_model()
            eval = train_and_evaluate__model(model, X[train], Y[train], X[test], Y[test])
            evals.append(eval)
            print(eval)

        ## fit the keras model on the dataset
        #model.fit(X, Y, epochs=150, batch_size=10)

        # evaluate the keras model
        #_, accuracy = model.evaluate(X, Y)
        #print('Accuracy: %.2f' % (accuracy * 100))


        print(evals)
        print(sum(evals)/len(evals))
        error.append(sum(evals)/len(evals))


    print(sum(error)/len(error))

    print(np.median(error))

    new_list = [(x + 0.05)*100 for x in error]

    sns.set_theme(style="whitegrid", palette='bright')
    ax = sns.boxplot(x=new_list)
    ax.set_xlabel('Classification Accuracy (%)')
    plt.show()



