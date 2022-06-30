# -*- coding: utf-8 -*-
#pylint: disable=line-too-long
#pylint: disable=invalid-name
#pylint: disable=no-name-in-module
#pylint: disable=import-error
#pylint: disable=too-many-locals
#pylint: disable=too-many-statements
#pylint: disable=unused-variable
#pylint: disable=expression-not-assigned
# Copyright 2022 Manuel Luci-Andrea Dell'Abate
#
# This file is part of CMEPDA Project: Lending Club loan data prediction
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>.

"""
In this module we build our Neural newtwork model and algorithm:
1)Basic Model
2)Dropout Model
3)Diamond Model
4) Hyperparameter tuning model
The last one can require a lot of time (in our simulation it needed more than 2 hours).
"""
import os
from collections import Counter
from imblearn.over_sampling import SMOTE
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from keras.models import load_model
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn import metrics
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


SEED = 42


def build_model(X_res):
    '''
    Basic model Neural Network

    Parameters
    ----------
    X_res : array
        rebalanced X_train

    Returns
    -------
    model :
        model.

    '''
    n_feature = X_res.shape[1]
    model = Sequential()
    model.add(Dense(128, input_dim=n_feature, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    return model


def build_DROPOUT_model(X_train1):
    '''
    Basic Neural Network with regularization Dropout

    Parameters
    ----------
    X_train1: array
        X_train1.

    Returns
    -------
    model :
        model.

    '''
    # define the model
    model = Sequential()
    n_feature = X_train1.shape[1]
    model.add(Dense(128, activation='relu', input_shape=(n_feature,)))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    # linear activation
    model.add(Dense(1))
    # compile the model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def build_model_diamond(X_train1):
    '''
    Diamond Neural Network

    Parameters
    ----------
    X_train1 : array
        X_train1.

    Returns
    -------
    model :
        model.

    '''
    n_feature = X_train1.shape[1]
    model = Sequential()
    model.add(Dense(128, input_dim=n_feature, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, input_dim=n_feature, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    return model



def basic_neural_network(data):
    '''
    This is basic form of neural netowork algorithm

    Parameters
    ----------
    data : DataFrame
        dataframe you want to use the tecnique.

    Returns
    -------
    None.

    '''
    X = data.drop("loan_repaid", axis=1).values
    y = data["loan_repaid"].values
    #splitting into training and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)
    #count how many values of y there are in training and test
    print(f'y_train: {np.unique(y_train, return_counts=True)}')
    print(f'y_test: {np.unique(y_test, return_counts=True)}')
    #rebalance the minority class
    print('Rebalnce data...')
    sm = SMOTE(random_state= SEED)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print(f'Resampled dataset shape {Counter(y_res)}')
    #Scaling
    scaler = MinMaxScaler()
    scaler.fit(X_res)
    X_res = scaler.transform(X_res)
    X_test = scaler.transform(X_test)
    #X_res.shape
    print('Starting Neural Network first model...')
    model1 = build_model(X_res)
    history1 = model1.fit(X_res, y_res, epochs = 50, batch_size = 64).history
    #fitting the model2 with a certain batch size
    print('Starting Neural Network second model...')
    model2 = build_model(X_res)
    history2 = model2.fit(X_res, y_res, epochs = 50, batch_size = 512).history
    print('Saving results...')
    #plot and save in folder Figure
    #Create folder Figure if it does not exist
    if os.path.isdir('Figures') is False:
        print('Creating Folder Figures...\n')
        os.mkdir('Figures')
    script_dir = os.path.dirname(__file__)
    results_dir = os.path.join(script_dir, 'Figures/')
    plt.plot(history1['loss'], label='Loss 64')
    plt.plot(history2['loss'], label='Loss 512')
    plt.xlabel('Epochs')
    plt.ylabel('Cross-Entropy')
    plt.legend()
    name = '(5)basic_neural_network.pdf'
    plt.savefig(results_dir + name, format="pdf", bbox_inches="tight")
    plt.show()
    #Prinitng accuracy
    test_loss_1, test_acc_1 = model1.evaluate(X_test, y_test)
    test_loss_2, test_acc_2 = model2.evaluate(X_test, y_test)
    print(f'Loss {test_loss_1}, Accuracy {test_acc_1}')
    print(f'Loss {test_loss_2}, Accuracy {test_acc_2}')
    #predict the array of results
    y_pred1 = (model1.predict(X_test) > 0.5).astype("int32")
    y_pred2 = (model2.predict(X_test) > 0.5).astype("int32")
    #Printing some metrics and save them in txt file
    #Save txt in  folder Results
    if os.path.isdir('Results') is False:
        print('Creating Folder Results...\n')
        os.mkdir('Results')
    path_results = './Results'
    out_name = 'basic_neural_network.txt'
    file_path = os.path.join(path_results, out_name)
    print("Accuracy NN:", metrics.accuracy_score(y_test, y_pred1))
    print("Precision NN:", metrics.precision_score(y_test, y_pred1))
    print("Recall NN:", metrics.recall_score(y_test, y_pred1))
    print("F1 Score NN:", metrics.f1_score(y_test, y_pred1))
    print("Accuracy NN:", metrics.accuracy_score(y_test, y_pred2))
    print("Precision NN:", metrics.precision_score(y_test, y_pred2))
    print("Recall NN:", metrics.recall_score(y_test, y_pred2))
    print("F1 Score NN:", metrics.f1_score(y_test, y_pred2))
    with open(file_path, 'w',encoding='utf-8') as file:
        print('#Epochs = 50 Batch size = 64:',file = file)
        print(f"Accuracy NN: {metrics.accuracy_score(y_test, y_pred1)}", file = file)
        print(f"Precision NN: {metrics.precision_score(y_test, y_pred1)}", file = file)
        print(f"Recall NN: {metrics.recall_score(y_test, y_pred1)}", file = file )
        print(f"F1 Score NN: {metrics.f1_score(y_test, y_pred1)}", file = file )
        print('#Epochs = 50 Batch size = 512:',file = file)
        print(f"Accuracy NN: {metrics.accuracy_score(y_test, y_pred2)}", file = file)
        print(f"Precision NN: {metrics.precision_score(y_test, y_pred2)}", file = file)
        print(f"Recall NN: {metrics.recall_score(y_test, y_pred2)}", file = file )
        print(f"F1 Score NN: {metrics.f1_score(y_test, y_pred2)}", file = file )
    #Plotting  and save the confusion matrix
    cm = confusion_matrix(y_test, y_pred1)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=['Charged Off', 'Fully Paid'])
    disp.plot()
    name = '(6)confusion_matrix_basic_neural_network.pdf'
    plt.savefig(results_dir + name, format="pdf", bbox_inches="tight")
    plt.show()


def test_neural_network(data):
    '''
    This tool analyzes overfitting of our neural network

    Parameters
    ----------
    data : DataFrame
        dataframe you want to use the tecnique.

    Returns
    -------
    None.

    '''
    X = data.drop("loan_repaid", axis=1).values
    y = data["loan_repaid"].values
    #splitting into training and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)
    #count how many values of y there are in training and test
    print(f'y_train: {np.unique(y_train, return_counts=True)}')
    print(f'y_test: {np.unique(y_test, return_counts=True)}')
    #rebalance the minority class
    print('Rebalnce data...')
    sm = SMOTE(random_state= SEED)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print(f'Resampled dataset shape {Counter(y_res)}')
    #Scaling
    scaler = MinMaxScaler()
    scaler.fit(X_res)
    X_res = scaler.transform(X_res)
    X_test = scaler.transform(X_test)
    #X_res.shape
    model3 = build_model(X_res)
    history3 = model3.fit(X_res, y_res, validation_data=(X_test, y_test),
                          epochs = 800, batch_size = 4096).history
    #plot and save in folder Figure
    #Create folder Figure if it does not exist
    if os.path.isdir('Figures') is False:
        print('Creating Folder Figures...\n')
        os.mkdir('Figures')
    script_dir = os.path.dirname(__file__)
    results_dir = os.path.join(script_dir, 'Figures/')
    plt.plot(history3['loss'], label='Train')
    plt.plot(history3['val_loss'], label='Val')
    plt.xlabel('Epochs')
    plt.ylabel('Cross-Entropy')
    plt.legend()
    name = '(7)overfitting_neural_network.pdf'
    plt.savefig(results_dir + name, format="pdf", bbox_inches="tight")
    plt.show()
    test_loss_3, test_acc_3 = model3.evaluate(X_test, y_test)
    print(f'Loss {test_loss_3}, Accuracy {test_acc_3}')
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    mc = tf.keras.callbacks.ModelCheckpoint('best_model_NOREG.h5',
                                            monitor='val_loss', save_best_only=True)
    #splitting into training and validition set
    X_train1, X_val, y_train1, y_val = train_test_split(X_res, y_res, test_size=0.3)
    print('Testing overfitting...')
    model4 = build_model(X_res)
    history4 = model4.fit(X_train1, y_train1, validation_data=(X_val, y_val), epochs = 1000,
               batch_size = 512, callbacks=[es, mc]).history
    test_loss_4, test_acc_4 = model4.evaluate(X_test, y_test)
    print(f'Loss {test_loss_4}, Accuracy {test_acc_4}')
    #printing some metrics and save them in txt file
    y_pred4 = (model4.predict(X_test) > 0.5).astype("int32")
    #Save txt in  folder Results
    if os.path.isdir('Results') is False:
        print('Creating Folder Results...\n')
        os.mkdir('Results')
    path_results = './Results'
    out_name = 'basic_neural_network.txt'
    file_path = os.path.join(path_results, out_name)
    print("Accuracy NN: {metrics.accuracy_score(y_test, y_pred4)}")
    print("Precision NN: {metrics.precision_score(y_test, y_pred4)}")
    print("Recall NN: {metrics.recall_score(y_test, y_pred4)}")
    print("F1 Score NN: { metrics.f1_score(y_test, y_pred4)}")
    with open(file_path, 'w',encoding='utf-8') as file:
        print(f"Accuracy NN: {metrics.accuracy_score(y_test, y_pred4)}", file = file)
        print(f"Precision NN: {metrics.precision_score(y_test, y_pred4)}", file = file)
        print(f"Recall NN: {metrics.recall_score(y_test, y_pred4)}", file = file )
        print(f"F1 Score NN: {metrics.f1_score(y_test, y_pred4)}", file = file )
    unique, counts = np.unique(y_pred4, return_counts=True)
    result = np.column_stack((unique, counts))
    print(result)
    cm = confusion_matrix(y_test, y_pred4)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=['Charged Off', 'Fully Paid'])
    disp.plot()
    name = '(8)test_confusion_matrix_basic_neural_network.pdf'
    plt.savefig(results_dir + name, format="pdf", bbox_inches="tight")
    plt.show()



def dropout_neural_network(data):
    '''
    This tool use dropout regolarization in our neural_network

    Parameters
    ----------
    data : DataFrame
        dataframe you want to use the tecnique.

    Returns
    -------
    None.

    '''
    X = data.drop("loan_repaid", axis=1).values
    y = data["loan_repaid"].values
    #splitting into training and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)
    #count how many values of y there are in training and test
    print(f'y_train: {np.unique(y_train, return_counts=True)}')
    print(f'y_test: {np.unique(y_test, return_counts=True)}')
    #rebalance the minority class
    print('Rebalnce data...')
    sm = SMOTE(random_state= SEED)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print(f'Resampled dataset shape {Counter(y_res)}')
    #Scaling
    scaler = MinMaxScaler()
    scaler.fit(X_res)
    X_res = scaler.transform(X_res)
    X_test = scaler.transform(X_test)
    #X_res.shape
    X_train1, X_val, y_train1, y_val = train_test_split(X_res, y_res, test_size=0.3)
    mc = tf.keras.callbacks.ModelCheckpoint('best_model_DROPOUT.h5', monitor='val_loss', save_best_only=True)
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    print('Starting dropout...')
    DROPOUT_model = build_DROPOUT_model(X_train1)
    h_DROPOUT = DROPOUT_model.fit(X_train1, y_train1, validation_data=(X_val, y_val),
                      epochs = 100, batch_size = 128, callbacks=[es, mc]).history
    ##laod best models and test them
    best_NOREG_model = load_model('best_model_NOREG.h5')
    best_DROPOUT_model = load_model('best_model_DROPOUT.h5')
    loss_NOREG, acc_NOREG = best_NOREG_model.evaluate(X_test, y_test)
    loss_DROPOUT, acc_DROPOUT = best_DROPOUT_model.evaluate(X_test, y_test)
    print(f'Loss {loss_NOREG}, Accuracy {acc_NOREG}')
    print(f'Loss {loss_DROPOUT}, Accuracy {acc_DROPOUT}')
    y_pred_DROPOUT = (DROPOUT_model.predict(X_test) > 0.5).astype("int32")
    #printing some metrics and save them in txt file
    #Save txt in  folder Results
    if os.path.isdir('Results') is False:
        print('Creating Folder Results...\n')
        os.mkdir('Results')
    path_results = './Results'
    out_name = 'dropout_neural_network.txt'
    file_path = os.path.join(path_results, out_name)
    print(f"Accuracy NN: {metrics.accuracy_score(y_test, y_pred_DROPOUT)}")
    print(f"Precision NN: {metrics.precision_score(y_test, y_pred_DROPOUT)}")
    print(f"Recall NN: {metrics.recall_score(y_test, y_pred_DROPOUT)}")
    print(f"F1 Score NN: {metrics.f1_score(y_test, y_pred_DROPOUT)}")
    print('Saving results...')
    with open(file_path, 'w',encoding='utf-8') as file:
        print(f"Accuracy NN: {metrics.accuracy_score(y_test, y_pred_DROPOUT)}", file = file)
        print(f"Precision NN: {metrics.precision_score(y_test, y_pred_DROPOUT)}", file = file)
        print(f"Recall NN: {metrics.recall_score(y_test, y_pred_DROPOUT)}", file = file)
        print(f"F1 Score NN: {metrics.f1_score(y_test, y_pred_DROPOUT)}", file = file)
    #plot and save in folder Figure
    #Create folder Figure if it does not exist
    if os.path.isdir('Figures') is False:
        print('Creating Folder Figures...\n')
        os.mkdir('Figures')
    script_dir = os.path.dirname(__file__)
    results_dir = os.path.join(script_dir, 'Figures/')
    cm = confusion_matrix(y_test, y_pred_DROPOUT)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=['Charged Off', 'Fully Paid'])
    disp.plot()
    name = '(9)confusion_matrix_dropout_neural_network.pdf'
    plt.savefig(results_dir + name, format="pdf", bbox_inches="tight")
    plt.show()


def diamond_neural_network(data):
    '''
    This tool use dropout regolarization in our  diamond neural_network

    Parameters
    ----------
    data : DataFrame
        dataframe you want to use the tecnique.

    Returns
    -------
    None.

    '''
    X = data.drop("loan_repaid", axis=1).values
    y = data["loan_repaid"].values
    #splitting into training and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)
    #count how many values of y there are in training and test
    print(f'y_train: {np.unique(y_train, return_counts=True)}')
    print(f'y_test: {np.unique(y_test, return_counts=True)}')
    #rebalance the minority class
    print('Rebalnce data...')
    sm = SMOTE(random_state= SEED)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print(f'Resampled dataset shape {Counter(y_res)}')
    #Scaling
    scaler = MinMaxScaler()
    scaler.fit(X_res)
    X_res = scaler.transform(X_res)
    X_test = scaler.transform(X_test)
    #X_res.shape
    X_train1, X_val, y_train1, y_val = train_test_split(X_res, y_res, test_size=0.3)
    DROPOUT_model_diamond = build_model_diamond(X_train1)
    mc = tf.keras.callbacks.ModelCheckpoint('best_model__diamond_DROPOUT.h5', monitor='val_loss', save_best_only=True)
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    print('Starting diamond neural network...')
    h_DROPOUT = DROPOUT_model_diamond.fit(X_train1, y_train1, validation_data=(X_val, y_val),
                              epochs = 100, batch_size = 512, callbacks=[es, mc]).history
    test_loss_5, test_acc_5 = DROPOUT_model_diamond.evaluate(X_test, y_test)
    print('Loss {test_loss_5}, Accuracy {test_acc_5}')
    y_pred5 = (DROPOUT_model_diamond.predict(X_test) > 0.5).astype("int32")
    #printing some metrics and save them in txt file
    #Save txt in  folder Results
    if os.path.isdir('Results') is False:
        print('Creating Folder Results...\n')
        os.mkdir('Results')
    path_results = './Results'
    out_name = 'dropout_diamond_neural_network.txt'
    file_path = os.path.join(path_results, out_name)
    print(f"Accuracy NN: {metrics.accuracy_score(y_test, y_pred5)}")
    print(f"Precision NN:{metrics.precision_score(y_test, y_pred5)}")
    print(f"Recall NN: {metrics.recall_score(y_test, y_pred5)}")
    print(f"F1 Score NN: {metrics.f1_score(y_test, y_pred5)}")
    print('Saving results...')
    with open(file_path, 'w',encoding='utf-8') as file:
        print(f"Accuracy NN: {metrics.accuracy_score(y_test, y_pred5)}", file = file)
        print(f"Precision NN:{metrics.precision_score(y_test, y_pred5)}", file = file)
        print(f"Recall NN: {metrics.recall_score(y_test, y_pred5)}", file = file)
        print(f"F1 Score NN: {metrics.f1_score(y_test, y_pred5)}", file = file)
    #plot and save in folder Figure
    #Create folder Figure if it does not exist
    if os.path.isdir('Figures') is False:
        print('Creating Folder Figures...\n')
        os.mkdir('Figures')
    script_dir = os.path.dirname(__file__)
    results_dir = os.path.join(script_dir, 'Figures/')
    cm = confusion_matrix(y_test, y_pred5)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=['Charged Off', 'Fully Paid'])
    disp.plot()
    name = '(10)confusion_matrix_dropout_diamond_neural_network.pdf'
    plt.savefig(results_dir + name, format="pdf", bbox_inches="tight")
    plt.show()


def hyper_neural_network(data):
    '''
    This tool use create and test a neural network with hyperparameters tuned parameter

    Parameters
    ----------
    data : DataFrame
        dataframe you want to use the tecnique.

    Returns
    -------
    None.

    '''
    X = data.drop("loan_repaid", axis=1).values
    y = data["loan_repaid"].values
    #splitting into training and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)
    #count how many values of y there are in training and test
    print(f'y_train: {np.unique(y_train, return_counts=True)}')
    print(f'y_test: {np.unique(y_test, return_counts=True)}')
    #rebalance the minority class
    print('Rebalnce data...')
    sm = SMOTE(random_state= SEED)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print(f'Resampled dataset shape {Counter(y_res)}')
    #Scaling
    scaler = MinMaxScaler()
    scaler.fit(X_res)
    X_res = scaler.transform(X_res)
    X_test = scaler.transform(X_test)
    #X_res.shape
    X_train1, X_val, y_train1, y_val = train_test_split(X_res, y_res, test_size=0.3)
    n_layers = [1, 2, 3, 4, 5]
    h_dim = [32, 64, 128]
    activation = ['relu', 'tanh', 'sigmoid']
    optimizer = ['adagrad', 'adam']
    params = dict(optimizer=optimizer, n_layers=n_layers, h_dim=h_dim, activation=activation)
    def build_model_opt(n_layers=3, h_dim=128, activation='relu', optimizer='adam'):
        '''
        Model neural network for hyperparamter

        Parameters
        ----------
        n_layers : int, optional
            number of layers. The default is 3.
        h_dim : int, optional
            number of node for layer. The default is 128.
        activation : string, optional
            activation function. The default is 'relu'.
        optimizer : string, optional
            optimizer. The default is 'adam'.

        Returns
        -------
        model :
            model.

        '''
        # define the model
        model = Sequential()
        n_feature = X_train1.shape[1]
        model.add(Dense(h_dim, activation=activation, input_shape=(n_feature,)))
        for i in range(n_layers - 1):
            model.add(Dense(h_dim, activation=activation))
            model.add(Dropout(0.2))
        # linear activation
        model.add(Dense(1))
        # compile the model
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model
    model = KerasRegressor(build_fn = build_model_opt)
    print('Finding best parameters...')
    rnd = RandomizedSearchCV(estimator=model, param_distributions=params, n_iter=5, cv=5)
    rnd_result = rnd.fit(X_train1, y_train1, epochs=200, batch_size=512, verbose=0)
    print(f"Best: {-rnd_result.best_score_} using {rnd_result.best_params_}")
    means = rnd_result.cv_results_['mean_test_score']
    stds = rnd_result.cv_results_['std_test_score']
    params = rnd_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print(f"{-mean} ({stdev}) with: {param}")
    clf = rnd_result.best_estimator_.model
    loss, acc = clf.evaluate(X_test, y_test)
    print(f'Loss {loss}, Accuracy {acc}')
    print(f'n_layers = {rnd_result.best_params_["n_layers"]}')
    print(f'h_dim = {rnd_result.best_params_["h_dim"]}')
    print(f'activation = {rnd_result.best_params_["activation"]}')
    print(f'optimizer = {rnd_result.best_params_["optimizer"]}) ')
    model_hpt = build_model_opt(
                    n_layers = rnd_result.best_params_['n_layers'],
                    h_dim = rnd_result.best_params_['h_dim'],
                    activation = rnd_result.best_params_['activation'],
                    optimizer = rnd_result.best_params_['optimizer'])
    mc = tf.keras.callbacks.ModelCheckpoint('best_model_hpt.h5', monitor='val_loss', save_best_only=True)
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    print('Starting hyperparameter neural network')
    h_hpt = model_hpt.fit(X_train1, y_train1, validation_data=(X_val, y_val),
                  epochs = 50, batch_size= 64, callbacks=[es, mc]).history
    test_loss_hpt, test_acc_hpt = model_hpt.evaluate(X_test, y_test)
    print(f'Loss {test_loss_hpt}, Accuracy {test_acc_hpt}')
    y_pred_hpt = (model_hpt.predict(X_test) > 0.5).astype("int32")
    #printing some metrics and save them in txt file
    #Save txt in  folder Results
    if os.path.isdir('Results') is False:
        print('Creating Folder Results...\n')
        os.mkdir('Results')
    path_results = './Results'
    out_name = 'hyper_neural_network.txt'
    file_path = os.path.join(path_results, out_name)
    print(f"Accuracy NN: {metrics.accuracy_score(y_test, y_pred_hpt)}")
    print(f"Precision NN: {metrics.precision_score(y_test, y_pred_hpt)}")
    print(f"Recall NN: {metrics.recall_score(y_test, y_pred_hpt)}")
    print(f"F1 Score NN: {metrics.f1_score(y_test, y_pred_hpt)}")
    with open(file_path, 'w',encoding='utf-8') as file:
        print(f"Accuracy NN: {metrics.accuracy_score(y_test, y_pred_hpt)}", file = file)
        print(f"Precision NN: {metrics.precision_score(y_test, y_pred_hpt)}", file = file)
        print(f"Recall NN: {metrics.recall_score(y_test, y_pred_hpt)}", file = file)
        print(f"F1 Score NN: {metrics.f1_score(y_test, y_pred_hpt)}", file = file)
    #plot and save in folder Figure
    #Create folder Figure if it does not exist
    if os.path.isdir('Figures') is False:
        print('Creating Folder Figures...\n')
        os.mkdir('Figures')
    script_dir = os.path.dirname(__file__)
    results_dir = os.path.join(script_dir, 'Figures/')
    cm = confusion_matrix(y_test, y_pred_hpt)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=['Charged Off', 'Fully Paid'])
    disp.plot()
    name = '(11)confusion_matrix_hyper_neural_network.pdf'
    plt.savefig(results_dir + name, format="pdf", bbox_inches="tight")
    plt.show()
    