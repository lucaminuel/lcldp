# -*- coding: utf-8 -*-
#pylint: disable=line-too-long
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
In this module we use Neural Network tecnique to dataset analyzed in data_analysis modules
"""
import time
import pandas as pd
from neural_network_tool import basic_neural_network, test_neural_network, dropout_neural_network, diamond_neural_network, hyper_neural_network
PATH = r'../data_analysis/data_analyzed.csv'



if __name__ == '__main__':
    start = time.time()
    print('Importing dataset...')
    data = pd.read_csv(PATH)
    print('Dataset imported!')
    print('Starting basic neural network...')
    basic_neural_network(data)
    print('Testing overfitting...')
    test_neural_network(data)
    print('Starting dropout...')
    dropout_neural_network(data)
    print('Starting diamond neural network...')
    diamond_neural_network(data)
    print('Starting hyperparameter neural network...')
    hyper_neural_network(data)
    mins = (time.time()-start)//60
    sec = (time.time()-start) % 60
    print(f"Elapsed time: {mins} min {sec:.2f} sec\n")
