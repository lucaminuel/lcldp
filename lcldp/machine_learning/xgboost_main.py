# -*- coding: utf-8 -*-
#pylint: disable=unused-wildcard-import
#pylint: disable=wildcard-import
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
In this module we use XGBoost Classifier tecnique.
We will try before without optimazing the hyperparameter, than we will do it to improve
our results.
"""
import time
import pandas as pd
from xgboost_tool import *
PATH = r'../data_analysis/data_analyzed.csv'


if __name__ == '__main__':
    start = time.time()
    print('Importing dataset...')
    data = pd.read_csv(PATH)
    print('Dataset imported!')
    #basic without rebalance
    print('Start basic XGBoost without rebalance')
    basic_xgboost(data , rebalance = 0, save = 1)
    #basic with rebalance
    print('Start basic XGBoost with rebalance')
    basic_xgboost(data , rebalance = 1, save = 1)
    #hyper without rebalance
    print('Start hyper XGBoost without rebalance')
    hyper_xgboost(data , rebalance = 0, save = 1)
    #hyper with rebalance
    print('Start hyper XGBoost with rebalance')
    hyper_xgboost(data , rebalance = 1, save = 1)
    mins = (time.time()-start)//60
    sec = (time.time()-start) % 60
    print(f"Elapsed time: {mins} min {sec:.2f} sec\n")
    