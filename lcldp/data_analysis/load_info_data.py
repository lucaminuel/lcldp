# -*- coding: utf-8 -*-
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
The main of this module is loading our datafrane and print its info
"""
import time
import pandas as pd



def load_info_data(data_path):
    '''
    This function read a dataframe csv from path and print its info

    Parameters
    ----------
    data_path : String
        dataframe which we want info .

    Returns
    -------
    data: Dataframe

    '''
    start = time.time()
    data = pd.read_csv(data_path)
    print('Loading data...\n')
    mins = (time.time()-start)//60
    sec = (time.time()-start) % 60
    data.info()
    print(f"Time to load the dataset is: {mins} min {sec:.2f} sec\n")
    return data
