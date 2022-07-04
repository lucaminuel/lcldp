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
In this second module of data analysis we prepare our dataset for machine learning tools.
Change parameter "SAVE" if you do not want to save the results
"""
import time
import numpy as np
import pandas as pd
from load_info_data import load_info_data
PATH=  r'..\lending_club_loan_two.csv'
SAVE = False


def fill_in_mort_acc(total_acc, mort_acc):
    '''
    This function replace Nan values of feature mort_acc with mean value of total_acc

    Parameters
    ----------
    total_acc : pandas.core.series.Series
        total_acc column of our dataset.
    mort_acc : pandas.core.series.Series
        mort_acc column of our dataset

    Returns
    -------
    mort_acc : pandas.core.series.Series
        new mort_acc with Nan values replaced.

    '''
    if np.isnan(mort_acc):
        return total_acc_avg[total_acc]
    return mort_acc


if __name__ == '__main__':
    #Import dataset and print it's info
    start = time.time()
    data = load_info_data(PATH)
    dropped_feature = []
    #First of all deal with missing values. The number of missing values in each feature
    #will determine the treatment of the feature: drop or replace with other values.
    #empt_title is the column with most NaN values, but it has too many unique values,
    #it is not feasible to keep them as feature for ML model. we will just drop it
    print(f'Missing value: \n {data.isnull().sum()/len(data)*100}')
    print(f'Number employment title feature: {data["emp_title"].nunique()}')
    dropped_feature.append('emp_length')
    #From previous module we showed how "Verified" and "Source Verified"
    #don't help us to destinguish Fully Paid from Charged Off'.
    #Same for employment length feature, we can drop it
    #Replace Verified and Source Verified to Verified
    data["verification_status"] = data["verification_status"]\
        .replace(["Verified", "Source Verified"], "Verified")
    dropped_feature.append('emp_title')
    #From previous module, number of charged off and total borrowers in percantage in each
    #intervals of employment length are relatively the same across, we can drop it
    #Delete NaN from emp_length and sort it
    sorted(data["emp_length"].dropna().unique())
    #It appears 'title' provides the same information as loan purprose
    #we can drop this feature.
    dropped_feature.append('title')
    #Recalling that mortage account feature has ~38000 missing values.
    #This is pretty significant as dropping this feature will significantly reduce
    #the size of the dataset. It is probably a good idea to replace the
    #missing values with some other values.
    #From correlation of mortage with other feature, total account has the highest
    #correlation with mortage account.
    #We will replace it eith neab value based on total account
    print(f'Unique values of Number of mortgage accounts: \
          \n{data["mort_acc"].value_counts()}')
    print(f'Correlation Matrix of Mortage account features: \n \
          {data.corr()["mort_acc"].sort_values()}')
    total_acc_avg = data.groupby("total_acc").mean()["mort_acc"]
    #We use lambda function to replace it
    data["mort_acc"] = data.apply(lambda x: fill_in_mort_acc\
                                  (x["total_acc"], x["mort_acc"]), axis=1)
    #We can remove the last Nan values
    data = data.dropna()
    print(f'Nan Values: \n {data.isnull().sum()}')
    #Make Fully Paid = 1, Chargef Off = 0
    data["loan_repaid"] = data["loan_status"].map({"Fully Paid":1, "Charged Off":0})
    dropped_feature.append('loan_status')
    #Deal with non-numeric type of data
    print(f'non-numeric type of data: \n \
          {data.select_dtypes(["object"]).columns}')
    #Term values has only "36 months" and "60" months, we'll grabb only numeric values
    #We use lambda function to do it
    data["term"] = data["term"].apply(lambda term: int(term[:3]))
    #Since sub-grade provides more information than grade, this featue will be dropped.
    dropped_feature.append('grade')
    #Preparing the data with binary classification (dummy data).
    dummy = pd.get_dummies(data["sub_grade"], drop_first=True)
    data = pd.concat([data.drop("sub_grade", axis=1), dummy], axis=1)
    dummy = pd.get_dummies(data[["verification_status", "application_type", \
                                 "initial_list_status", "purpose"]], drop_first=True)
    data = pd.concat([data.drop(["verification_status", "application_type",\
                                 "initial_list_status", "purpose"], axis=1), dummy], axis=1)
    #To simplify, we can set home_ownership just as: None, Any and Other
    print(f'Home Ownership feature : \n {data["home_ownership"].value_counts()}')
    data["home_ownership"] = data["home_ownership"].replace(["NONE", "ANY"], "OTHER")
    dummy = pd.get_dummies(data["home_ownership"], drop_first=True)
    data = pd.concat([data.drop("home_ownership", axis=1), dummy], axis=1)
    #For address ZIP code may have some sort of influence in the outcome. We grab ZIP code
    #from the address
    print(f'Address: \n {data["address"].value_counts()}')
    #We use lambda function to grab only the zip
    data["zip_code"] = data["address"].apply(lambda address: address[-5:])
    dummy = pd.get_dummies(data["zip_code"], drop_first=True)
    data = pd.concat([data.drop("zip_code", axis=1), dummy], axis=1)
    dropped_feature.append('address')
    dropped_feature.append('issue_d')
    #The feature "Earliest credit line" may be a key factor as
    #it provides some sort of a time series information.
    #Grabbing the year as our time series feature.
    print(f' Earliste credit line: \n {data["earliest_cr_line"].value_counts()}')
    data["earliest_cr_line"] = data["earliest_cr_line"].apply(lambda year: int(year[-4:]))
    #Let'save our dataset
    data = data.drop(dropped_feature, axis=1)
    if SAVE is True:
        data.to_csv('data_analyzed.csv',index = False)
    mins = (time.time()-start)//60
    sec = (time.time()-start) % 60
    data.info()
    print(f"Elapsed time: {mins} min {sec:.2f} sec\n")
    