#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import pandas as pd
import numpy as np
import itertools
import pickle

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['figure.dpi'] = 900
plt.rc('font',family='Times New Roman')

os.chdir('/ML data and analysis')

# =============================================================================
# Loading the files
# =============================================================================

### Recoms database
recoms = pd.read_csv("recoms_detail_1992_2020.csv", encoding='latin-1')
recoms_cols = ['TICKER', 'CUSIP', 'CNAME', 'OFTIC', 'AMASKCD', 'ACTDATS',
               'ESTIMID', 'ANALYST', 'EMASKCD', 'IRECCD', 'ITEXT',
               'ERECCD', 'ETEXT', 'USFIRM',
               'REVDATS', 'ANNDATS']
recoms = recoms[recoms_cols]

# Creating date variables
recoms['rev_year'] = recoms['ACTDATS'].astype(str).str.slice(stop=4).astype(int)

# Keeping the last annual recommendation per analyst
recoms = recoms.dropna(subset= ['ANALYST'])
recoms.sort_values(by=['CUSIP','AMASKCD','ACTDATS'], inplace=True)
recoms = recoms.groupby(['CUSIP', 'AMASKCD', 'rev_year']).tail(1)
recoms_cols = ['CUSIP', 'rev_year', 'AMASKCD', 'IRECCD']
recoms = recoms[recoms_cols]
recoms['IRECCD'] = 6 - recoms['IRECCD']

# =============================================================================
# Creating the structural coherence variable based on Zuckerman (2004)
# =============================================================================
# dict of all analyst recs by year
recoms_years = {}
years = sorted(list(recoms['rev_year'].drop_duplicates()))
for year in years:
    recoms_years[year] = recoms[recoms['rev_year']==year]
    
# transform to a list of firms covered by analyst
recoms_years_list = {}
for year, ay in recoms_years.items():
    ay = ay.groupby(['AMASKCD'])['CUSIP'].unique()
    recoms_years_list[year] = ay
    
    
# for each year create an analyst matrix
def intersection(lst1, lst2):
    # Use of hybrid method
    temp = set(lst2)
    lst3 = [value for value in lst1 if value in temp]
    return lst3

analyst_matrix_y = {}
for year in years:
    print(year)
    ay = recoms_years_list[year]
    ay = ay.reset_index()
    ay['temp'] = 1
    ay_matrix = ay.merge(ay, on=['temp'], how='outer')
    ay_matrix.drop(columns='temp', inplace=True)
    ay_matrix.set_index(['AMASKCD_x', 'AMASKCD_y'], inplace=True)
    ay_matrix['firms_union_n'] = ay_matrix.apply(
        lambda x: 
            len(intersection(list(x['CUSIP_x']) , list(x['CUSIP_y'])))
            , axis=1)
    ay_matrix['analyst_n1'] = ay_matrix['CUSIP_x'].apply(lambda x: len(x))
    ay_matrix['analyst_n2'] = ay_matrix['CUSIP_y'].apply(lambda x: len(x))
    ay_matrix['max_ay'] = ay_matrix[['analyst_n1','analyst_n2']].max(axis=1)
    ay_matrix['analyst_overlap'] = ay_matrix['firms_union_n']/ay_matrix['max_ay']                                       
    ay_matrix = ay_matrix[['analyst_overlap']]
    analyst_matrix_y[year] = ay_matrix
    
# for each firm year create a measure of coherence
def Reverse(tuples):
    new_tup = tuples[::-1]
    return new_tup

def get_overlaps(row):
    n = row['n_analysts']
    overlaps = []
    combinations = row['analyst_combs']
    if n==1:
        return overlaps
    else:
        for a_pair in combinations:
            try:
                i2 = ay.loc[a_pair]['analyst_overlap']
                overlaps.append(i2)
            except:
                a_pair = Reversed(a_pair)
                i2 = test2.loc[a_pair]['analyst_overlap']
                overlaps.append(i2)               
        return overlaps

def calculate_coherence(row):
    overlaps = row['overlaps_sum']
    n = row['n_analysts']
    if n==1:
        return np.nan
    else:
        denom = (n*(n-1))/2
        return overlaps/denom
    

firms_y = {}
for year in years:
    print (year)
    ay = analyst_matrix_y[year]
    firms = recoms_years[year].groupby(['CUSIP'])['AMASKCD'].unique().reset_index()
    firms['analyst_combs'] = firms['AMASKCD'].apply(
        lambda x: list(itertools.combinations(x, 2)))
    firms['n_analysts'] = firms['AMASKCD'].apply(
        lambda x: len(x))   
    firms['overlaps'] = firms.apply(
        lambda x: get_overlaps(x), axis=1)  
    firms['overlaps_sum'] = firms['overlaps'].apply(lambda x: sum(x))
    firms['coherence'] = firms.apply(
        lambda x: calculate_coherence(x), axis=1)     
    
    firms_y[year] = firms
    
with open('firm_coherence.pkl', 'wb') as handle:
    pickle.dump(firms_y, handle, protocol=pickle.HIGHEST_PROTOCOL)


# transform to df
firm_y_years = list(firms_y.keys())

audience_dispersion_df = pd.DataFrame()
for year in firm_y_years:
    firm_y_df = pd.DataFrame.from_dict(firms_y[year])
    firm_y_df['year'] = year
    audience_dispersion_df = pd.concat([audience_dispersion_df, firm_y_df])
    
audience_dispersion_df = audience_dispersion_df[['CUSIP', 'year', 'n_analysts', 'coherence']]

audience_dispersion_df.to_pickle('audience_disperions_Zuckerman2004.pkl')
