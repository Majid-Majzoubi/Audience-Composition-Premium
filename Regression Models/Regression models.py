#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
plt.rc('font',family='Times New Roman')

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 900


# =============================================================================
# IMPORTING KEY FILES
# =============================================================================
os.chdir('/ML data and analysis')

# composition premium file
composition_premium = pd.read_pickle('predicted_values_distances_frag_lda70_full_JS.pkl')

# audience enagement variable
audience_engagement = pd.read_pickle(r'transcripts/diversity_engagement_controls_lda30.pkl')

# controls
controls = pd.read_pickle('database_controls_all.pkl')
controls = controls[['CUSIP', 'year', 'trade_volume_log', 'ROE',
               'market_share_sic4', 
                'adv_f',
                 'intang_f', 'dpt_f', 
                 'mergers', 'number_segments',
                 'prototype_sim_sic4','no_firms_sic4', 'hhi_sic4',
                 'industry_heter_sic4', 'NUMREC_mean_w99',
                  ]]

# audience fragmentation -- Zuckerman method
audience_fragmentation_z =pd.read_pickle('audience_fragmentation_Zuckerman.pkl')
    
# =============================================================================
# REGRESSIONS FOR AUDIENCE COMPOSITION PREMIUM
# =============================================================================
reg_comp_prem = pd.merge(composition_premium, controls,
                                left_on=['cusip', 'year'], right_on=['CUSIP', 'year'],
                                how='left')
reg_comp_prem.set_index(['CUSIP', 'year'], inplace=True)

iv_cols_mainreg = ['prem_less_mean', 'trade_volume_log', 'ROE',
               'market_share_sic4', 
                'adv_f',
                 'intang_f', 'dpt_f', 
                 'mergers', 'number_segments',
                 'prototype_sim_sic4','no_firms_sic4', 'hhi_sic4',
                 'industry_heter_sic4', 'NUMREC_mean_w99',
                 'constant']

reg_comp_prem = reg_comp_prem.dropna(subset = iv_cols_mainreg)

# Model 1 - Main effect - no controls
est = sm.OLS(reg_comp_prem['MEANREC_mean_inv'], 
             reg_comp_prem[['prem_less_mean', 'constant']])
est2 = est.fit()
print(est2.summary())
model1 = pd.DataFrame([est2.params, est2.pvalues]).T
model1_r = pd.DataFrame([est2.nobs, est2.rsquared, est2.fvalue, est2.f_pvalue])

# Model 2 - Main effect - with controls
est = sm.OLS(reg_comp_prem['MEANREC_mean_inv'], reg_comp_prem[iv_cols_mainreg])
est2 = est.fit()
print(est2.summary())

model2 = pd.DataFrame([est2.params, est2.pvalues]).T
model2_r = pd.DataFrame([est2.nobs, est2.rsquared, est2.fvalue, est2.f_pvalue])

# =============================================================================
# REGRESSIONS FOR AUDIENCE FRAGMENTATION MODERATION
# =============================================================================
iv_cols_fragreg = ['prem_less_mean', 
                   'frag_interaction', 'audience_frag',
                   'trade_volume_log', 'ROE',
               'market_share_sic4', 
                'adv_f',
                 'intang_f', 'dpt_f', 
                 'mergers', 'number_segments',
                 'prototype_sim_sic4', 
                 'no_firms_sic4', 'hhi_sic4',
                 'industry_heter_sic4', 'NUMREC_mean_w99',
                 'constant']

# Model 3 - Moderation effect of audience fragmentation
est = sm.OLS(reg_comp_prem['MEANREC_mean_inv'], reg_comp_prem[iv_cols_fragreg])
est2 = est.fit()
print(est2.summary())

model3 = pd.DataFrame([est2.params, est2.pvalues]).T
model3_r = pd.DataFrame([est2.nobs, est2.rsquared, est2.fvalue, est2.f_pvalue])

# =============================================================================
# REGRESSIONS FOR AUDIENCE ENGAGEMENT VARIABLE
# =============================================================================
# Creating one year lag between audience engagement and the DV
audience_engagement['year_lag'] = audience_engagement['year'] - 1
reg_comp_prem_engage = pd.merge(composition_premium, audience_engagement,
                                left_on=['cusip', 'year'], right_on=['cusip', 'year_lag'],
                                how='left')

# Adding 1 to all control variables
reg_comp_prem_engage[['n_questions', 'n_persons', 'n_question_words',
       'n_present_words', 'n_total_words']] = reg_comp_prem_engage[[
           'n_questions', 'n_persons', 'n_question_words',
       'n_present_words', 'n_total_words']] +1
  
# Taking a log transformation of all control variables          
reg_comp_prem_engage[['n_questions', 'n_persons', 'n_question_words',
       'n_present_words', 'n_total_words']] = np.log(reg_comp_prem_engage[[
           'n_questions', 'n_persons', 'n_question_words',
       'n_present_words', 'n_total_words']])


reg_comp_prem_engage.dropna(subset=['prem_less_mean', 'diversity_engagement'],
                            inplace=True)

reg_comp_prem_engage['diversity_engagement_sqr'] =(
    reg_comp_prem_engage['diversity_engagement'].pow(2))

reg_comp_prem_engage['constant'] = 1

# Model 4 - Main quadratic effect - no controls
est = sm.OLS(reg_comp_prem_engage['prem_less_mean'], reg_comp_prem_engage[[
    'diversity_engagement', 'diversity_engagement_sqr', 'constant']])
est2 = est.fit()
model4 = pd.DataFrame([est2.params, est2.pvalues]).T
print(est2.summary())
model4_r = pd.DataFrame([est2.nobs, est2.rsquared, est2.fvalue, est2.f_pvalue])

# Model 5 - Main quadratic effect - with controls
est = sm.OLS(reg_comp_prem_engage['prem_less_mean'], reg_comp_prem_engage[[
    'diversity_engagement', 'diversity_engagement_sqr',
    'n_questions', 'n_persons', 'n_question_words',
       'n_present_words', 'n_total_words', 'constant']])
est2 = est.fit()
model5 = pd.DataFrame([est2.params, est2.pvalues]).T
print(est2.summary())
model5_r = pd.DataFrame([est2.nobs, est2.rsquared, est2.fvalue, est2.f_pvalue])



# =============================================================================
# DESCRIPTIVES
# =============================================================================
# Putting regression results together
models_1to3 = pd.concat([model1, model2, model3], axis=1)
models_4to5 = pd.concat([model4, model5], axis=1)

models_r_1to3 = pd.concat([model1_r, model2_r, model3_r], axis=1)
models_r_4to5 = pd.concat([model4_r, model5_r], axis=1)

models_1to3_mean = reg_comp_prem_frag[['MEANREC_mean_inv', 'prem_less_mean', 'audience_frag',
                   'trade_volume_log', 'ROE',
               'market_share_sic4', 
                'adv_f',
                 'intang_f', 'dpt_f', 
                 'mergers', 'number_segments',
                 'prototype_sim_sic4','no_firms_sic4', 'hhi_sic4',
                 'industry_heter_sic4', 'NUMREC_mean_w99']].mean()

models_1to3_std = reg_comp_prem_frag[['MEANREC_mean_inv', 'prem_less_mean', 'audience_frag',
                   'trade_volume_log', 'ROE',
               'market_share_sic4', 
                'adv_f',
                 'intang_f', 'dpt_f', 
                 'mergers', 'number_segments',
                 'prototype_sim_sic4','no_firms_sic4', 'hhi_sic4',
                 'industry_heter_sic4', 'NUMREC_mean_w99']].std()


models_4to5_mean = reg_comp_prem_engage[['diversity_engagement','n_questions', 
                                         'n_persons', 'n_question_words',
       'n_present_words', 'n_total_words']].mean()

models_4to5_std = reg_comp_prem_engage[['diversity_engagement','n_questions', 
                                         'n_persons', 'n_question_words',
       'n_present_words', 'n_total_words']].std()



# =============================================================================
# ADDITIONAL ANALYSIS - AUDIENCE FRAGMENTATION ZUCKERMAN METHOD
# =============================================================================
audience_fragmentation_z.sort_values(['CUSIP', 'year'], inplace=True)
audience_fragmentation_z['frag_lag'] = audience_fragmentation_z.groupby(
    ['CUSIP'])['fragmentation'].shift()

reg_comp_prem_frag = pd.merge(reg_comp_prem, audience_fragmentation_z,
                                left_index=True, right_on=['CUSIP', 'year'],
                                how='left')

reg_comp_prem_frag['frag_interaction'] =(
    reg_comp_prem_frag['frag_lag'] * reg_comp_prem_frag['prem_less_mean'])

iv_cols_fragreg = ['prem_less_mean', 'frag_interaction', 'frag_lag',
                   'trade_volume_log', 'ROE',
               'market_share_sic4', 
                 'adv_f',
                 'intang_f', 'dpt_f', 
                 'mergers', 'number_segments',
                 'prototype_sim_sic4','no_firms_sic4', 'hhi_sic4',
                 'industry_heter_sic4', 'NUMREC_mean_w99',
                 'constant']

reg_comp_prem_frag = reg_comp_prem_frag.dropna(subset = iv_cols_fragreg)

# Model 3 - Using Zuckerman's method for measuring audience fragmentation
est = sm.OLS(reg_comp_prem_frag['MEANREC_mean_inv'], reg_comp_prem_frag[iv_cols_fragreg])
est2 = est.fit()
print(est2.summary())


