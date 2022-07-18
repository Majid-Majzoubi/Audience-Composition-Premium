# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
from funk_svd import SVD
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from scipy.spatial import distance

import seaborn as sns
import matplotlib.pyplot as plt
plt.rc('font',family='Times New Roman')

import matplotlib as mpl


os.chdir('/ML data and analysis')
mpl.rcParams['figure.dpi'] = 900

# =============================================================================
# LOADING AND PREPARING THE RECOMMENDATIONS FILE
# =============================================================================
### Recoms database
recoms = pd.read_csv("recoms_detail_1992_2020.csv", encoding='latin-1')
recoms_cols = ['TICKER', 'CUSIP', 'CNAME', 'OFTIC', 'AMASKCD', 'ACTDATS',
               'ESTIMID', 'ANALYST', 'EMASKCD', 'IRECCD', 'ITEXT',
               'ERECCD', 'ETEXT', 'USFIRM',
               'REVDATS', 'ANNDATS']
recoms = recoms[recoms_cols]

# Creating the date variables
recoms['rev_year'] = recoms['ACTDATS'].astype(str).str.slice(stop=4).astype(int)

# Keeping the last annual recommendation per analyst
recoms = recoms.dropna(subset= ['ANALYST'])
recoms.sort_values(by=['CUSIP','AMASKCD','ACTDATS'], inplace=True)
recoms = recoms.groupby(['CUSIP', 'AMASKCD', 'rev_year']).tail(1)
recoms_cols = ['CUSIP', 'rev_year', 'AMASKCD', 'IRECCD']
recoms = recoms[recoms_cols]
recoms['IRECCD'] = 6 - recoms['IRECCD']

# Keeping only analysts and firms with a minimum level of observations
analyst_min = 10
firm_year_min = 5
recoms_select = recoms.groupby(['AMASKCD']).filter(lambda x: len(x) > analyst_min)
recoms_select = recoms_select.groupby(['CUSIP', 'rev_year']).filter(
        lambda x: len(x) > firm_year_min)


# Loading the LDA transformed corpus - keeping only recoms for which
# the firm is in the corpus
lda_corpus_df2 = pd.read_csv('10ks/lda70_corpus.csv').iloc[:,1:]
lda_corpus_df3 = lda_corpus_df2.set_index(['CUSIPH', 'year']).drop(columns='fdate')

recoms_select = pd.merge(recoms_select, lda_corpus_df2,
                            left_on=['CUSIP', 'rev_year'], right_on=['CUSIPH', 'year'],
                            how='inner')

recoms_select['firm_year'] = list(zip(recoms_select['CUSIP'],
             recoms_select['rev_year']))
recoms_select = recoms_select[['AMASKCD', 'firm_year', 'IRECCD']]
recoms_select.columns = ['u_id', 'i_id', 'rating']

# =============================================================================
# TUNING HYPERPARAMETERS FOR THE FUNK-SVD MODEL
# =============================================================================
train = recoms_select.sample(frac=0.8, random_state=7)
val = recoms_select.drop(train.index.tolist()).sample(frac=0.5, random_state=8)
test = recoms_select.drop(train.index.tolist()).drop(val.index.tolist())


# Tuning for the number of factors - BEST RESULT: 400
mae_list = []
mse_list = []
n_factors = []
for i in range(50, 1000, 50):
    svd = SVD(lr= 0.005, reg=0.2, n_epochs=100, early_stopping=False, 
          n_factors=i, min_rating=1, max_rating=5, shuffle=True)
    svd.fit(X=train)
    pred = svd.predict(val)
    mse = mean_squared_error(val['rating'], pred)
    mse_list.append(mse)
    mae = mean_absolute_error(val['rating'], pred)
    mae_list.append(mae)
    n_factors.append(i)
    print(f'Test MAE: {mae:.2f}')

plt.plot(n_factors, mse_list)
plt.xlabel('Number of factors - From 50 to 1000 at 50 increments')
plt.ylabel('Mean squared error') 
plt.savefig("MSE_50_1000_SVD_nfactors.jpg", dpi=300)


# Tuning for learning rate - BEST RESULT: 0.005
mae_list = []
mse_list = []
learning_rates = [ 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009,
                  0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009,
                   0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
for i in learning_rates:
    svd = SVD(lr= i, reg=0.2, n_epochs=100, early_stopping=False, 
          n_factors=400, min_rating=1, max_rating=5, shuffle=True)
    svd.fit(X=train)
    pred = svd.predict(val)
    mse = mean_squared_error(val['rating'], pred)
    mse_list.append(mse)
    mae = mean_absolute_error(val['rating'], pred)
    mae_list.append(mae)
    print(f'Test MAE: {mae:.2f}')

plt.plot(learning_rates, mse_list)
plt.xlabel('Different learning rates - From 0.0001 to 0.1')
plt.ylabel('Mean squared error') 
plt.savefig("MSE_SVD_lr.jpg", dpi=300)

# Turning for the regularization term - RESULT: 0.2
mae_list = []
mse_list = []
reg_rates = [ 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009,
                   0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
                   0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
for i in reg_rates:
    svd = SVD(lr= 0.005, reg=i, n_epochs=100, early_stopping=False, 
          n_factors=400, min_rating=1, max_rating=5, shuffle=True)
    svd.fit(X=train)
    pred = svd.predict(val)
    mse = mean_squared_error(val['rating'], pred)
    mse_list.append(mse)
    mae = mean_absolute_error(val['rating'], pred)
    mae_list.append(mae)
    print(f'Test MAE: {mae:.2f}')

plt.plot(reg_rates, mse_list)
plt.xlabel('Different learning rates - From 0.001 to 1')
plt.ylabel('Mean squared error') 
plt.savefig("MSE_SVD_regterm.jpg", dpi=300)

# =============================================================================
# TRAIN THE MAIN FUNK-SVD USING HYPERPARAMETERS FROM ABOVE
# FACTORS=400, LEARNING RATE=0.005, REG TERM=0.2
# =============================================================================
svd = SVD(lr= 0.005, reg=0.2, n_epochs=250, early_stopping=False, 
          n_factors=400, min_rating=1, max_rating=5, shuffle=True)
svd.fit(X=train, X_val=val)


# =============================================================================
# TESTING THE TRAINED MODEL
# =============================================================================
pred_ratings = svd.predict(test)
actual_ratings = test["rating"]
test['pred_rating'] = pred_ratings
test['constant'] = 1

# Regressing actual ratings on predicted ratings
est = sm.OLS(test['rating'], test[['pred_rating', 'constant']])
est2 = est.fit()
print(est2.summary())

# Plotting the actual versus predicted ratings
sns.set(color_codes=True, font='Times New Roman')
sns.lmplot(x="pred_rating", y="rating", data=test,
           x_ci=0, aspect=1.5, x_bins=100, truncate=True)
plt.xlim(2,5)
plt.ylim(2,5)
plt.xlabel('Predicted analyst recommendations')
plt.ylabel('Observed analyst recommendations') 

plt.savefig("predict_vs_actual.jpg", dpi=300)



# =============================================================================
# CONSTRUCTING ANALYST AVERAGE LDA PER YEAR
# =============================================================================
# Merging recoms database with corpus - keep only recoms that we have the firm in corpus
recoms_limited = recoms.groupby(['AMASKCD']).filter(lambda x: len(x) > analyst_min)
recoms_limited = recoms_limited.groupby(['CUSIP', 'rev_year']).filter(
        lambda x: len(x) > firm_year_min)

analyst_topics = pd.merge(recoms_limited, lda_corpus_df2,
                            left_on=['CUSIP', 'rev_year'], 
                            right_on=['CUSIPH', 'year'],
                            how='inner')


analyst_topics_year = analyst_topics.drop(columns=[
         'CUSIP', 'rev_year', 'CUSIPH', 'IRECCD', 'fdate' ]).groupby([
                'AMASKCD', 'year']).mean()

# =============================================================================
# MEASURING DISTANCE BETWEEN A FIRM AND EACH ANALYST'S PORTFOLIO 
# ALL POSSIBLE COMBINATIONS OF ANALYST-FIRM-YEAR
# THIS BLOCK TAKES A LONG TIME -- 24-48 HOURS
# =============================================================================
db = recoms_select.copy()

db[['cusip', 'year']] = pd.DataFrame(db['i_id'].tolist(), index=db.index) 

# Create all possible combinations of firm-analyst-year
db_all = pd.merge(db, db[['u_id','year']].drop_duplicates(), 
                    on=['year'], how='outer')
db_all = db_all[['i_id', 'u_id_y']]
db_all.columns = ['i_id', 'u_id']

# Use Funk-SVD to estimate predictions for all possible combinations
pred_all = svd.predict(db_all)
db_all['pred_rating'] = pred_all


# Measuring similarity between analyst and firm using the LDA transformed corpus
db_all[['cusip', 'year']] = pd.DataFrame(db_all['i_id'].tolist(), index=db_all.index) 
db_all = db_all.drop_duplicates(['cusip', 'year', 'u_id'])
db_all['index_n'] =  np.arange(len(db_all))


# calculating JS Shannon similarity
def get_sim(x):
    dis = distance.jensenshannon(lda_corpus_df3.loc[(x['cusip'], x['year'])].iloc[0]
                , analyst_topics_year.loc[(x['u_id'],x['year'])])
    sim = 1-dis
    print (x['index_n'])
    return sim

db_all['sim'] = db_all.apply(
        lambda x: get_sim(x), axis=1)


# =============================================================================
# IMPLEMENTING DISPERSION IN AUDIENCE PREDISPOSITION
# =============================================================================
db_all_dispersion = db_all.groupby('i_id')['pred_rating'].std()
db_all_dispersion = db_all_dispersion.reset_index(name='dispersion')
db_all_dispersion[['cusip', 'year']] = pd.DataFrame(
    db_all_dispersion['i_id'].tolist(), index=db_all_dispersion.index)
del db_all_dispersion["i_id"]

db_all_dispersion.to_pickle('predicted_values_dispersion.pkl')

# =============================================================================
# IMPLEMENTING THE AUDIENCE COMPOSITION PREMIUM VARIABLE
# =============================================================================
db_all['sim_mean0'] = db_all['sim']
db_all['sim_mean0'] = db_all['sim_mean0'].apply(lambda x: 0 if x<7.706687e-02 else x)

db_all.loc[db_all['sim_mean0'] < 3.857389e-02, 'sim_mean0' ] = 0 # 0 for farther 75%

db_all['sim_mean0_sq'] = np.square(db_all['sim_mean0'])
db_all['sim_mean0_sq'] = np.square(db_all['sim']) # instead of above
db_all['sim_sq'] = np.square(db_all['sim'])

db_all['rating_sim_mean0_sq'] = db_all['pred_rating'] * db_all['sim_mean0_sq']
db_all['rating_sim_sq'] = db_all['pred_rating'] * db_all['sim_sq']
db_all['rating_sim'] = db_all['pred_rating'] * db_all['sim']


db_all_vars = db_all.groupby(
        ['i_id'])[['sim', 'rating_sim', 'sim_mean0_sq', 'sim_sq', 'rating_sim_mean0_sq',
        'rating_sim_sq']].sum().reset_index()

db_all_vars['prem'] = db_all_vars['rating_sim']/db_all_vars['sim']
db_all_vars['prem_sq'] = db_all_vars['rating_sim_sq']/db_all_vars['sim_sq']
db_all_vars['prem_sq_mean0'] = db_all_vars['rating_sim_mean0_sq']/db_all_vars['sim_mean0_sq']

db_all.pred_rating_0 = db_all.pred_rating
db_all.loc[db_all['sim'] < 3.857389e-02, 'pred_rating_0' ] = np.nan

db_all_vars_pred_mean = db_all.groupby(
        ['i_id'])['pred_rating'].mean()
db_all_vars['pred_rating_mean_all'] = db_all_vars_pred_mean.values
db_all_vars['prem_less_mean'] = db_all_vars['prem']  - db_all_vars['pred_rating_mean_all']

db_actual = recoms_select.groupby('i_id')['rating'].mean().reset_index()
db_actual = pd.merge(db_actual, db_all_vars, on='i_id', how='inner')
db_actual = db_actual.dropna()
db_actual['constant'] = 1

db_actual.to_pickle('reg_df_nocontrols_JS.pkl')

db_actual[['cusip', 'year']] = pd.DataFrame(
    db_actual['i_id'].tolist(), index=db_actual.index)
db_actual[['rating', 'prem_less_mean', 'constant',
       'cusip', 'year']].to_pickle('predicted_values_distances_lda70_full_JS.pkl')

# Plotting the frequency distribution for audience composition premium
db_actual.prem_less_mean.hist()
sns.set(color_codes=True, font='Times New Roman')
plt.xlabel('Audience composition premium')
plt.ylabel('Frequency') 
plt.savefig("composition_premium.jpg", dpi=300)

# =============================================================================
# MERGING WITH DISPERIONS VARIABLE
# =============================================================================
db_actual_dispersion = pd.merge(db_actual, db_all_dispersion,
                          on=['cusip', 'year'], how='inner')

db_actual_dispersion['dispersion_interaction'] =(
    db_actual_dispersion['dispersion'] * db_actual_dispersion['prem_less_mean'])

db_actual_dispersion.to_pickle('predicted_values_distances_dispersion_lda70_full_JS.pkl')



