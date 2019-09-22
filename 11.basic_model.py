import pandas as pd
import numpy as np
import os
import pickle
import sys

input_folder = 'inputs/'

print('Reading input data')
df_item = pd.read_csv(os.path.join(input_folder, 'item_data.csv'))
df_log = pd.read_csv(os.path.join(input_folder, 'view_log.csv'))
df_train = pd.read_csv(os.path.join(input_folder, 'train.csv'))
df_test = pd.read_csv(os.path.join(input_folder, 'test.csv'))

key_cols = ['user_id', 'impression_id']

print('Converting dates')
# conversion for dates
df_log['server_time'] = pd.to_datetime(df_log['server_time'])
df_train['impression_time'] = pd.to_datetime(df_train['impression_time'])
df_test['impression_time'] = pd.to_datetime(df_test['impression_time'])
# convert to integers
# for i in [1, 2, 3]:
    # df_item['category_{:d}'.format(i)] = df_item['category_{:d}'.format(i)].astype(int)

# get weekday and hour
df_train['weekday'] = df_train['impression_time'].dt.weekday
df_train['hour'] = df_train['impression_time'].dt.hour

df_test['weekday'] = df_test['impression_time'].dt.weekday
df_test['hour'] = df_test['impression_time'].dt.hour

print('Creating historic CTRs')
# create historic ctr
col_list = ['weekday', 'hour', 'app_code', 'is_4G', 'os_version', ['weekday', 'hour']]
for col in col_list:
    col_name = col
    if (isinstance(col,list)):
        col_name = '_'.join(col)
    df_temp = df_train.groupby(col)\
                      .apply(lambda x: pd.Series([sum(x['is_click'])/len(x)], index=['click_rate']))\
                      .reset_index().rename(columns={'click_rate':col_name+'_ctr'})
    df_train = pd.merge(left=df_train, right=df_temp, how='left', on=col)
    df_test = pd.merge(left=df_test, right=df_temp, how='left', on=col)

print('Appending train and test data')
# concatenate train and test files to create a unified file
# is_click will be null for test data
df = pd.concat([df_train, df_test], axis=0, sort=False)

df_model = df.loc[:, key_cols + ['impression_time', 'is_click'] + \
                      ['_'.join(x)+'_ctr' if isinstance(x, list) else x+'_ctr' for x in col_list]]

df1 = df.copy()
# feature_days_list = [1, 3]
feature_days_list = [1, 3, 7, 14, 30]

print('Merging logs data')
# merge and filter relevant view logs, keep a lag of 7 days
for lag_days in range(1, 8):
  # lag_days = 7
  df = pd.merge(left=df1, right=df_log, how='left', on=['user_id'])
  df = df.loc[(df['impression_time']-df['server_time']).dt.days >= lag_days, :]
  df = pd.merge(left=df, right=df_item, how='left', on=['item_id'])


  # features from view logs data

  print('Creating visits in last n days')
  # count of site visits in last n days
  for i in feature_days_list:
      print(i)
      df_temp = df.loc[(df['impression_time']-df['server_time']).dt.days <= lag_days+i, :]\
                   .groupby(key_cols).count()['item_id']\
                   .reset_index()\
                   .rename(columns={'item_id':'cnt_visit_l{:d}d'.format(i)})
      df_model = pd.merge(left=df_model, right=df_temp, how='left', on=key_cols)

  print('Creating visits in category in last n days')
  # count of site visits in different categories in last n days
  for col in ['category_1']:
      for cat in list(map(int, df_item[col].unique().tolist())):
          for i in feature_days_list:
              print(col, cat, i)
              df_temp = df.loc[((df['impression_time']-df['server_time']).dt.days <= lag_days+i) &\
                               (df[col] == cat), :]\
                          .groupby(key_cols).count()['item_id']\
                          .reset_index()\
                          .rename(columns={'item_id':'cnt_visit_{:s}_{:d}_l{:d}d'.format(col, cat, i)})
              df_model = pd.merge(left=df_model, right=df_temp, how='left', on=key_cols)

  print('Creating unique item visits in last n days')
  for i in feature_days_list:
    print(i)
    df_temp = df.loc[((df['impression_time']-df['server_time']).dt.days <= lag_days+i), :]\
                .drop_duplicates(key_cols+['item_id'])
    # count of unique items
    df_temp2 = df_temp.groupby(key_cols).count()['item_id']\
                .reset_index()\
                .rename(columns={'item_id':'cnt_uniq_item_l{:d}d'.format(i)})
    df_model = pd.merge(left=df_model, right=df_temp2, how='left', on=key_cols)

    # avg price of unique items
    df_temp2 = df_temp.groupby(key_cols).sum()['item_price']\
                .reset_index()\
                .rename(columns={'item_price':'avg_price_uniq_item_l{:d}d'.format(i)})
    df_model = pd.merge(left=df_model, right=df_temp2, how='left', on=key_cols)

    df_model['avg_price_uniq_item_l{:d}d'.format(i)] = df_model['avg_price_uniq_item_l{:d}d'.format(i)]\
                                                        /df_model['cnt_uniq_item_l{:d}d'.format(i)]
    del df_temp2
  
  # count of unique categories visited
  print('Creating count of unique categories')
  for col in ['category_1', 'category_2', 'category_3', 'product_type']:
    for i in feature_days_list:
      print(i)
      df_temp = df.loc[((df['impression_time']-df['server_time']).dt.days <= lag_days+i), :]\
                  .drop_duplicates(key_cols+[col])\
                  .groupby(key_cols).count()[col]\
                  .reset_index()\
                  .rename(columns={col:'cnt_uniq_{:s}_l{:d}d'.format(col, i)})
      df_model = pd.merge(left=df_model, right=df_temp, how='left', on=key_cols)


  print('Creating cnt days since last visit')
  df_temp = df.loc[:, :]\
              .sort_values(key_cols+['server_time'], ascending=[1,1,0])\
              .drop_duplicates(key_cols)
  df_temp['cnt_days_since_last_visit'] = (df_temp['impression_time']-df_temp['server_time']).dt.total_seconds()/300
  df_model = pd.merge(left=df_model, right=df_temp[key_cols+['cnt_days_since_last_visit']], how='left', on=key_cols)

  df_model.to_csv(os.path.join(input_folder, 'df_model_lag_days_{:d}.csv'.format(lag_days)), index=False)
