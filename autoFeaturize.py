import pandas as pd
import numpy as np
import psycopg2
from psutil import virtual_memory
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
import gensim
from gensim.models import Word2Vec
import nltk
from nltk.corpus import stopwords
from datetime import datetime
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from math import sqrt
from sklearn.feature_selection import SelectFromModel
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.under_sampling import InstanceHardnessThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import chi2, mutual_info_regression, mutual_info_classif
from sklearn.feature_selection import SelectKBest
from itertools import combinations
from collections import defaultdict
import copy
from scipy import interp
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pickle
import time
import re
import logging
import json
import resource
import psutil
import random
import config_paths
from clickhouse_driver import Client
import sys
sys.path.insert(0, '/opt/cafebot/bots/')
import _helper
import os

con = psycopg2.connect(dbname=config_paths.dbname, user=config_paths.user, host=config_paths.host, port = config_paths.port, password=config_paths.password)

logging.basicConfig(filename = config_paths.autofeaturelog, filemode = 'w', level = logging.DEBUG, format = '%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def query_data(sql, isalien, down_path):
    client = Client('localhost',port=9000,user='default',database='cafebot',compression=True)
    if isalien == 0:
        result, columns = client.execute(sql.replace('#_#','"'), with_column_types=True)
    else:
        result, columns = client.execute(sql.replace("#_#","'"), with_column_types=True)
    df = pd.DataFrame(result, columns=[tuple[0] for tuple in columns])
    if df.shape[0] <2:
        chsize=1
    else:
        chsize = df.shape[0]*0.9
    df.to_csv(''+down_path+'/file.csv', index = False, chunksize=int(chsize))

    df = pd.read_csv(''+down_path+'/file.csv', chunksize=int(chsize))
    df = pd.concat(df, ignore_index=True)
    return(df)

#remaping the target column
def remap(colname):
    lab = LabelEncoder()
    remap.colname = lab.fit_transform(colname.values)

# Function to convert string columns to date&time
def to_date_time(colname):
    to_date_time.colname =  pd.to_datetime(colname, infer_datetime_format = True)

def cleanhtml(sentence):
    cleaned = re.sub(r'<.*?>',r' ', sentence)
    return cleaned

def cleanpunc(word):
    clean = re.sub(r'[?|!|\'|"|#]',r' ', word)
    cleanr = re.sub(r'[.|,|)|(|\|/]',r' ', clean)
    return cleanr


def alphabet_position_numeric(colname):
    lst_of_sent = []
    str1 = ''
    final_str = []

    for sent in colname.values:
        filtered_sent = []
        for word in str(sent).split():
            for clean_word in word.split():
                clean_word = ''.join(str(ord(c)) for c in clean_word)
                filtered_sent.append(clean_word)
        lst_of_sent.append(filtered_sent)

    for lst in lst_of_sent:
        str1 = ''.join(lst)
        final_str.append(str1)

    alphabet_position_numeric.colname = final_str


#Vectorizing text data using basic Neural Network
def vectorize_text(colname):
    lst_of_sent = []

    for sent in colname.values:
        filtered_sent = []
        sent = cleanhtml(sent)
        for words in sent.split():
            for cleaned_words in cleanpunc(words).split():
                if(cleaned_words.isalpha()) & (len(cleaned_words) > 3):
                    if(cleaned_words not in stop):
                        filtered_sent.append(cleaned_words.lower())
        lst_of_sent.append(filtered_sent)

    w2v_model = gensim.models.Word2Vec(lst_of_sent, min_count = 3, size = 50, workers = -1)

    sent_vectors = []

    for sent in lst_of_sent:
        sent_vec = np.zeros(50)
        cnt_words = 0
        for w in sent:
            try:
                vec = w2v_model.wv[w]
                sent_vec += vec
                cnt_words += 1
            except:
                pass
        sent_vec /= cnt_words
        sent_vectors.append(sent_vec)

    vectorize_text.colname = list(sent_vectors)

def clean_currency(word):
    cleaning = re.sub(r'[^\d.]', '', word)
    return cleaning


def drop_constant_columns(dataframe):
    cols = []
    for column in dataframe.columns:
        if len(dataframe[column].unique()) == 1:
            dataframe.drop(column, axis=1)
            cols.append(column)
    return dataframe, cols            

def high_cardin_cols(dataframe, targ_col):
    card_l = []
    for col in dataframe.columns:
        if dataframe[col].dtypes == "object":
            l = []
            if col != ''+targ_col+'':
                val = dataframe.shape[0] / dataframe[col].unique().size
                if val <= 1.20:
                    l.append(col)
        else:
            l = []
        card_l.extend(l)
    return card_l

def fetch_transform_cols(f_d):
    ls = []
    for l in f_d.columns.tolist():
        a = []
        if 'OHE' in l:
            s_o = l.split('_')
            a.append(s_o[1])
        elif 'vector' in l:
            s_v = l.split('_')
            a.append(s_v[1])
        else:
            pass
        ls.extend(a)

    ls = list(set(ls))
    return ls

def fetch_static_cols(f_d):
    ls = []
    for l in f_d.columns.tolist():
        a = []
        if 'OHE' not in l and 'vector' not in l and 'lag' not in l and '_div_by_' not in l and '_mult_by_' not in l and '_minus_' not in l and '_plus_' not in l and '_squared' not in l:
            a.append(l)
        ls.extend(a)

    ls = list(set(ls))
    return ls

def money_to_numeric(colname):
    lst_of_sent = []
    str1 = ''
    final_str = []

    for sent in colname.values:
        filtered_sent = []
        for word in sent.split():
            for clean_word in clean_currency(word).split():
                filtered_sent.append(clean_word)
        lst_of_sent.append(filtered_sent)

    for lst in lst_of_sent:
        str1 = ''.join(lst)
        final_str.append(str1)

    money_to_numeric.colname = final_str


def gen_numeric_interactions(df, columns, target , operations=['/','*','-','+']):
    if target in columns:
        columns.remove(target)

    copy_columns = copy.deepcopy(columns)
    fe_df = pd.DataFrame()
    for combo_col in combinations(columns,2):
        if '/' in operations:
            fe_df['{}_div_by_{}'.format(combo_col[0], combo_col[1]) ] = (df[combo_col[0]]*1.) / df[combo_col[1]]
        if '*' in operations:
            fe_df['{}_mult_by_{}'.format(combo_col[0], combo_col[1]) ] = df[combo_col[0]] * df[combo_col[1]]
        if '-' in operations:
            fe_df['{}_minus_{}'.format(combo_col[0], combo_col[1]) ] = df[combo_col[0]] - df[combo_col[1]]
        if '+' in operations:
            fe_df['{}_plus_{}'.format(combo_col[0], combo_col[1]) ] = df[combo_col[0]] + df[combo_col[1]]

    for each_col in copy_columns:
        fe_df['{}_squared'.format(each_col) ] = df[each_col].pow(2)
        
    # fe_df[columns] = df[columns]
    return fe_df

def left_subtract(l1,l2):
    lst = []
    for i in l1:
        if i not in l2:
            lst.append(i)
    return lst

def return_dictionary_list(lst_of_tuples):
    """ Returns a dictionary of lists if you send in a list of Tuples"""
    orDict = defaultdict(list)
    # iterating over list of tuples
    for key, val in lst_of_tuples:
        orDict[key].append(val)
    return orDict

def find_remove_duplicates(list_of_values):
    """
    Removes duplicates from a list to return unique values - USED ONLY ONCE
    """
    output = []
    seen = set()
    for value in list_of_values:
        if value not in seen:
            output.append(value)
            seen.add(value)
    return output

def find_remove_columns_with_infinity(df, remove=False):
    """
    This function finds all columns in a dataframe that have inifinite values (np.inf or -np.inf)
    It returns a list of column names. If the list is empty, it means no columns were found.
    """
    nums = df.select_dtypes(include='number').columns.tolist()
    dfx = df[nums]
    sum_rows = np.isinf(dfx).values.sum()
    add_cols =  list(dfx.columns.to_series()[np.isinf(dfx).any()])
    if sum_rows > 0:
        print('    there are %d rows and %d columns with infinity in them...' %(sum_rows,len(add_cols)))
        if remove:
            ### here you need to use df since the whole dataset is involved ###
            nocols = [x for x in df.columns if x not in add_cols]
            print("    Shape of dataset before %s and after %s removing columns with infinity" %(df.shape,(df[nocols].shape,)))
            return df[nocols]
        else:
            ## this will be a list of columns with infinity ####
            return add_cols
    else:
        ## this will be an empty list if there are no columns with infinity
        return add_cols

def FE_selection(df, numvars, modeltype, target, corr_limit = 0.70, verbose=0, dask_xgboost_flag=False):
    """
    Feature selection if the target column is given
    """
    df = copy.deepcopy(df)
    df_target = df[target]
    df = df[numvars]
    ### for some reason, doing a mass fillna of vars doesn't work! Hence doing it individually!
    null_vars = np.array(numvars)[df.isnull().sum()>0]
    for each_num in null_vars:
        df[each_num] = df[each_num].fillna(0)
    target = copy.deepcopy(target)

    ### This is a shorter version of getting unduplicated and highly correlated vars ##
    #correlation_dataframe = df.corr().abs().unstack().sort_values().drop_duplicates()
    ### This change was suggested by such3r on GitHub issues. Added Dec 30, 2022 ###
    correlation_dataframe = df.corr().abs().unstack().sort_values().round(7).drop_duplicates()
    corrdf = pd.DataFrame(correlation_dataframe[:].reset_index())
    corrdf.columns = ['var1','var2','coeff']
    corrdf1 = corrdf[corrdf['coeff']>=corr_limit]
    ### Make sure that the same var is not correlated to itself! ###
    corrdf1 = corrdf1[corrdf1['var1'] != corrdf1['var2']]
    correlated_pair = list(zip(corrdf1['var1'].values.tolist(),corrdf1['var2'].values.tolist()))
    corr_pair_dict = dict(return_dictionary_list(correlated_pair))
    corr_list = find_remove_duplicates(corrdf1['var1'].values.tolist()+corrdf1['var2'].values.tolist())
    keys_in_dict = list(corr_pair_dict.keys())
    reverse_correlated_pair = [(y,x) for (x,y) in correlated_pair]
    reverse_corr_pair_dict = dict(return_dictionary_list(reverse_correlated_pair))
    #### corr_pair_dict is used later to make the network diagram to see which vars are correlated to which
    for key, val in reverse_corr_pair_dict.items():
        if key in keys_in_dict:
            if len(key) > 1:
                corr_pair_dict[key] += val
        else:
            corr_pair_dict[key] = val
    
    ###### This is for ordering the variables in the highest to lowest importance to target ###
    if len(corr_list) == 0:
        final_list = list(correlation_dataframe)
        print('Selecting all (%d) variables since none of numeric vars are highly correlated...' %len(numvars))
        return numvars
    else:
        if isinstance(target, list):
            target = target[0]
        max_feats = len(corr_list)
        if modeltype == 'Regression':
            sel_function = mutual_info_regression
            #fs = SelectKBest(score_func=sel_function, k=max_feats)
        else:
            sel_function = mutual_info_classif
            #fs = SelectKBest(score_func=sel_function, k=max_feats)
        ##### you must ensure there are no infinite nor null values in corr_list df ##
        df_fit = df[corr_list]
        ### Now check if there are any NaN values in the dataset #####
        
        if df_fit.isnull().sum().sum() > 0:
            df_fit = df_fit.dropna()
        else:
            print('    there are no null values in dataset...')
        ##### Reduce memory usage and find mutual information score ####       
        #try:
        #    df_fit = reduce_mem_usage(df_fit)
        #except:
        #    print('Reduce memory erroring. Continuing...')
        ##### Ready to perform fit and find mutual information score ####
        
        #try:
            #fs.fit(df_fit, df_target)
        if modeltype == 'Regression':
            fs = mutual_info_regression(df_fit, df_target, n_neighbors=5, discrete_features=False, random_state=42)
        else:
            fs = mutual_info_classif(df_fit, df_target, n_neighbors=5, discrete_features=False, random_state=42)
        # except:
        #     print('    SelectKBest() function is erroring. Returning with all %s variables...' %len(numvars))
        #     return numvars
        try:
            #######   This is the main section where we use mutual info score to select vars        
            #mutual_info = dict(zip(corr_list,fs.scores_))
            mutual_info = dict(zip(corr_list,fs))
            #### The first variable in list has the highest correlation to the target variable ###
            sorted_by_mutual_info =[key for (key,val) in sorted(mutual_info.items(), key=lambda kv: kv[1],reverse=True)]
            #####   Now we select the final list of correlated variables ###########
            selected_corr_list = []
            #### You have to make multiple copies of this sorted list since it is iterated many times ####
            orig_sorted = copy.deepcopy(sorted_by_mutual_info)
            copy_sorted = copy.deepcopy(sorted_by_mutual_info)
            copy_pair = copy.deepcopy(corr_pair_dict)
            #### select each variable by the highest mutual info and see what vars are correlated to it
            for each_corr_name in copy_sorted:
                ### add the selected var to the selected_corr_list
                selected_corr_list.append(each_corr_name)
                for each_remove in copy_pair[each_corr_name]:
                    #### Now remove each variable that is highly correlated to the selected variable
                    if each_remove in copy_sorted:
                        copy_sorted.remove(each_remove)
            ##### Now we combine the uncorrelated list to the selected correlated list above
            rem_col_list = left_subtract(numvars,corr_list)
            final_list = rem_col_list + selected_corr_list
            removed_cols = left_subtract(numvars, final_list)
        except Exception as e:
            print('    SULOV Method crashing due to %s' %e)
            #### Dropping highly correlated Features fast using simple linear correlation ###
            removed_cols = remove_highly_correlated_vars_fast(df,corr_limit)
            final_list = left_subtract(numvars, removed_cols)
        if len(removed_cols) > 0:
            print('    Removing (%d) highly correlated variables:' %(len(removed_cols)))
            if len(removed_cols) <= 30:
                print('    %s' %removed_cols)
            if len(final_list) <= 30:
                print('    Following (%d) vars selected: %s' %(len(final_list),final_list))
        ##############    D R A W   C O R R E L A T I O N   N E T W O R K ##################
        selected = copy.deepcopy(final_list)
        if verbose:
            try:
                import networkx as nx
                #### Now start building the graph ###################
                gf = nx.Graph()
                ### the mutual info score gives the size of the bubble ###
                multiplier = 2100
                for each in orig_sorted:
                    gf.add_node(each, size=int(max(1,mutual_info[each]*multiplier)))
                ######### This is where you calculate the size of each node to draw
                sizes = [mutual_info[x]*multiplier for x in list(gf.nodes())]
                ####  The sizes of the bubbles for each node is determined by its mutual information score value
                corr = df_fit.corr()
                high_corr = corr[abs(corr)>corr_limit]
                ## high_corr is the dataframe of a few variables that are highly correlated to each other
                combos = combinations(corr_list,2)
                ### this gives the strength of correlation between 2 nodes ##
                multiplier = 20
                for (var1, var2) in combos:
                    if np.isnan(high_corr.loc[var1,var2]):
                        pass
                    else:
                        gf.add_edge(var1, var2,weight=multiplier*high_corr.loc[var1,var2])
                ######## Now start building the networkx graph ##########################
                widths = nx.get_edge_attributes(gf, 'weight')
                nodelist = gf.nodes()
                cols = 5
                height_size = 5
                width_size = 15
                rows = int(len(corr_list)/cols)
                if rows < 1:
                    rows = 1
                plt.figure(figsize=(width_size,min(20,height_size*rows)))
                pos = nx.shell_layout(gf)
                nx.draw_networkx_nodes(gf,pos,
                                       nodelist=nodelist,
                                       node_size=sizes,
                                       node_color='blue',
                                       alpha=0.5)
                nx.draw_networkx_edges(gf,pos,
                                       edgelist = widths.keys(),
                                       width=list(widths.values()),
                                       edge_color='lightblue',
                                       alpha=0.6)
                pos_higher = {}
                x_off = 0.04  # offset on the x axis
                y_off = 0.04  # offset on the y axis
                for k, v in pos.items():
                    pos_higher[k] = (v[0]+x_off, v[1]+y_off)
                if len(selected) == 0:
                    nx.draw_networkx_labels(gf, pos=pos_higher,
                                        labels=dict(zip(nodelist,nodelist)),
                                        font_color='black')
                else:
                    nx.draw_networkx_labels(gf, pos=pos_higher,
                                        labels = dict(zip(nodelist,[x+' (selected)' if x in selected else x+' (removed)' for x in nodelist])),
                                        font_color='black')
                plt.box(True)
                plt.title("""In SULOV, we repeatedly remove features with lower mutual info scores among highly correlated pairs (see figure),
                            SULOV selects the feature with higher mutual info score related to target when choosing between a pair. """, fontsize=10)
                plt.suptitle('How SULOV Method Works by Removing Highly Correlated Features', fontsize=20,y=1.03)
                red_patch = mpatches.Patch(color='blue', label='Bigger circle denotes higher mutual info score with target')
                blue_patch = mpatches.Patch(color='lightblue', label='Thicker line denotes higher correlation between two variables')
                plt.legend(handles=[red_patch, blue_patch],loc='best')
                plt.show()
                #####    N E T W O R K     D I A G R A M    C O M P L E T E
                return final_list
            except Exception as e:
                print('    Networkx library visualization crashing due to %s' %e)
                print('Completed SULOV. %d features selected' %len(final_list))
        else:
            print('Completed SULOV. %d features selected' %len(final_list))
        return final_list

def transformData(fileid=0, filePath='', targetCol='', timeCol='', classification=1,  standardize=False, test_size=0.2, downloadPath='.', sourcequery = '', isbeta = 0, isalien = 0, islocal=1, tablename='', down_path_remote='', language='english', expertDic=[{'num_inter': True, 'cat_embeds': True, 'drop_cols': []}]):    
    up_path = filePath
    targ_col = targetCol
    time_col = timeCol
    down_path = downloadPath

    if standardize == 1:
        standardize = True
    elif standardize == 0:
        standardize = False
    else:
        pass

    start = time.time()
    samp='auto'

    if classification == 1:
        try:
            if config_paths.isquery == 0:
                if isbeta==1:                                
                    n = sum(1 for line in open(r''+up_path+'')) - 1 #number of records in file (excludes header)
                    if n>999:
                        s=999
                    else:
                        s=n
                        #s = 999 #desired sample size
                    skip = sorted(random.sample(range(1,n+1),n-s)) #the 0-indexed header will not be included in the skip list
                    data = pd.read_csv(r''+up_path+'', skiprows=skip,quotechar='"',skipinitialspace=True)
                else:
                    data = pd.read_csv(r''+up_path+'',quotechar='"',skipinitialspace=True)
            else:
                data = query_data(sourcequery, isalien, down_path)

            if expertDic[0]['drop_cols']:
                data.drop(expertDic[0]['drop_cols'], axis=1, inplace=True)

            data_leak = data
            if targ_col != '':
                data_leak = data_leak.drop(''+targ_col+'', axis = 1)
            data_leak.to_csv(''+down_path+'/train_data_not_processed.csv', index = False)

            #uploading schema
            data_sch = pd.read_csv(r''+down_path+'/schema.csv')

            #Dropping null columns
            t_null = data.isnull().sum().reset_index()
            t_null.columns = ['cols', 'sum_null']

            null_col = []
            for row in t_null.values:
                if (row[1]/data.shape[0])*100 >= 25:
                    null_col.append(row[0])
                else:
                    pass

            if targ_col in null_col:
                null_col.remove(targ_col)
                data = data.drop(null_col, axis = 1)
            else:
                data = data.drop(null_col, axis = 1)

            if targ_col != '':
                data = data[~data[targ_col].isna()]
            #data_sch = data_sch.drop(null_col, axis = 1)

            #data_str = data_sch.columns[data_sch.isin(['STRING']).any()]
            data_dt = data_sch.columns[data_sch.isin(['DATETIME']).any()]

            #data_mon = data_sch.columns[data_sch.isin(['MONEY']).any()]
            if targ_col != '':
                targCount = data[targ_col].value_counts().reset_index()
                targCount.columns = ['val','val_count']

                if int(targCount.shape[0]) <= 1:
                    raise ValueError('The target column consists of only 1 or no class. Please check your data and run the model again !')
                elif int(targCount.shape[0]) >= 50:
                    raise ValueError('Please input a target column with classes less than/equal to 50 for classification modelling !')
                else:
                    pass

                targCountCheck = targCount[targCount.val_count <=25]
                if targCountCheck.empty:
                    pass
                else:
                    raise ValueError('The target column consists of a class which has very few observations for modelling')

            for dt_col in data_dt:
                try:
                    if dt_col == time_col:
                        data = data[~data[dt_col].isna()]
                        to_date_time(data[dt_col])
                        data[dt_col] = to_date_time.colname
                    else:
                        data = data.drop(dt_col, axis = 1)
                except:
                    if dt_col not in data.columns.tolist():
                        pass
                    else:
                        raise ValueError('The given datetime column for timeseries has mixed datetime formats. Please standardize the format !')


            data_cat = data.select_dtypes(exclude=[np.number])

            for col in data.columns:
                    data[col] = data[col].fillna(data[col].mode()[0])

            try:
                stop = stopwords.words(language)
            except:
                nltk.download('stopwords')
                stop = stopwords.words(language)

            def data_seg(dataset, colname, feat_eng_pre_time):                    
                if feat_eng_pre_time <=20:
                    try:
                        if samp == 'over':
                            print('oversampling initiated')
                            label = dataset[''+colname+'']
                            data_feat = dataset.drop(''+colname+'', axis = 1)
                            smote = SMOTE(sampling_strategy='minority', n_jobs = -1)
                            x_sm, y_sm = smote.fit_sample(data_feat, label)
                            if x_sm.shape[0] >= 100:
                                data_seg.x_tr, data_seg.x_test, data_seg.y_tr, data_seg.y_test = train_test_split(x_sm, y_sm, test_size = test_size, random_state = 42, shuffle=True)
                            else:
                                raise ValueError('Data not sufficient')
                        elif samp == 'under':
                            print('undersampling initiated')
                            label = dataset[''+colname+'']
                            data_feat = dataset.drop(''+colname+'', axis = 1)
                            smote = SMOTEENN(enn=EditedNearestNeighbours(sampling_strategy='majority'), n_jobs = -1)
                            x_sm, y_sm = smote.fit_sample(data_feat, label)
                            if x_sm.shape[0] >= 100:
                                data_seg.x_tr, data_seg.x_test, data_seg.y_tr, data_seg.y_test = train_test_split(x_sm, y_sm, test_size = test_size, random_state = 42, shuffle=True)
                            else:
                                raise ValueError('Data not sufficient')
                        elif samp == 'auto':
                            print('autosampling initiated')
                            label = dataset[''+colname+'']
                            data_feat = dataset.drop(''+colname+'', axis = 1)
                            smote = SMOTEENN(sampling_strategy = 'auto', n_jobs = -1)
                            x_sm, y_sm = smote.fit_resample(data_feat, label)
                            if x_sm.shape[0] >= 100:
                                data_seg.x_tr, data_seg.x_test, data_seg.y_tr, data_seg.y_test = train_test_split(x_sm, y_sm, test_size = test_size, random_state = 42, shuffle=True)
                            else:
                                raise ValueError('Data not sufficient')
                    except:
                        label = dataset[''+colname+'']
                        data_feat = dataset.drop(''+colname+'', axis = 1)
                        #smote = SMOTEENN(sampling_strategy = 'auto', smote=SMOTE(k_neighbors = 1), n_jobs = -1)
                        #x_sm, y_sm = smote.fit_sample(data_feat, label)
                        data_seg.x_tr, data_seg.x_test, data_seg.y_tr, data_seg.y_test = train_test_split(data_feat, label, test_size = test_size, random_state = 42, shuffle=True)
                else:
                    print('No sampling initiated')
                    label = dataset[''+colname+'']
                    data_feat = dataset.drop(''+colname+'', axis = 1)
                    data_seg.x_tr, data_seg.x_test, data_seg.y_tr, data_seg.y_test = train_test_split(data_feat, label, test_size = test_size, random_state = 42, shuffle=True)

            for col in data_cat.columns:
                data[col] = data[col].fillna(data_cat[col].value_counts().idxmax())

            data, constant_cols = drop_constant_columns(data)

            card_cols = high_cardin_cols(data, targ_col)
            data = data.drop(card_cols, axis = 1)
            
            if targ_col != '':
                if data[targ_col].dtype == 'float64':
                    remap(data[targ_col].astype(str))
                    data[targ_col] = remap.colname
                else:
                    pass

            if expertDic[0]['num_inter'] == True:

                interactcols = [col for col in data.columns if data[col].dtype == 'int64' or data[col].dtype == 'float64' or data[col].dtype == 'int32' or data[col].dtype == 'float32' or data[col].dtype == 'int16' or data[col].dtype == 'float16' or data[col].dtype == 'uint8' or data[col].dtype == 'uint16']
                interactDF = gen_numeric_interactions(data, interactcols, targ_col)

                if targ_col != '':
                    data_dim = int((len(interactDF)*interactDF.shape[1])/1e6)
                    interactDF.drop(find_remove_columns_with_infinity(interactDF), axis=1, inplace=True)

                    if len(interactcols) >= 1:
                        if data_dim < 50:
                            try:
                                if data[targ_col].dtypes == 'object':
                                    data_temp = data
                                    remap(data_temp[targ_col].astype(str))
                                    data_temp[targ_col] = remap.colname
                                    interactDF[targ_col] = data_temp[targ_col]
                                    del data_temp
                                else:
                                    interactDF[targ_col] = data[targ_col]
                                final_list = FE_selection(interactDF, interactDF.columns.tolist(), 'Classification', targ_col)
                            except:
                                print('SULOV method is erroring. Continuing ...')
                                final_list = copy.deepcopy(numvars)
                        else:
                                print('Running SULOV on smaller dataset sample since data size %s m > 50 m. Continuing ...' %int(data_dim))
                                data_temp = interactDF.sample(n=10000, replace=True, random_state=99)
                                final_list = FE_selection(interactDF, interactDF.columns.tolist(), 'Classification', targ_col)
                                del data_temp

                        final_list.remove(targ_col)
                        interactDF = interactDF[final_list]
            else:
                interactcols= []

            if time_col == 'null' or time_col == 'None' or time_col == '':
                vector_cols = []
                onehot_cols = []
                if expertDic[0]['cat_embeds'] == True:
                    for col in data.columns:
                        if data[col].dtypes == "object":
                            if col == ''+targ_col+'' or data[col].dtype == np.bool_:
                                remap(data[col].astype(str))
                                data[col] = remap.colname
                            elif col != ''+targ_col+'':
                                l = []
                                df_rand = data.sample(frac=0.15)
                                for row in df_rand[col]:
                                    leng = len(str(row))
                                    l.append(leng)
                                avg = np.mean(l)
                                maxx = np.max(l)
                                uniq = data[col].unique().size
                                vec_cols = []
                                ohe_cols = []
                                if avg >= 25 and uniq >= 100 and uniq <= 500:
                                    print('vector')
                                    vectorize_text(data[col])
                                    vec_cols.append(col)
                                    data[col] = vectorize_text.colname
                                    data2 = data.loc[:, data.columns.intersection([col])]
                                    data2 = pd.DataFrame(data2[col].values.tolist(), index = data2.index)
                                    data2 = data2.add_suffix('_'+col+'_vector')
                                    data = data.drop(col, axis =1)
                                    data = data.join(data2)
                                elif avg >= 6 and avg <=24 and uniq <= 50:
                                    print('ohe')
                                    ohe_cols.append(col)
                                    encoded_features = pd.get_dummies(data[col], prefix = 'OHE'+'_'+col)
                                    data = data.drop(col, axis =1)
                                    data = data.join(encoded_features)
                                elif maxx <= 50:
                                    alphabet_position_numeric(data[col])
                                    data[col] = alphabet_position_numeric.colname
                                else:
                                    data = data.drop(col, axis = 1)
                            try:
                                vec_cols
                            except:
                                pass
                            else:
                                vector_cols.extend(vec_cols)
                                onehot_cols.extend(ohe_cols)
            else:
                data = data.sort_values(by = ''+time_col+'')

                vector_cols = []
                onehot_cols = []
                if expertDic[0]['cat_embeds'] == True:
                    for col in data.columns:
                        if data[col].dtypes == "object":
                            if col == ''+targ_col+'' or data[col].dtype == np.bool_:
                                remap(data[col].astype(str))
                                data[col] = remap.colname
                            elif col != ''+targ_col+'':
                                l = []
                                df_rand = data.sample(frac=0.15)
                                for row in df_rand[col]:
                                    leng = len(str(row))
                                    l.append(leng)
                                avg = np.mean(l)
                                maxx = np.max(l)
                                uniq = data[col].unique().size
                                vec_cols = []
                                ohe_cols = []
                                if avg >= 25 and uniq >= 100 and uniq <= 500:
                                    print('vector')
                                    vectorize_text(data[col])
                                    vec_cols.append(col)
                                    data[col] = vectorize_text.colname
                                    data2 = data.loc[:, data.columns.intersection([col])]
                                    data2 = pd.DataFrame(data2[col].values.tolist(), index = data2.index)
                                    data2 = data2.add_suffix('_'+col+'_vector')
                                    data = data.drop(col, axis =1)
                                    data = data.join(data2)
                                elif avg >= 6 and avg <=24 and uniq <= 100:
                                    print('ohe')
                                    ohe_cols.append(col)
                                    encoded_features = pd.get_dummies(data[col], prefix = 'OHE'+'_'+col)
                                    data = data.drop(col, axis =1)
                                    data = data.join(encoded_features)
                                elif maxx <= 50:
                                    alphabet_position_numeric(data[col])
                                    data[col] = alphabet_position_numeric.colname
                                else:
                                    data = data.drop(col, axis = 1)
                            try:
                                vec_cols
                            except:
                                pass
                            else:
                                vector_cols.extend(vec_cols)
                                onehot_cols.extend(ohe_cols)

                time_cols = {'vals' : time_col}
                time_file = open(''+down_path+'/time_vals', 'wb')
                pickle.dump(time_cols, time_file)
                time_file.close()

                if data[time_col].dt.minute.all() == 0 and data[time_col].dt.hour.all() == 0:
                    #date
                    data['year']=data[time_col].dt.year
                    data['quarter'] = data[time_col].dt.quarter
                    data['month']=data[time_col].dt.month 
                    data['day']=data[time_col].dt.day
                    data['weekofyear'] = data[time_col].dt.weekofyear
                    data['dayofweek_num']=data[time_col].dt.dayofweek
                    data = data.drop(time_col, axis =1)
                else:
                    #date
                    data['year']=data[time_col].dt.year
                    data['quarter'] = data[time_col].dt.quarter
                    data['month']=data[time_col].dt.month 
                    data['day']=data[time_col].dt.day
                    data['weekofyear'] = data[time_col].dt.weekofyear
                    data['dayofweek_num']=data[time_col].dt.dayofweek

                    #time
                    data['Hour'] = data[time_col].dt.hour 
                    data['minute'] = data[time_col].dt.minute
                    data['second'] = data[time_col].dt.second
                    data = data.drop(time_col, axis =1)

            #for mon_col in data_mon:
            #    data[mon_col] = data[mon_col].apply(str)
            #    money_to_numeric(data[mon_col])
            #    data[mon_col] = money_to_numeric.colname
            if expertDic[0]['num_inter'] == True:
                data = pd.concat([data, interactDF], axis =1)

            if expertDic[0]['cat_embeds'] == True:
                data = data.apply(pd.to_numeric, errors ='coerce')
            data = data.replace([np.inf, -np.inf], np.nan)
            data = data.fillna(data.mean())

            regex = re.compile(r"\[|\]|<", re.IGNORECASE)
            data.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in data.columns]
            
            if targ_col != '':
                if data[targ_col].dtypes == 'float64':
                    data[targ_col] = data[targ_col].astype(int)
                else:
                    pass

            if targ_col != '':
                classs = data[''+targ_col+'']
                dataa = data.drop(''+targ_col+'', axis = 1)
            else:
                dataa = data

            vector_cols = list(set(vector_cols))
            onehot_cols = list(set(onehot_cols))
            constant_cols = list(set(constant_cols))
            card_cols = list(set(card_cols))
            interactcols = list(set(interactcols))
            dropcols = list(set(expertDic[0]['drop_cols']))

            trans_dic = {"onehotencode" : onehot_cols, "vectorize" : vector_cols, "constant" : constant_cols, "cardinal" : card_cols, "interactions" : interactcols, "target" : [targ_col], "dropped" : dropcols}
            
            if time_col == '':
                example_df = data_leak[fetch_static_cols(dataa)+onehot_cols+vector_cols]
                example_df.iloc[0:11,:].to_csv(''+down_path+'/example.csv', index = False)
            else:
                fsc = set(fetch_static_cols(dataa)) - set(list(['year', 'quarter', 'month', 'day', 'weekofyear', 'dayofweek_num', 'Hour', 'minute', 'second']))
                fsc_ls = list(fsc)
                example_df = data_leak[fsc_ls+onehot_cols+vector_cols+[time_col]]
                example_df.iloc[0:11,:].to_csv(''+down_path+'/example.csv', index = False)
            
            if standardize == True:
                std = StandardScaler()
                print(data.info(verbose=1))

                tempstdcols = []
                for i,v in enumerate(data.dtypes.values):
                    if v.name == 'int64' or v.name == 'float64' or v.name == 'int32' or v.name == 'float32' or v.name == 'int16' or v.name == 'float16' or v.name == 'uint8' or v.name == 'uint16':
                        tempstdcols.append(data.columns.tolist()[i])

                print(tempstdcols)
                if tempstdcols:
                    if targ_col in tempstdcols:
                        tempstdcols.remove(targ_col)

                    dataStd = std.fit_transform(data[tempstdcols])
                    pickle.dump(std, open(down_path+'/scaler.pkl','wb'))
                    data = pd.concat([pd.DataFrame(dataStd, columns = data[tempstdcols].columns.tolist()), data[[i for i in data.columns.tolist() if i not in tempstdcols]]], axis=1)

                trans_dic["standard"] = tempstdcols
            else:
                trans_dic["standard"] = []
                pass

            with open(down_path+'/transform.json', 'w') as fp:
                json.dump(trans_dic, fp)
            
            dataa.iloc[0:100,:].to_csv(''+down_path+'/preprocessed_data.csv', index = False)
            
            feat_end_pre = time.time()
            feat_eng_pre_time = float((feat_end_pre - start)/60)

            if targ_col != '':
                data_seg(data, targ_col, feat_eng_pre_time)
                pd.DataFrame(data_seg.x_tr, columns = dataa.columns.tolist()).to_csv(''+down_path+'/train_data_processed.csv', index = False)
            else:
                dataa.to_csv(''+down_path+'/train_data_processed.csv', index = False)
            
            if isalien == 1:
                return data_seg.x_tr, data_seg.x_test, data_seg.y_tr, data_seg.y_test, 'null'
            else:
                pass
                # autoTrainDF = pd.DataFrame(data_seg.x_tr, columns = dataa.columns.tolist())
                # autoTestDF = pd.DataFrame(data_seg.x_test, columns = dataa.columns.tolist())
                # autoTrainDF[targ_col] = data_seg.y_tr
                # autoTestDF[targ_col] = data_seg.y_test
                # vertical_stack = pd.concat([autoTrainDF, autoTestDF], axis=0)
                
            if islocal == 1:
                _helper.fileid = fileid
                _helper.publishbot(data)
            elif islocal == 0:
                data.to_csv(down_path+'/'+tablename+'.csv', index = False)
                ftp_server.put(config_paths.nifi_in_path+'/'+tablename+'.csv', down_path_remote+'/'+tablename+'.csv')
            
            cur = con.cursor()
            cur.execute("Update queue set status = 2, err = 'AutoFeature engineering Completed' where fileflowid = %d" %(int(fileid)))
            con.commit()
            cur.close()

        except Exception as e:
            cur = con.cursor()
            cur.execute("Update queue set status = -2 ,err = '" + str(e).replace("'","''") + "' where fileflowid=%d" %(int(fileid)))
            con.commit()
            cur.close()
            logger.error('autoFeature transformation crashed', exc_info=True)
            x_tr = pd.DataFrame()
            x_test = pd.DataFrame()
            y_tr = pd.DataFrame()
            y_test = pd.DataFrame()
            return x_tr, x_test, y_tr, y_test, str(e)
            
            
    elif classification == 0:
        try:
            if config_paths.isquery == 0:
                if isbeta==1:                                
                    n = sum(1 for line in open(r''+up_path+'')) - 1 #number of records in file (excludes header)
                    if n>999:
                        s=999
                    else:
                        s=n
                        #s = 999 #desired sample size
                    skip = sorted(random.sample(range(1,n+1),n-s)) #the 0-indexed header will not be included in the skip list
                    data = pd.read_csv(r''+up_path+'', skiprows=skip,quotechar='"',skipinitialspace=True)
                else:
                    data = pd.read_csv(r''+up_path+'',quotechar='"',skipinitialspace=True)
            else:
                data = query_data(sourcequery, isalien, down_path)

            if expertDic[0]['drop_cols']:
                data.drop(expertDic[0]['drop_cols'], axis=1, inplace=True)

            data_leak = data
            if targ_col != '':
                data_leak = data_leak.drop(''+targ_col+'', axis = 1)
            data_leak.to_csv(''+down_path+'/train_data_not_processed.csv', index = False)

            data_sch = pd.read_csv(r''+down_path+'/schema.csv')

            #Dropping null columns
            t_null = data.isnull().sum().reset_index()
            t_null.columns = ['cols', 'sum_null']

            null_col = []
            for row in t_null.values:
                if (row[1]/data.shape[0])*100 >= 50:
                    null_col.append(row[0])
                else:
                    pass

            data = data.drop(null_col, axis = 1)
            if targ_col != '':
                data = data[~data[targ_col].isna()]
            #data_sch = data_sch.drop(null_col, axis = 1)

            #data_str = data_sch.columns[data_sch.isin(['STRING']).any()]
            data_dt = data_sch.columns[data_sch.isin(['DATETIME']).any()]
            #data_mon = data_sch.columns[data_sch.isin(['MONEY']).any()]

            for dt_col in data_dt:
                try:
                    if dt_col == time_col:
                        data = data[~data[dt_col].isna()]
                        to_date_time(data[dt_col])
                        data[dt_col] = to_date_time.colname
                    else:
                        data = data.drop(dt_col, axis = 1)
                except:
                    if dt_col not in data.columns.tolist():
                        pass
                    else:
                        raise ValueError('The given datetime column for timeseries has mixed datetime formats. Please standardize the format !')

            
            data_cat = data.select_dtypes(exclude=[np.number])
            
            try:
                stop = stopwords.words(language)
            except:
                nltk.download('stopwords')
                stop = stopwords.words(language)

            for col in data.columns:
                data[col] = data[col].fillna(data[col].mode()[0])

            if targ_col != '':
                targCount = data[targ_col].value_counts().reset_index()
                targCount.columns = ['val','val_count']
                    
                if int(targCount.shape[0]) < 50:
                    raise ValueError('Please input a continuous target column for regression modelling !')
                else:
                    pass

            def data_seg(dataset, colname):
                label = dataset[''+colname+'']
                data_feat = dataset.drop(''+colname+'', axis = 1)
                data_seg.x_tr, data_seg.x_test, data_seg.y_tr, data_seg.y_test = train_test_split(data_feat, label, test_size = test_size, random_state = 42, shuffle=True)

            for col in data_cat.columns:
                data[col] = data[col].fillna(data_cat[col].value_counts().idxmax())

            data, constant_cols = drop_constant_columns(data)

            card_cols = high_cardin_cols(data, targ_col)
            data = data.drop(card_cols, axis = 1)

            if expertDic[0]['num_inter'] == True:
                interactcols = [col for col in data.columns if data[col].dtype == 'int64' or data[col].dtype == 'float64' or data[col].dtype == 'int32' or data[col].dtype == 'float32' or data[col].dtype == 'int16' or data[col].dtype == 'float16' or data[col].dtype == 'uint8' or data[col].dtype == 'uint16']
                interactDF = gen_numeric_interactions(data, interactcols, targ_col)

                if targ_col != '':
                    data_dim = int((len(interactDF)*interactDF.shape[1])/1e6)
                    interactDF.drop(find_remove_columns_with_infinity(interactDF), axis=1, inplace=True)

                    if len(interactcols) >= 1:
                        if data_dim < 50:
                            try:
                                interactDF[targ_col] = data[targ_col]
                                final_list = FE_selection(interactDF, interactDF.columns.tolist(), 'Regression', targ_col)
                            except:
                                print('SULOV method is erroring. Continuing ...')
                                final_list = copy.deepcopy(interactDF.columns.tolist())
                        else:
                                print('Running SULOV on smaller dataset sample since data size %s m > 50 m. Continuing ...' %int(data_dim))
                                data_temp = interactDF[:10000]
                                final_list = FE_selection(interactDF, interactDF.columns.tolist(), 'Regression', targ_col)
                                del data_temp

                        final_list.remove(targ_col)
                        interactDF = interactDF[final_list]
            else:
                interactcols = []

            if time_col == '':
                vector_cols = []
                onehot_cols = []
                if expertDic[0]['cat_embeds'] == True:
                    for col in data.columns:
                        if data[col].dtypes == "object":
                            if col == ''+targ_col+'' or data[col].dtype == np.bool_:
                                remap(data[col].astype(str))
                                data[col] = remap.colname
                            elif col != ''+targ_col+'':
                                l = []
                                df_rand = data.sample(frac=0.15)
                                for row in df_rand[col]:
                                    leng = len(str(row))
                                    l.append(leng)
                                avg = np.mean(l)
                                maxx = np.max(l)
                                uniq = data[col].unique().size
                                vec_cols = []
                                ohe_cols = []
                                if avg >= 25 and uniq >= 100 and uniq <= 500:
                                    print('vector')
                                    vectorize_text(data[col])
                                    vec_cols.append(col)
                                    data[col] = vectorize_text.colname
                                    data2 = data.loc[:, data.columns.intersection([col])]
                                    data2 = pd.DataFrame(data2[col].values.tolist(), index = data2.index)
                                    data2 = data2.add_suffix('_'+col+'_vector')
                                    data = data.drop(col, axis =1)
                                    data = data.join(data2)
                                elif avg >= 6 and avg <=24 and uniq <= 100:
                                    print('ohe')
                                    ohe_cols.append(col)
                                    encoded_features = pd.get_dummies(data[col], prefix = 'OHE'+'_'+col)
                                    data = data.drop(col, axis =1)
                                    data = data.join(encoded_features)
                                elif maxx <= 50:
                                    alphabet_position_numeric(data[col])
                                    data[col] = alphabet_position_numeric.colname
                                else:
                                    data = data.drop(col, axis = 1)
                            try:
                                vec_cols
                            except:
                                pass
                            else:
                                vector_cols.extend(vec_cols)
                                onehot_cols.extend(ohe_cols)
            else:
                data = data.sort_values(by = ''+time_col+'')

                vector_cols = []
                onehot_cols = []
                if expertDic[0]['cat_embeds'] == True:
                    for col in data.columns:
                        if data[col].dtypes == "object":
                            if col == ''+targ_col+'' or data[col].dtype == np.bool_:
                                remap(data[col].astype(str))
                                data[col] = remap.colname
                            elif col != ''+targ_col+'':
                                l = []
                                df_rand = data.sample(frac=0.15)
                                for row in df_rand[col]:
                                    leng = len(str(row))
                                    l.append(leng)
                                avg = np.mean(l)
                                maxx = np.max(l)
                                uniq = data[col].unique().size
                                vec_cols = []
                                ohe_cols = []
                                if avg >= 25 and uniq >= 100 and uniq <= 520:
                                    print('vector')
                                    vectorize_text(data[col])
                                    vec_cols.append(col)
                                    data[col] = vectorize_text.colname
                                    data2 = data.loc[:, data.columns.intersection([col])]
                                    data2 = pd.DataFrame(data2[col].values.tolist(), index = data2.index)
                                    data2 = data2.add_suffix('_'+col+'_vector')
                                    data = data.drop(col, axis =1)
                                    data = data.join(data2)
                                elif avg >= 6 and avg <=24 and uniq <= 100:
                                    print('ohe')
                                    ohe_cols.append(col)
                                    encoded_features = pd.get_dummies(data[col], prefix = 'OHE'+'_'+col)
                                    data = data.drop(col, axis =1)
                                    data = data.join(encoded_features)
                                elif maxx <= 50:
                                    alphabet_position_numeric(data[col])
                                    data[col] = alphabet_position_numeric.colname
                                else:
                                    data = data.drop(col, axis = 1)
                            try:
                                vec_cols
                            except:
                                pass
                            else:
                                vector_cols.extend(vec_cols)
                                onehot_cols.extend(ohe_cols)

                if data[time_col].dt.minute.all() == 0 and data[time_col].dt.hour.all() == 0:
                    #date
                    data['year']=data[time_col].dt.year
                    data['quarter'] = data[time_col].dt.quarter
                    data['month']=data[time_col].dt.month 
                    data['day']=data[time_col].dt.day
                    data['weekofyear'] = data[time_col].dt.weekofyear
                    data['dayofweek_num']=data[time_col].dt.dayofweek

                    cor = data.corr()
                    if targ_col != '':
                        cor_targ = cor[targ_col]

                        cor_targ_df = pd.DataFrame({'columns' : cor_targ.index, 'value' : cor_targ.values})
                        cor_targ_df

                        positive_cor = cor_targ_df.loc[cor_targ_df['value'] >= 0.2, 'columns']
                        negative_cor = cor_targ_df.loc[cor_targ_df['value'] <= -0.3, 'columns']

                        l_pos = positive_cor.values
                        l_neg = negative_cor.values
                        print(l_pos)
                        print(l_neg)

                        time_columns = np.concatenate([l_pos, l_neg])

                        time_columns = list(time_columns)

                        time_columns.remove(targ_col)

                        str_time_columns = ','.join(time_columns)

        #                     cur = con.cursor()
        #                     cur.execute("Update queue set time_transform_col = '%s' where id = %d" %(str_time_columns, fileid))
        #                     con.commit()
        #                     cur.close()

                        time_cols = {'vals' : time_col}
                        time_file = open(''+down_path+'/time_vals', 'wb')
                        pickle.dump(time_cols, time_file)
                        time_file.close()

                        if len(l_pos) != 0:
                            for l in l_pos:
                                if l != targ_col:

                                    time_trans_cols = {'vals' : str_time_columns.split(',')}
                                    time_trans_file = open(''+down_path+'/time_trans_vals', 'wb')
                                    pickle.dump(time_trans_cols, time_trans_file)
                                    time_trans_file.close()
                                        
                                    #lags
                                    data['lag_transformer_'+l+'_1'] = data[l].shift(1)
                                    data['lag_transformer_'+l+'_2'] = data[l].shift(2)
                                    data['lag_transformer_'+l+'_3'] = data[l].shift(3)
                                    data['lag_transformer_'+l+'_4'] = data[l].shift(4)
                                    data['lag_transformer_'+l+'_5'] = data[l].shift(5)
                                    data['lag_transformer_'+l+'_6'] = data[l].shift(6)
                                    data['lag_transformer_'+l+'_7'] = data[l].shift(7)
                                    data['lag_transformer_'+l+'_14'] = data[l].shift(14)
                                    data['lag_transformer_'+l+'_21'] = data[l].shift(21)
                                    data['lag_transformer_'+l+'_28'] = data[l].shift(28)

                                    #lags interaction
                                    data['lag_inter_'+l+'_12'] = data['lag_transformer_'+l+'_2'] - data['lag_transformer_'+l+'_1']
                                    data['lag_inter_'+l+'_23'] = data['lag_transformer_'+l+'_3'] - data['lag_transformer_'+l+'_2']
                                    data['lag_inter_'+l+'_34'] = data['lag_transformer_'+l+'_4'] - data['lag_transformer_'+l+'_3']
                                    data['lag_inter_'+l+'_45'] = data['lag_transformer_'+l+'_5'] - data['lag_transformer_'+l+'_4']
                                    data['lag_inter_'+l+'_56'] = data['lag_transformer_'+l+'_6'] - data['lag_transformer_'+l+'_5']
                                    data['lag_inter_'+l+'_67'] = data['lag_transformer_'+l+'_7'] - data['lag_transformer_'+l+'_6']
                                    data['lag_inter_'+l+'_714'] = data['lag_transformer_'+l+'_14'] - data['lag_transformer_'+l+'_7']
                                    data['lag_inter_'+l+'_1421'] = data['lag_transformer_'+l+'_21'] - data['lag_transformer_'+l+'_14']
                                    data['lag_inter_'+l+'_2128'] = data['lag_transformer_'+l+'_28'] - data['lag_transformer_'+l+'_21']

                                    #lags aggregate
                                    #mean
                                    data['lag_aggregate_mean_'+l+'_7'] = data[l].shift(7).mean()
                                    data['lag_aggregate_mean_'+l+'_14'] = data[l].shift(14).mean()
                                    data['lag_aggregate_mean_'+l+'_21'] = data[l].shift(21).mean()
                                    data['lag_aggregate_mean_'+l+'_28'] = data[l].shift(28).mean()
                                    #max
                                    data['lag_aggregate_max_'+l+'_7'] = data[l].shift(7).max()
                                    data['lag_aggregate_max_'+l+'_14'] = data[l].shift(14).max()
                                    data['lag_aggregate_max_'+l+'_21'] = data[l].shift(21).max()
                                    data['lag_aggregate_max_'+l+'_28'] = data[l].shift(28).max()
                                    #stddev
                                    data['lag_aggregate_std_'+l+'_7'] = data[l].shift(7).std()
                                    data['lag_aggregate_std_'+l+'_14'] = data[l].shift(14).std()
                                    data['lag_aggregate_std_'+l+'_21'] = data[l].shift(21).std()
                                    data['lag_aggregate_std_'+l+'_28'] = data[l].shift(28).std()
                                else:
                                    pass
                        elif len(l_neg) != 0:
                            for l in l_neg:
                                if l != targ_col:

                                    time_trans_cols = {'vals' : str_time_columns.split(',')}
                                    time_trans_file = open(''+down_path+'/time_trans_vals', 'wb')
                                    pickle.dump(time_trans_cols, time_trans_file)
                                    time_trans_file.close()

                                    #lags
                                    data['lag_transformer_'+l+'_1'] = data[l].shift(1)
                                    data['lag_transformer_'+l+'_2'] = data[l].shift(2)
                                    data['lag_transformer_'+l+'_3'] = data[l].shift(3)
                                    data['lag_transformer_'+l+'_4'] = data[l].shift(4)
                                    data['lag_transformer_'+l+'_5'] = data[l].shift(5)
                                    data['lag_transformer_'+l+'_6'] = data[l].shift(6)
                                    data['lag_transformer_'+l+'_7'] = data[l].shift(7)
                                    data['lag_transformer_'+l+'_14'] = data[l].shift(14)
                                    data['lag_transformer_'+l+'_21'] = data[l].shift(21)
                                    data['lag_transformer_'+l+'_28'] = data[l].shift(28)

                                    #lags interaction
                                    data['lag_inter_'+l+'_12'] = data['lag_transformer_'+l+'_2'] - data['lag_transformer_'+l+'_1']
                                    data['lag_inter_'+l+'_23'] = data['lag_transformer_'+l+'_3'] - data['lag_transformer_'+l+'_2']
                                    data['lag_inter_'+l+'_34'] = data['lag_transformer_'+l+'_4'] - data['lag_transformer_'+l+'_3']
                                    data['lag_inter_'+l+'_45'] = data['lag_transformer_'+l+'_5'] - data['lag_transformer_'+l+'_4']
                                    data['lag_inter_'+l+'_56'] = data['lag_transformer_'+l+'_6'] - data['lag_transformer_'+l+'_5']
                                    data['lag_inter_'+l+'_67'] = data['lag_transformer_'+l+'_7'] - data['lag_transformer_'+l+'_6']
                                    data['lag_inter_'+l+'_714'] = data['lag_transformer_'+l+'_14'] - data['lag_transformer_'+l+'_7']
                                    data['lag_inter_'+l+'_1421'] = data['lag_transformer_'+l+'_21'] - data['lag_transformer_'+l+'_14']
                                    data['lag_inter_'+l+'_2128'] = data['lag_transformer_'+l+'_28'] - data['lag_transformer_'+l+'_21']

                                    #lags aggregate
                                    #mean
                                    data['lag_aggregate_mean_'+l+'_7'] = data[l].shift(7).mean()
                                    data['lag_aggregate_mean_'+l+'_14'] = data[l].shift(14).mean()
                                    data['lag_aggregate_mean_'+l+'_21'] = data[l].shift(21).mean()
                                    data['lag_aggregate_mean_'+l+'_28'] = data[l].shift(28).mean()
                                    #max
                                    data['lag_aggregate_max_'+l+'_7'] = data[l].shift(7).max()
                                    data['lag_aggregate_max_'+l+'_14'] = data[l].shift(14).max()
                                    data['lag_aggregate_max_'+l+'_21'] = data[l].shift(21).max()
                                    data['lag_aggregate_max_'+l+'_28'] = data[targ_col].shift(28).max()
                                    #stddev
                                    data['lag_aggregate_std_'+l+'_7'] = data[l].shift(7).std()
                                    data['lag_aggregate_std_'+l+'_14'] = data[l].shift(14).std()
                                    data['lag_aggregate_std_'+l+'_21'] = data[l].shift(21).std()
                                    data['lag_aggregate_std_'+l+'_28'] = data[l].shift(28).std()
                                else:
                                    pass
                        else:
                            pass
                    else:
                        pass
                    data = data.drop(time_col, axis =1)
                else:
                    #date
                    data['year']=data[time_col].dt.year
                    data['quarter'] = data[time_col].dt.quarter
                    data['month']=data[time_col].dt.month 
                    data['day']=data[time_col].dt.day
                    data['weekofyear'] = data[time_col].dt.weekofyear
                    data['dayofweek_num']=data[time_col].dt.dayofweek

                    #time
                    data['Hour'] = data[time_col].dt.hour 
                    data['minute'] = data[time_col].dt.minute
                    data['second'] = data[time_col].dt.second

                    if targ_col != '':
                        cor = data.corr()
                        cor_targ = cor[targ_col]

                        cor_targ_df = pd.DataFrame({'columns' : cor_targ.index, 'value' : cor_targ.values})
                        cor_targ_df

                        positive_cor = cor_targ_df.loc[cor_targ_df['value'] >= 0.2, 'columns']
                        negative_cor = cor_targ_df.loc[cor_targ_df['value'] <= -0.3, 'columns']

                        l_pos = positive_cor.values
                        l_neg = negative_cor.values

                        time_columns = np.concatenate([l_pos, l_neg])

                        time_columns = list(time_columns)

                        time_columns.remove(targ_col)

                        str_time_columns = ','.join(time_columns)

        #                     cur = con.cursor()
        #                     cur.execute("Update queue set time_transform_col = '%s' where id = %d" %(str_time_columns, fileid))
        #                     con.commit()
        #                     cur.close()

                        time_cols = {'vals' : time_col}
                        time_file = open(''+down_path+'/time_vals', 'wb')
                        pickle.dump(time_cols, time_file)
                        time_file.close()

                        if len(l_pos) != 0:
                            for l in l_pos:
                                if l != targ_col:
                                    time_trans_cols = {'vals' : str_time_columns.split(',')}
                                    time_trans_file = open(''+down_path+'/time_trans_vals', 'wb')
                                    pickle.dump(time_trans_cols, time_trans_file)
                                    time_trans_file.close()
                                    #lags
                                    data['lag_transformer_'+l+'_1'] = data[l].shift(1)
                                    data['lag_transformer_'+l+'_2'] = data[l].shift(2)
                                    data['lag_transformer_'+l+'_3'] = data[l].shift(3)
                                    data['lag_transformer_'+l+'_4'] = data[l].shift(4)
                                    data['lag_transformer_'+l+'_5'] = data[l].shift(5)
                                    data['lag_transformer_'+l+'_6'] = data[l].shift(6)
                                    data['lag_transformer_'+l+'_7'] = data[l].shift(7)
                                    data['lag_transformer_'+l+'_14'] = data[l].shift(14)
                                    data['lag_transformer_'+l+'_21'] = data[l].shift(21)
                                    data['lag_transformer_'+l+'_28'] = data[l].shift(28)

                                    #lags interaction
                                    data['lag_inter_'+l+'_12'] = data['lag_transformer_'+l+'_2'] - data['lag_transformer_'+l+'_1']
                                    data['lag_inter_'+l+'_23'] = data['lag_transformer_'+l+'_3'] - data['lag_transformer_'+l+'_2']
                                    data['lag_inter_'+l+'_34'] = data['lag_transformer_'+l+'_4'] - data['lag_transformer_'+l+'_3']
                                    data['lag_inter_'+l+'_45'] = data['lag_transformer_'+l+'_5'] - data['lag_transformer_'+l+'_4']
                                    data['lag_inter_'+l+'_56'] = data['lag_transformer_'+l+'_6'] - data['lag_transformer_'+l+'_5']
                                    data['lag_inter_'+l+'_67'] = data['lag_transformer_'+l+'_7'] - data['lag_transformer_'+l+'_6']
                                    data['lag_inter_'+l+'_714'] = data['lag_transformer_'+l+'_14'] - data['lag_transformer_'+l+'_7']
                                    data['lag_inter_'+l+'_1421'] = data['lag_transformer_'+l+'_21'] - data['lag_transformer_'+l+'_14']
                                    data['lag_inter_'+l+'_2128'] = data['lag_transformer_'+l+'_28'] - data['lag_transformer_'+l+'_21']

                                    #lags aggregate
                                    #mean
                                    data['lag_aggregate_mean_'+l+'_7'] = data[l].shift(7).mean()
                                    data['lag_aggregate_mean_'+l+'_14'] = data[l].shift(14).mean()
                                    data['lag_aggregate_mean_'+l+'_21'] = data[l].shift(21).mean()
                                    data['lag_aggregate_mean_'+l+'_28'] = data[l].shift(28).mean()
                                    #max
                                    data['lag_aggregate_max_'+l+'_7'] = data[l].shift(7).max()
                                    data['lag_aggregate_max_'+l+'_14'] = data[l].shift(14).max()
                                    data['lag_aggregate_max_'+l+'_21'] = data[l].shift(21).max()
                                    data['lag_aggregate_max_'+l+'_28'] = data[l].shift(28).max()
                                    #stddev
                                    data['lag_aggregate_std_'+l+'_7'] = data[l].shift(7).std()
                                    data['lag_aggregate_std_'+l+'_14'] = data[l].shift(14).std()
                                    data['lag_aggregate_std_'+l+'_21'] = data[l].shift(21).std()
                                    data['lag_aggregate_std_'+l+'_28'] = data[l].shift(28).std()
                                else:
                                    pass
                        elif len(l_neg) != 0:
                            for l in l_neg:
                                if l != targ_col:
                                    time_trans_cols = {'vals' : str_time_columns.split(',')}
                                    time_trans_file = open(''+down_path+'/time_trans_vals', 'wb')
                                    pickle.dump(time_trans_cols, time_trans_file)
                                    time_trans_file.close()
                                    #lags
                                    data['lag_transformer_'+l+'_1'] = data[l].shift(1)
                                    data['lag_transformer_'+l+'_2'] = data[l].shift(2)
                                    data['lag_transformer_'+l+'_3'] = data[l].shift(3)
                                    data['lag_transformer_'+l+'_4'] = data[l].shift(4)
                                    data['lag_transformer_'+l+'_5'] = data[l].shift(5)
                                    data['lag_transformer_'+l+'_6'] = data[l].shift(6)
                                    data['lag_transformer_'+l+'_7'] = data[l].shift(7)
                                    data['lag_transformer_'+l+'_14'] = data[l].shift(14)
                                    data['lag_transformer_'+l+'_21'] = data[l].shift(21)
                                    data['lag_transformer_'+l+'_28'] = data[l].shift(28)

                                    #lags interaction
                                    data['lag_inter_'+l+'_12'] = data['lag_transformer_'+l+'_2'] - data['lag_transformer_'+l+'_1']
                                    data['lag_inter_'+l+'_23'] = data['lag_transformer_'+l+'_3'] - data['lag_transformer_'+l+'_2']
                                    data['lag_inter_'+l+'_34'] = data['lag_transformer_'+l+'_4'] - data['lag_transformer_'+l+'_3']
                                    data['lag_inter_'+l+'_45'] = data['lag_transformer_'+l+'_5'] - data['lag_transformer_'+l+'_4']
                                    data['lag_inter_'+l+'_56'] = data['lag_transformer_'+l+'_6'] - data['lag_transformer_'+l+'_5']
                                    data['lag_inter_'+l+'_67'] = data['lag_transformer_'+l+'_7'] - data['lag_transformer_'+l+'_6']
                                    data['lag_inter_'+l+'_714'] = data['lag_transformer_'+l+'_14'] - data['lag_transformer_'+l+'_7']
                                    data['lag_inter_'+l+'_1421'] = data['lag_transformer_'+l+'_21'] - data['lag_transformer_'+l+'_14']
                                    data['lag_inter_'+l+'_2128'] = data['lag_transformer_'+l+'_28'] - data['lag_transformer_'+l+'_21']

                                    #lags aggregate
                                    #mean
                                    data['lag_aggregate_mean_'+l+'_7'] = data[l].shift(7).mean()
                                    data['lag_aggregate_mean_'+l+'_14'] = data[l].shift(14).mean()
                                    data['lag_aggregate_mean_'+l+'_21'] = data[l].shift(21).mean()
                                    data['lag_aggregate_mean_'+l+'_28'] = data[l].shift(28).mean()
                                    #max
                                    data['lag_aggregate_max_'+l+'_7'] = data[l].shift(7).max()
                                    data['lag_aggregate_max_'+l+'_14'] = data[l].shift(14).max()
                                    data['lag_aggregate_max_'+l+'_21'] = data[l].shift(21).max()
                                    data['lag_aggregate_max_'+l+'_28'] = data[l].shift(28).max()
                                    #stddev
                                    data['lag_aggregate_std_'+l+'_7'] = data[l].shift(7).std()
                                    data['lag_aggregate_std_'+l+'_14'] = data[l].shift(14).std()
                                    data['lag_aggregate_std_'+l+'_21'] = data[l].shift(21).std()
                                    data['lag_aggregate_std_'+l+'_28'] = data[l].shift(28).std()
                                else:
                                    pass
                        else:
                            pass
                    else:
                        pass
                    data = data.drop(time_col, axis =1)

            if expertDic[0]['num_inter'] == True:
                data = pd.concat([data, interactDF], axis =1)

            if expertDic[0]['cat_embeds'] == True:
                data = data.apply(pd.to_numeric, errors ='coerce')
            data = data.replace([np.inf, -np.inf], np.nan)
            data = data.fillna(data.mean())

            regex = re.compile(r"\[|\]|<", re.IGNORECASE)
            data.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in data.columns]
            
            if targ_col != '':
                if data[targ_col].dtypes == 'float64':
                    data[targ_col] = data[targ_col].astype(int)
                else:
                    pass

            if targ_col != '':
                classs = data[''+targ_col+'']
                dataa = data.drop(''+targ_col+'', axis = 1)
            else:
                dataa = data

            vector_cols = list(set(vector_cols))
            onehot_cols = list(set(onehot_cols))
            constant_cols = list(set(constant_cols))
            card_cols = list(set(card_cols))
            interactcols = list(set(interactcols))
            dropcols = list(set(expertDic[0]['drop_cols']))

            trans_dic = {"onehotencode" : onehot_cols, "vectorize" : vector_cols, "constant" : constant_cols, "cardinal" : card_cols, "interactions" : interactcols, "target" : [targ_col], "dropped" : dropcols}

            if time_col == '':
                example_df = data_leak[fetch_static_cols(dataa)+onehot_cols+vector_cols]
                print(example_df.columns)
                example_df.iloc[0:11,:].to_csv(''+down_path+'/example.csv', index = False)
            else:
                fsc = set(fetch_static_cols(dataa)) - set(list(['year', 'quarter', 'month', 'day', 'weekofyear', 'dayofweek_num', 'Hour', 'minute', 'second']))
                fsc_ls = list(fsc)
                example_df = data_leak[fsc_ls+onehot_cols+vector_cols+[time_col]]
                example_df.iloc[0:11,:].to_csv(''+down_path+'/example.csv', index = False)

            if standardize == True:
                std = StandardScaler()

                tempstdcols = []
                for i,v in enumerate(data.dtypes.values):
                    if v.name == 'int64' or v.name == 'float64' or v.name == 'int32' or v.name == 'float32' or v.name == 'int16' or v.name == 'float16' or v.name == 'uint8' or v.name == 'uint16':
                        tempstdcols.append(data.columns.tolist()[i])

                if tempstdcols:
                    if targ_col in tempstdcols:
                        tempstdcols.remove(targ_col)

                    dataStd = std.fit_transform(data[tempstdcols])
                    pickle.dump(std, open(down_path+'/scaler.pkl','wb'))
                    data = pd.concat([pd.DataFrame(dataStd, columns = data[tempstdcols].columns.tolist()), data[[i for i in data.columns.tolist() if i not in tempstdcols]]], axis=1)

                trans_dic["standard"] = tempstdcols
            else:
                trans_dic["standard"] = []
                pass

            with open(down_path+'/transform.json', 'w') as fp:
                json.dump(trans_dic, fp)

            dataa.iloc[0:100,:].to_csv(''+down_path+'/preprocessed_data.csv', index = False)
            
            feat_end_pre = time.time()
            feat_eng_pre_time = float((feat_end_pre - start)/60)

            if targ_col != '':
                data_seg(data, targ_col)
                pd.DataFrame(data_seg.x_tr, columns = dataa.columns.tolist()).to_csv(''+down_path+'/train_data_processed.csv', index = False)
            else:
                dataa.to_csv(''+down_path+'/train_data_processed.csv', index = False)
            
            if isalien == 1:
                return data_seg.x_tr, data_seg.x_test, data_seg.y_tr, data_seg.y_test, 'null'
            else:
                pass
                # autoTrainDF = pd.DataFrame(data_seg.x_tr, columns = dataa.columns.tolist())
                # autoTestDF = pd.DataFrame(data_seg.x_test, columns = dataa.columns.tolist())
                # autoTrainDF[targ_col] = data_seg.y_tr
                # autoTestDF[targ_col] = data_seg.y_test
                # vertical_stack = pd.concat([autoTrainDF, autoTestDF], axis=0)
                
            if islocal == 1:
                _helper.fileid = fileid
                _helper.publishbot(data)
            elif islocal == 0:
                data.to_csv(down_path+'/'+tablename+'.csv', index = False)
                ftp_server.put(config_paths.nifi_in_path+'/'+tablename+'.csv', down_path_remote+'/'+tablename+'.csv')

            cur = con.cursor()
            cur.execute("Update queue set status = 2, err = 'AutoFeature engineering Completed' where fileflowid = %d" %(int(fileid)))
            con.commit()
            cur.close()

        except Exception as e:
            cur = con.cursor()
            cur.execute("Update queue set status = -2 ,err = '" + str(e).replace("'","''") + "' where fileflowid=%d" %(int(fileid)))
            con.commit()
            cur.close()
            logger.error('autoFeature transformation crashed', exc_info=True)
            x_tr = pd.DataFrame()
            x_test = pd.DataFrame()
            y_tr = pd.DataFrame()
            y_test = pd.DataFrame()
            return x_tr, x_test, y_tr, y_test, str(e)

def transformDataPredict(fileid = 0, filePath='', classification=1,  standardize=False, downloadPath='.', sourcequery = '', isbeta = 0, isalien = 0, islocal = 1, tablename='', down_path_remote='', language='english'):  
    up_path = filePath
    down_path = downloadPath

    if standardize == 1:
        standardize = True
    elif standardize == 0:
        standardize = False
    else:
        pass

    if classification == 1:
        #try:
        if config_paths.isquery == 0:
            if isbeta==1:                                
                n = sum(1 for line in open(r''+up_path+'')) - 1 #number of records in file (excludes header)
                if n>999:
                    s=999
                else:
                    s=n
                    #s = 999 #desired sample size
                skip = sorted(random.sample(range(1,n+1),n-s)) #the 0-indexed header will not be included in the skip list
                data = pd.read_csv(r''+up_path+'', skiprows=skip,quotechar='"',skipinitialspace=True)
            else:
                data = pd.read_csv(r''+up_path+'',quotechar='"',skipinitialspace=True)
        else:
            if isinstance(sourcequery, pd.DataFrame):
                data = sourcequery
            else:
                data = query_data(sourcequery, isalien, down_path)

        if os.path.exists(down_path+'/transform.json'):
            with open(down_path+'/transform.json', 'r') as fp:
                trans_dic = json.load(fp)
        else:
            pass

        ohe_cols = trans_dic["onehotencode"]
        vec_cols = trans_dic["vectorize"]
        const_cols = trans_dic["constant"]
        card_cols = trans_dic["cardinal"]
        interactcols = trans_dic["interactions"]
        targ_col = trans_dic["target"]
        drop_cols = trans_dic["dropped"]
        tempstd_cols = trans_dic["standard"]

        data.drop(drop_cols, axis=1, inplace=True)

        data_not_preprocessed = data

        #uploading schema
        data_sch = pd.read_csv(r''+down_path+'/schema.csv')

        #Data Validation and Correction
        correct_data = pd.read_csv(r''+down_path+'/example.csv')
        train_fin_data = pd.read_csv(r''+down_path+'/train_data_processed.csv')


        if len(correct_data.columns.tolist()) <= len(data.columns.tolist()):
            data = data[correct_data.columns.tolist()]
            data_sch = data_sch[correct_data.columns.tolist()]
        else:
            pass


        #Dropping null columns
        t_null = data.isnull().sum().reset_index()
        t_null.columns = ['cols', 'sum_null']

        null_col = []
        for row in t_null.values:
            if (row[1]/data.shape[0])*100 >= 50:
                null_col.append(row[0])
            else:
                pass

        data = data.drop(null_col, axis = 1)
        data_sch = data_sch.drop(null_col, axis = 1)

        data_str = data_sch.columns[data_sch.isin(['STRING']).any()]
        data_dt = data_sch.columns[data_sch.isin(['DATETIME']).any()]
        data_mon = data_sch.columns[data_sch.isin(['MONEY']).any()]

        #pred_row = data.shape[0]
        #train_row = training_data.shape[0]

        #data = training_data.append(data, ignore_index = True)

        for col in data.columns:
            data[col] = data[col].fillna(data[col].mode()[0])

        try:
            stop = stopwords.words(language)
        except:
            nltk.download('stopwords')
            stop = stopwords.words(language)

        if os.path.exists(''+down_path+'/time_vals'):
            infile = open(''+down_path+'/time_vals','rb')
            time_dict = pickle.load(infile)
            infile.close()

            time_col = time_dict.get('vals')
        else:
            time_col = 'null'

        for dt_col in data_dt:
            try:
                if dt_col == time_col:
                    data = data[~data[dt_col].isna()]
                    to_date_time(data[dt_col])
                    data[dt_col] = to_date_time.colname
                else:
                    data = data.drop(dt_col, axis = 1)
            except:
                if dt_col not in data.columns.tolist():
                    pass
                else:
                    data = data.drop(dt_col, axis = 1)


        if not ohe_cols:
            pass
        else:
            for o_col in ohe_cols:
                if o_col in data.columns.tolist():
                    encoded_features = pd.get_dummies(data[o_col], prefix = 'OHE'+'_'+o_col)
                    data = data.drop(o_col, axis =1)
                    data = data.join(encoded_features)
                else:
                    pass

        if not vec_cols:
            pass
        else:
            for v_col in vec_cols:
                if v_col in data.columns.tolist():
                    vectorize_text(data[v_col])
                    data[v_col] = vectorize_text.colname
                    data2 = data.loc[:, data.columns.intersection([v_col])]
                    data2 = pd.DataFrame(data2[v_col].values.tolist(), index = data2.index)
                    data2 = data2.add_suffix('_'+v_col+'_vector')
                    data = data.drop(v_col, axis =1)
                    data = data.join(data2)
                else:
                    pass

        if not const_cols:
            pass
        else:
            for col in const_cols:
                if col in data.columns.tolist():
                    data.drop(col, axis=1, inplace=True)

        if not card_cols:
            pass
        else:
            for col in card_cols:
                if col in data.columns.tolist():
                    data.drop(col, axis=1,inplace=True)

        if not interactcols:
            pass
        else:
            interactDF = gen_numeric_interactions(data, interactcols, targ_col)

        data_str = list(set(data_str) - set(ohe_cols))
        data_str = list(set(data_str) - set(vec_cols))

        if time_col == 'null' or time_col == 'None':
            for col in data.columns:
                if data[col].dtypes == "object":
                    l = []
                    df_rand = data.sample(frac=0.15)
                    for row in df_rand[col]:
                        leng = len(str(row))
                        l.append(leng)
                    avg = np.mean(l)
                    maxx = np.max(l,initial=0)
                    uniq = data[col].unique().size
                    if maxx <= 50:
                        alphabet_position_numeric(data[col])
                        data[col] = alphabet_position_numeric.colname
                    else:
                        data = data.drop(col, axis = 1)
        else:
            data = data.sort_values(by = ''+time_col+'')

            for col in data.columns:
                if data[col].dtypes == "object":
                    l = []
                    df_rand = data.sample(frac=0.15)
                    for row in df_rand[col]:
                        leng = len(str(row))
                        l.append(leng)
                    avg = np.mean(l)
                    maxx = np.max(l,initial=0)
                    uniq = data[col].unique().size
                    if maxx <= 50:
                        alphabet_position_numeric(data[col])
                        data[col] = alphabet_position_numeric.colname
                    else:
                        data = data.drop(col, axis = 1)

            if data[time_col].dt.minute.all() == 0 and data[time_col].dt.hour.all() == 0:
                #date
                data['year']=data[time_col].dt.year
                data['quarter'] = data[time_col].dt.quarter
                data['month']=data[time_col].dt.month 
                data['day']=data[time_col].dt.day
                data['weekofyear'] = data[time_col].dt.weekofyear
                data['dayofweek_num']=data[time_col].dt.dayofweek
                data = data.drop(time_col, axis =1)
            else:
                #date
                data['year']=data[time_col].dt.year
                data['quarter'] = data[time_col].dt.quarter
                data['month']=data[time_col].dt.month 
                data['day']=data[time_col].dt.day
                data['weekofyear'] = data[time_col].dt.weekofyear
                data['dayofweek_num']=data[time_col].dt.dayofweek

                #time
                data['Hour'] = data[time_col].dt.hour 
                data['minute'] = data[time_col].dt.minute
                data['second'] = data[time_col].dt.second
                data = data.drop(time_col, axis =1)

        for mon_col in data_mon:
            data[mon_col] = data[mon_col].apply(str)
            money_to_numeric(data[mon_col])
            data[mon_col] = money_to_numeric.colname

        if not interactcols:
            pass
        else:
            data = pd.concat([data, interactDF], axis =1)

        data = data.apply(pd.to_numeric, errors ='coerce')
        data = data.replace([np.inf, -np.inf], np.nan)
        data = data.fillna(data.mean())

        regex = re.compile(r"\[|\]|<", re.IGNORECASE)
        data.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in data.columns]

        f_comp = pd.read_csv(down_path+'/train_data_processed.csv')

        try:
            sub_cols = list(set(f_comp.columns.tolist()) - set(data.columns.tolist()))

            if not sub_cols:
                pass
            else:
                for col in sub_cols:
                    data[col] = 0

            if len(train_fin_data.columns.tolist()) <= len(data.columns.tolist()):
                data = data[train_fin_data.columns.tolist()]
            else:
                pass
        except:
            raise ValueError('The columns of the input dataset do not match the columns of the training dataset')

        if not sub_cols:
            pass
        else:
            for col in sub_cols:
                data[col] = 0

        if standardize == True:
            std = pickle.load(open(down_path+'/scaler.pkl','rb'))
            dataStd = std.transform(data[tempstd_cols])
            data = pd.concat([pd.DataFrame(dataStd, columns = data[tempstd_cols].columns.tolist()), data[[i for i in data.columns.tolist() if i not in tempstd_cols]]], axis=1)
        else:
            pass

        #data = data.iloc[train_row:train_row+pred_row,:]
        data.to_csv(down_path+'/pred_data_processed.csv', index = False)

        if isalien == 1:
            return data
        else:
            pass

        if islocal == 1:
            _helper.fileid = fileid
            _helper.publishbot(data)
        elif islocal == 0:
            data.to_csv(down_path+'/'+tablename+'.csv', index = False)
            ftp_server.put(config_paths.nifi_in_path+'/'+tablename+'.csv', down_path_remote+'/'+tablename+'.csv')

        cur = con.cursor()
        cur.execute("Update queue set status = 2, err = 'AutoFeature engineering Completed' where fileflowid = %d" %int(fileid))
        con.commit()
        cur.close()
        # except Exception as e:
        #     print(e)
        #     cur = con.cursor()
        #     cur.execute("Update queue set status = -2 ,err = '" + str(e).replace("'","''") + "' where fileflowid=%d" %int(fileid))
        #     con.commit()
        #     cur.close()
        #     logger.error('Feature transformation on pre-built autoFeature model crashed', exc_info=True)
        #     data = pd.DataFrame()
        #     return data
            
    elif classification == 0:
        #try:
        if config_paths.isquery == 0:
            if isbeta==1:                                
                n = sum(1 for line in open(r''+up_path+'')) - 1 #number of records in file (excludes header)
                if n>999:
                    s=999
                else:
                    s=n
                    #s = 999 #desired sample size
                skip = sorted(random.sample(range(1,n+1),n-s)) #the 0-indexed header will not be included in the skip list
                data = pd.read_csv(r''+up_path+'', skiprows=skip,quotechar='"',skipinitialspace=True)
            else:
                data = pd.read_csv(r''+up_path+'',quotechar='"',skipinitialspace=True)
        else:
            if isinstance(sourcequery, pd.DataFrame):
                data = sourcequery
            else:
                data = query_data(sourcequery, isalien, down_path)

        if os.path.exists(down_path+'/transform.json'):
            with open(down_path+'/transform.json', 'r') as fp:
                trans_dic = json.load(fp)
        else:
            pass

        ohe_cols = trans_dic["onehotencode"]
        vec_cols = trans_dic["vectorize"]
        const_cols = trans_dic["constant"]
        card_cols = trans_dic["cardinal"]
        interactcols = trans_dic["interactions"]
        targ_col = trans_dic["target"]
        drop_cols = trans_dic["dropped"]
        tempstd_cols = trans_dic["standard"]

        data.drop(drop_cols, axis=1, inplace=True)
            
        data_not_preprocessed = data

        #uploading schema
        data_sch = pd.read_csv(r''+down_path+'/schema.csv')

        #Data Validation and Correction
        correct_data = pd.read_csv(r''+down_path+'/example.csv')
        train_fin_data = pd.read_csv(r''+down_path+'/train_data_processed.csv')

        if len(correct_data.columns.tolist()) <= len(data.columns.tolist()):
            data = data[correct_data.columns.tolist()]
            data_sch = data_sch[correct_data.columns.tolist()]
        else:
            pass

        #Dropping null columns
        t_null = data.isnull().sum().reset_index()
        t_null.columns = ['cols', 'sum_null']

        null_col = []
        for row in t_null.values:
            if (row[1]/data.shape[0])*100 >= 50:
                null_col.append(row[0])
            else:
                pass

        data = data.drop(null_col, axis = 1)
        data_sch = data_sch.drop(null_col, axis = 1)

        data_str = data_sch.columns[data_sch.isin(['STRING']).any()]
        data_dt = data_sch.columns[data_sch.isin(['DATETIME']).any()]
        data_mon = data_sch.columns[data_sch.isin(['MONEY']).any()]

        #training_data = pd.read_csv(r''+down_path+'/train_data_not_processed.csv')

        #pred_row = data.shape[0]
        #train_row = training_data.shape[0]

        #data = training_data.append(data, ignore_index = True)

        for col in data.columns:
            data[col] = data[col].fillna(data[col].mode()[0])

        try:
            stop = stopwords.words(language)
        except:
            nltk.download('stopwords')
            stop = stopwords.words(language)

        if os.path.exists(''+down_path+'/time_vals'):
            infile = open(''+down_path+'/time_vals','rb')
            time_dict = pickle.load(infile)
            infile.close()

            time_col = time_dict.get('vals')
        else:
            time_col = 'null'

        for dt_col in data_dt:
            try:
                if dt_col == time_col:
                    data = data[~data[dt_col].isna()]
                    to_date_time(data[dt_col])
                    data[dt_col] = to_date_time.colname
                else:
                    data = data.drop(dt_col, axis = 1)
            except:
                if dt_col not in data.columns.tolist():
                    pass
                else:
                    data = data.drop(dt_col, axis = 1)


        if not ohe_cols:
            pass
        else:
            for o_col in ohe_cols:
                if o_col in data.columns.tolist():
                    encoded_features = pd.get_dummies(data[o_col], prefix = 'OHE'+'_'+o_col)
                    data = data.drop(o_col, axis =1)
                    data = data.join(encoded_features)
                else:
                    pass

        if not vec_cols:
            pass
        else:
            for v_col in vec_cols:
                if v_col in data.columns.tolist():
                    vectorize_text(data[v_col])
                    data[v_col] = vectorize_text.colname
                    data2 = data.loc[:, data.columns.intersection([v_col])]
                    data2 = pd.DataFrame(data2[v_col].values.tolist(), index = data2.index)
                    data2 = data2.add_suffix('_'+v_col+'_vector')
                    data = data.drop(v_col, axis =1)
                    data = data.join(data2)
                else:
                    pass

        if not const_cols:
            pass
        else:
            for col in const_cols:
                if col in data.columns.tolist():
                    data.drop(col, axis=1, inplace=True)

        if not card_cols:
            pass
        else:
            for col in card_cols:
                if col in data.columns.tolist():
                    data.drop(col, axis=1,inplace=True)

        if not interactcols:
            pass
        else:
            interactDF = gen_numeric_interactions(data, interactcols, targ_col)

        data_str = list(set(data_str) - set(ohe_cols))
        data_str = list(set(data_str) - set(vec_cols))

        
        if time_col == 'null' or time_col == 'None':
            for col in data.columns:
                if data[col].dtypes == "object":
                    l = []
                    df_rand = data.sample(frac=0.15)
                    for row in df_rand[col]:
                        leng = len(str(row))
                        l.append(leng)
                    avg = np.mean(l)
                    maxx = np.max(l,initial=0)
                    uniq = data[col].unique().size
                    if maxx <= 50:
                        alphabet_position_numeric(data[col])
                        data[col] = alphabet_position_numeric.colname
                    else:
                        data = data.drop(col, axis = 1)
        else:
            data = data.sort_values(by = ''+time_col+'')

            for col in data.columns:
                if data[col].dtypes == "object":
                    l = []
                    df_rand = data.sample(frac=0.15)
                    for row in df_rand[col]:
                        leng = len(str(row))
                        l.append(leng)
                    avg = np.mean(l)
                    maxx = np.max(l,initial=0)
                    uniq = data[col].unique().size
                    if maxx <= 50:
                        alphabet_position_numeric(data[col])
                        data[col] = alphabet_position_numeric.colname
                    else:
                        data = data.drop(col, axis = 1)

            if data[time_col].dt.minute.all() == 0 and data[time_col].dt.hour.all() == 0:
                #date
                data['year']=data[time_col].dt.year
                data['quarter'] = data[time_col].dt.quarter
                data['month']=data[time_col].dt.month 
                data['day']=data[time_col].dt.day
                data['weekofyear'] = data[time_col].dt.weekofyear
                data['dayofweek_num']=data[time_col].dt.dayofweek

    #                     cur = con.cursor()
    #                     cur.execute("Select time_transform_col from queue WHERE modelkey ='%s' AND time_transform_col IS NOT NULL order by id desc limit 1 " %modelkey)
    #                     for row in cur:
    #                         time_transform_col = row[0]
    #                     cur.close()
                if os.path.exists(down_path+'/time_trans_vals'):
                    infile = open(''+down_path+'/time_trans_vals','rb')
                    time_trans_dict = pickle.load(infile)
                    infile.close()

                    time_tranform_col_ls = time_trans_dict.get('vals')
                else:
                    time_tranform_col_ls = []

                if len(time_tranform_col_ls) != 0:
                    for l in time_tranform_col_ls:
                        #lags
                        data['lag_transformer_'+l+'_1'] = data[l].shift(1)
                        data['lag_transformer_'+l+'_2'] = data[l].shift(2)
                        data['lag_transformer_'+l+'_3'] = data[l].shift(3)
                        data['lag_transformer_'+l+'_4'] = data[l].shift(4)
                        data['lag_transformer_'+l+'_5'] = data[l].shift(5)
                        data['lag_transformer_'+l+'_6'] = data[l].shift(6)
                        data['lag_transformer_'+l+'_7'] = data[l].shift(7)
                        data['lag_transformer_'+l+'_14'] = data[l].shift(14)
                        data['lag_transformer_'+l+'_21'] = data[l].shift(21)
                        data['lag_transformer_'+l+'_28'] = data[l].shift(28)

                        #lags interaction
                        data['lag_inter_'+l+'_12'] = data['lag_transformer_'+l+'_2'] - data['lag_transformer_'+l+'_1']
                        data['lag_inter_'+l+'_23'] = data['lag_transformer_'+l+'_3'] - data['lag_transformer_'+l+'_2']
                        data['lag_inter_'+l+'_34'] = data['lag_transformer_'+l+'_4'] - data['lag_transformer_'+l+'_3']
                        data['lag_inter_'+l+'_45'] = data['lag_transformer_'+l+'_5'] - data['lag_transformer_'+l+'_4']
                        data['lag_inter_'+l+'_56'] = data['lag_transformer_'+l+'_6'] - data['lag_transformer_'+l+'_5']
                        data['lag_inter_'+l+'_67'] = data['lag_transformer_'+l+'_7'] - data['lag_transformer_'+l+'_6']
                        data['lag_inter_'+l+'_714'] = data['lag_transformer_'+l+'_14'] - data['lag_transformer_'+l+'_7']
                        data['lag_inter_'+l+'_1421'] = data['lag_transformer_'+l+'_21'] - data['lag_transformer_'+l+'_14']
                        data['lag_inter_'+l+'_2128'] = data['lag_transformer_'+l+'_28'] - data['lag_transformer_'+l+'_21']

                        #lags aggregate
                        #mean
                        data['lag_aggregate_mean_'+l+'_7'] = data[l].shift(7).mean()
                        data['lag_aggregate_mean_'+l+'_14'] = data[l].shift(14).mean()
                        data['lag_aggregate_mean_'+l+'_21'] = data[l].shift(21).mean()
                        data['lag_aggregate_mean_'+l+'_28'] = data[l].shift(28).mean()
                        #max
                        data['lag_aggregate_max_'+l+'_7'] = data[l].shift(7).max()
                        data['lag_aggregate_max_'+l+'_14'] = data[l].shift(14).max()
                        data['lag_aggregate_max_'+l+'_21'] = data[l].shift(21).max()
                        data['lag_aggregate_max_'+l+'_28'] = data[l].shift(28).max()
                        #stddev
                        data['lag_aggregate_std_'+l+'_7'] = data[l].shift(7).std()
                        data['lag_aggregate_std_'+l+'_14'] = data[l].shift(14).std()
                        data['lag_aggregate_std_'+l+'_21'] = data[l].shift(21).std()
                        data['lag_aggregate_std_'+l+'_28'] = data[l].shift(28).std()
                else:
                    pass
                data = data.drop(time_col, axis =1)
            else:
                #date
                data['year']=data[time_col].dt.year
                data['quarter'] = data[time_col].dt.quarter
                data['month']=data[time_col].dt.month 
                data['day']=data[time_col].dt.day
                data['weekofyear'] = data[time_col].dt.weekofyear
                data['dayofweek_num']=data[time_col].dt.dayofweek

                #time
                data['Hour'] = data[time_col].dt.hour 
                data['minute'] = data[time_col].dt.minute
                data['second'] = data[time_col].dt.second

    #                     cur = con.cursor()
    #                     cur.execute("Select time_transform_col from queue WHERE modelkey ='%s' AND time_transform_col IS NOT NULL order by id desc limit 1 " %modelkey)
    #                     for row in cur:
    #                         time_transform_col = row[0]
    #                     cur.close()

                if os.path.exists(down_path+'/time_trans_vals'):
                    infile = open(''+down_path+'/time_trans_vals','rb')
                    time_trans_dict = pickle.load(infile)
                    infile.close()

                    time_tranform_col_ls = time_trans_dict.get('vals')
                    print(time_transform_col_ls)
                else:
                    pass                                                    

                if len(time_tranform_col_ls) != 0:
                    for l in time_tranform_col_ls:
                        #lags
                        data['lag_transformer_'+l+'_1'] = data[l].shift(1)
                        data['lag_transformer_'+l+'_2'] = data[l].shift(2)
                        data['lag_transformer_'+l+'_3'] = data[l].shift(3)
                        data['lag_transformer_'+l+'_4'] = data[l].shift(4)
                        data['lag_transformer_'+l+'_5'] = data[l].shift(5)
                        data['lag_transformer_'+l+'_6'] = data[l].shift(6)
                        data['lag_transformer_'+l+'_7'] = data[l].shift(7)
                        data['lag_transformer_'+l+'_14'] = data[l].shift(14)
                        data['lag_transformer_'+l+'_21'] = data[l].shift(21)
                        data['lag_transformer_'+l+'_28'] = data[l].shift(28)

                        #lags interaction
                        data['lag_inter_'+l+'_12'] = data['lag_transformer_'+l+'_2'] - data['lag_transformer_'+l+'_1']
                        data['lag_inter_'+l+'_23'] = data['lag_transformer_'+l+'_3'] - data['lag_transformer_'+l+'_2']
                        data['lag_inter_'+l+'_34'] = data['lag_transformer_'+l+'_4'] - data['lag_transformer_'+l+'_3']
                        data['lag_inter_'+l+'_45'] = data['lag_transformer_'+l+'_5'] - data['lag_transformer_'+l+'_4']
                        data['lag_inter_'+l+'_56'] = data['lag_transformer_'+l+'_6'] - data['lag_transformer_'+l+'_5']
                        data['lag_inter_'+l+'_67'] = data['lag_transformer_'+l+'_7'] - data['lag_transformer_'+l+'_6']
                        data['lag_inter_'+l+'_714'] = data['lag_transformer_'+l+'_14'] - data['lag_transformer_'+l+'_7']
                        data['lag_inter_'+l+'_1421'] = data['lag_transformer_'+l+'_21'] - data['lag_transformer_'+l+'_14']
                        data['lag_inter_'+l+'_2128'] = data['lag_transformer_'+l+'_28'] - data['lag_transformer_'+l+'_21']

                        #lags aggregate
                        #mean
                        data['lag_aggregate_mean_'+l+'_7'] = data[l].shift(7).mean()
                        data['lag_aggregate_mean_'+l+'_14'] = data[l].shift(14).mean()
                        data['lag_aggregate_mean_'+l+'_21'] = data[l].shift(21).mean()
                        data['lag_aggregate_mean_'+l+'_28'] = data[l].shift(28).mean()
                        #max
                        data['lag_aggregate_max_'+l+'_7'] = data[l].shift(7).max()
                        data['lag_aggregate_max_'+l+'_14'] = data[l].shift(14).max()
                        data['lag_aggregate_max_'+l+'_21'] = data[l].shift(21).max()
                        data['lag_aggregate_max_'+l+'_28'] = data[l].shift(28).max()
                        #stddev
                        data['lag_aggregate_std_'+l+'_7'] = data[l].shift(7).std()
                        data['lag_aggregate_std_'+l+'_14'] = data[l].shift(14).std()
                        data['lag_aggregate_std_'+l+'_21'] = data[l].shift(21).std()
                        data['lag_aggregate_std_'+l+'_28'] = data[l].shift(28).std()
                else:
                    pass
                data = data.drop(time_col, axis =1)

        for mon_col in data_mon:
            data[mon_col] = data[mon_col].apply(str)
            money_to_numeric(data[mon_col])
            data[mon_col] = money_to_numeric.colname

        if not interactcols:
            pass
        else:
            data = pd.concat([data, interactDF], axis =1)

        data = data.apply(pd.to_numeric, errors ='coerce')
        data = data.replace([np.inf, -np.inf], np.nan)
        data = data.fillna(data.mean())

        regex = re.compile(r"\[|\]|<", re.IGNORECASE)
        data.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in data.columns]

        f_comp = pd.read_csv(down_path+'/train_data_processed.csv')

        try:
            sub_cols = list(set(f_comp.columns.tolist()) - set(data.columns.tolist()))

            if not sub_cols:
                pass
            else:
                for col in sub_cols:
                    data[col] = 0
            
            if len(train_fin_data.columns.tolist()) <= len(data.columns.tolist()):
                data = data[train_fin_data.columns.tolist()]
            else:
                pass
        except:
            raise ValueError('The columns of the input dataset do not match the columns of the training dataset')

        if standardize == True:
            std = pickle.load(open(down_path+'/scaler.pkl','rb'))
            print(tempstd_cols)
            dataStd = std.transform(data[tempstd_cols])
            data = pd.concat([pd.DataFrame(dataStd, columns = data[tempstd_cols].columns.tolist()), data[[i for i in data.columns.tolist() if i not in tempstd_cols]]], axis=1)
        else:
            pass

        #data = data.iloc[train_row:train_row+pred_row,:]
        data.to_csv(down_path+'/pred_data_processed.csv', index = False)
        
        if isalien == 1:
            return data
        else:
            pass
        
        if islocal == 1:
            _helper.fileid = fileid
            _helper.publishbot(data)
        elif islocal == 0:
            data.to_csv(down_path+'/'+tablename+'.csv', index = False)
            ftp_server.put(config_paths.nifi_in_path+'/'+tablename+'.csv', down_path_remote+'/'+tablename+'.csv')
        
        cur = con.cursor()
        cur.execute("Update queue set status = 2, err = 'AutoFeature engineering Completed' where fileflowid = %d" %int(fileid))
        con.commit()
        cur.close()
        
        return data
        # except Exception as e:
        #     cur = con.cursor()
        #     cur.execute("Update queue set status = -2 ,err = '" + str(e).replace("'","''") + "' where fileflowid=%d" %int(fileid))
        #     con.commit()
        #     cur.close()
        #     logger.error('Feature transformation on pre-built autoFeature model crashed', exc_info=True)
        #     data = pd.DataFrame()
        #     return data