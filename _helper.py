#Import libraries
import numpy as np
import pandas as pd
import json
from sklearn.preprocessing import LabelEncoder
import psycopg2
#import logging
#import h2o
import psycopg2
import time
from clickhouse_driver import Client
import plotly.graph_objects as go
import plotly.express as px
import os

generated_tablename=''
tablename=""
publish_path=''
publish_path2=''
projectid = 0
userid = 0
#publish_nifi_path=r'D:\eda\opt\cafebot\nifi\in'
publish_nifi_path='/opt/cafebot/nifi/in'
publish_path='/opt/cafebot/nifi/in'
publish_path2='/opt/cafebot/nifi/in'
chart_down_path=''
sql=''
sqlImgRepo=''
sqlheader=''
sql1=''
sql2=''
sqlheader1=''
sqlheader2=''
con = psycopg2.connect(dbname="cafebot", user="postgres", host="localhost", port = "6435", password="cafebot@2020")
#con = psycopg2.connect(dbname="postgres", user="postgres", host="localhost", port = "5432", password="02042020")
fileid = 0
sql='SELECT * from cafebot.trx limit 100'
def view(df,n=100):
        df = df.iloc[:n,:]
        return df.to_json(orient='records')

def chart(_fig):
    _fig.write_html(chart_down_path + '/chart.html', include_plotlyjs = 'cdn',config={'displayModeBar': False}, full_html = False, default_width='65%', default_height='65%')

def db_close(con):
    if con.closed == 0:
        con.close()
    else:
        pass

def publish(df):
    file_name = str(round(time.time() * 1000))
    file_name ='PT' + file_name +'S'
    #publish_nifi_path_gen= publish_nifi_path + '\\' + file_name + '.csv'
    publish_nifi_path_gen= publish_nifi_path + '/' + file_name + '.csv'
    #print(df)
    df.to_csv(publish_nifi_path_gen,index=False)
    ver = isexist(tablename)
    print(ver)
    
    if ver == 0:
        usr_t_name  = tablename
        insert(fileid, file_name,projectid, usr_t_name, ver,userid) 
    else:
        usr_t_name  = tablename + '_V' + str(ver)
        
        insert(fileid, file_name,projectid, usr_t_name, ver,userid)
    return 1
    
def publishsplit(df,df2):
    try:
        df.to_csv(publish_path, index = False)
        time.sleep(10)
        df2.to_csv(publish_path2, index = False)
        status(fileid,3,'Data Published')
        
    except Exception as e:
        print(e)
        status(fileid,-3,'Data Published Failed')        


def data():    
    try:
        try:
            client = Client('localhost',port=9000,user='default',database='cafebot',compression=True)
            #replace #_# with '
            result, columns = client.execute(sql.replace("#_#","'"), with_column_types=True)
        except:
            raise ValueError('CH database not working properly !')

        df = pd.DataFrame(result, columns=[tuple[0] for tuple in columns])

        if len(os.listdir('/opt/cafebot/bots/temp_files')) == 0:
            pass
        else:
            for f in os.listdir('/opt/cafebot/bots/temp_files'):
                os.remove(os.path.join('/opt/cafebot/bots/temp_files', f))

        if df.shape[0] <= 10000:
            uuid = str(round(time.time() * 1000))
            df.to_csv('/opt/cafebot/bots/temp_files/'+uuid+'.csv', index = False)
            df = pd.read_csv('/opt/cafebot/bots/temp_files/'+uuid+'.csv', low_memory=False)
        else:
            chsize = df.shape[0]*0.9
            uuid = str(round(time.time() * 1000))
            df.to_csv('/opt/cafebot/bots/temp_files/'+uuid+'.csv', index = False, chunksize=int(chsize))

            df = pd.read_csv('/opt/cafebot/bots/temp_files/'+uuid+'.csv', low_memory=False, chunksize=int(chsize))
            df = pd.concat(df, ignore_index=True)
        
        status(fileid,1,'Data loading Completed')
        return(df)
    except Exception as e:
        print(e)
        status(fileid,-1,'Data loading Failed')
        df = pd.DataFrame()
        return df

def dataIMG():    
    try:
        try:
            client = Client('localhost',port=9000,user='default',database='cafebot',compression=True)
            #replace #_# with '        
            result, columns = client.execute(sqlImgRepo.replace("#_#","'"), with_column_types=True)
        except:
            raise ValueError('CH database not working properly !')

        df = pd.DataFrame(result, columns=[tuple[0] for tuple in columns])
        
        status(fileid,1,'Data loading Completed')
        return(df)
    except Exception as e:
        print(e)
        status(fileid,-1,'Data loading Failed')
        df = pd.DataFrame()
        return df

def data_2():    
    try:
        try:
            client = Client('localhost',port=9000,user='default',database='cafebot',compression=True)
            #replace #_# with '        
            result_1, columns_1 = client.execute(sql1.replace('#_#','"'), with_column_types=True)
        except:
            raise ValueError('First DataFrame Loading Failed !')

        df_1 = pd.DataFrame(result_1, columns=[tuple[0] for tuple in columns_1])
        chsize = df_1.shape[0]*0.9
        df_1.to_csv('/opt/cafebot/bots/file_df_1.csv', index = False, chunksize=int(chsize))

        df_1 = pd.read_csv('/opt/cafebot/bots/file_df_1.csv', chunksize=int(chsize))
        df_1 = pd.concat(df_1, ignore_index=True)
        status(fileid,1,'First DataFrame Loaded Successfully !')
        
        
        
        try:
            client = Client('localhost',port=9000,user='default',database='cafebot',compression=True)
            #replace #_# with '        
            result_2, columns_2 = client.execute(sql2.replace('#_#','"'), with_column_types=True)
        except:
            raise ValueError('Second DataFrame Loading Failed !')
            
            
        df_2 = pd.DataFrame(result_2, columns=[tuple[0] for tuple in columns_2])
        chsize = df_2.shape[0]*0.9
        df_2.to_csv('/opt/cafebot/bots/file_df_2.csv', index = False, chunksize=int(chsize))

        df_2 = pd.read_csv('/opt/cafebot/bots/file_df_2.csv', chunksize=int(chsize))
        df_2 = pd.concat(df_2, ignore_index=True)

        status(fileid,1,'Second DataFrame Loaded Successfully !')
        return(df_1,df_2)
        
    except Exception as e:
        #print(e)
        status(fileid,-1,'Data loading Failed')
        df1 = pd.DataFrame()
        return df1
        
    
    
def insert(fileid, file_name, projectid,usr_t_name, ver,userid):
    #projectid = int(str(projectid).replace('"',''))
    #ver = int(str(ver).replace('"',''))
    cur = con.cursor()
    cur.execute("INSERT INTO filename (fileid,projectid, fname, version,userid) VALUES('%s',%d,'%s',%d,%d)"%(file_name,int(projectid),usr_t_name,int(ver),int(userid)))
    cur.execute("UPDATE queue set tablename = '%s' where fileflowid = %s"%(file_name,fileid))
    con.commit()
    cur.close()
    

def status(fileid, st,e):
    e1=str(e).replace("'","''")
    cur = con.cursor()
    cur.execute("UPDATE queue SET status = "+str(st)+",err = '"+e1+"',updated= now() where fileflowid=%s" %fileid)
    con.commit()
    cur.close()
    return 1
    
def pred_exec(mojofileid, fileflowid):
    cur = con.cursor()
    sql = "select ff.fileid,ff.modelname,qe.tablename from fileflow ff inner join queue qe on ff.id=qe.fileflowid where ff.id=%s" %fileflowid
    cur.execute(sql)
    filename=""
    tablename=""
    for row in cur:
        filename = row[1]
        tablename = row[2]

    cur.execute("insert into mojoexec(mojofileid,filename,tablename) values ("+str(mojofileid)+",'" + filename + "','" + tablename + "')")
    con.commit()
    cur.close()
    return 1

def increment(version):
    version+=1
    return version


def isexist(userdefinedTablename):
    cur = con.cursor()
    cur.execute("SELECT version from filename where upper(fname) like upper('" +userdefinedTablename+ "_V%') or upper(fname) like upper('" +userdefinedTablename+ "')" + " order by version desc limit 1");
    version = cur.fetchone()
    con.commit()
    cur.close()
    if version == None:
        ver = 0
    else:
        for i in version:
            ver = increment(i)
            #print(ver)
    
    return ver

def publishbot(df):
    cur = con.cursor()
    cur.execute("Select tablename from queue where fileflowid=%s" %fileid);
    con.commit()
    for row in cur:
        filname = row[0]
    df.to_csv(publish_nifi_path+ '/' +filname+ '.csv', index = False)
    cur.close()

def updateshiftDetails(dictF, fileid):
    print(dictF)
    cur = con.cursor()
    cur.execute("UPDATE queue set shiftdict = '%s' where fileflowid=%d" %(dictF, int(fileid)))
    con.commit()
    cur.close()