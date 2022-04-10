#%%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


path_das=r'C:\Users\gabri\Dropbox\Working_Model_Aimsun_TLV\Das Anlysis April 2022\DataBase\activity_schedule'
das = pd.read_csv(path_das, header=None, names=['person_id', 'tour_no', 'tour_type', 'stop_no', 'stop_type', 'stop_location', 'stop_zone', 'stop_mode', 'primary_stop', 'arrival_time', 'departure_time', 'prev_stop_location',  'prev_stop_zone', 'prev_stop_departure_time', 'drivetrain', 'make', 'model'])

path_fms=r'C:\Users\gabri\Dropbox\PreDay_calibration_01-2022\FMS ANLYSIS\DATABASE\FMS_MARCH_2022.csv'
activity_df = pd.read_csv(path_fms)

am=pd.read_csv(r"C:\Users\gabri\Dropbox\Working_Model_Aimsun_TLV\Das Anlysis April 2022\DataBase\AMCOST.csv",header='infer')



Rings = pd.DataFrame({'superZone':[1,2,3,4,5,6,7,8], 'ring': ['Core','Inner','Middle','Middle','Middle','Outer','Outer','Outer']})

path_fare=r'C:\Users\gabri\Dropbox\Working_Model_Aimsun_TLV\Das Anlysis April 2022\DataBase\fare_key.csv'
fare = pd.read_csv(path_fare)
fare.loc[fare['ring']=='מחוץ למטרופולין','ring']='A'
fare.loc[fare['ring']=='טבעת חיצונית ','ring']='B'
fare.loc[fare['ring']=='טבעת אמצעית ','ring']='C'
fare.loc[fare['ring']=='טבעת פנימית ','ring']='D'

#%% פילטרים על ה -FMS


## סינון מס' 1 - כל מי שגר בטבעת החיצונית התורים שלהם ימחקו


# פילטר מספר  1 - כל מי גר בטבעת החיצונים , כל העצירות שלו עפות מכל הסוגים
activity_df['HomeNewTaz'] = pd.to_numeric(activity_df['HomeNewTaz'])

activity_df['superZone'] = (activity_df.HomeNewTaz/1000).apply(np.floor)
activity_df = activity_df.merge(Rings, on='superZone', how='left')
activity_df=activity_df.rename(columns={'ring': "HOME Ring"})
activity_df=activity_df[activity_df['HOME Ring']!='Outer']
print(activity_df['finalWF'].sum())

# פליטר מספר 2 - נסיעות לחוץ המטרופולין ומתוך המטרופולין עפות
activity_df = activity_df[~((activity_df['NewTazSource']=='--')|(activity_df['NewTaz_x']=='--')|(activity_df['NewTazSource'].isna())|(activity_df['NewTaz_x'].isna()))]
activity_df['NewTaz_x'] = pd.to_numeric(activity_df['NewTaz_x'])
print(activity_df['finalWF'].sum())

# פליטר מספר 3 - כל הנסיעות הנמצאות בתורים שראשי של התור נמצא באותו אזור כמו המתגוררים עפות
activity_df=activity_df[activity_df['PrimeNewTaz']!='--'] ##  פילטר 3.1 נוסף תורים שהפעילות המרכזית שלהם מחוץ למטרופולין
print(activity_df['finalWF'].sum())
## פילטר עודכן ב -29.3 : נשארו תורים למטרת חינוך ועבודה ראשי על אף שהם פנימיים  
activity_df['PrimeNewTaz'] = pd.to_numeric(activity_df['PrimeNewTaz'])
activity_df=activity_df[(activity_df['HomeNewTaz']!=activity_df['PrimeNewTaz'])|(activity_df['Tour Activity name']=='Education')|((activity_df['Tour Activity name']=='Work')&(activity_df['mainActivity_x']==2))]
print(activity_df['finalWF'].sum())


## פילטר מספר 4 - לחלק מהעצירות הnשניות שנשארו יכול להיות מצב שהעצירה היא או באזור של פעילות ראשית או באזור של מגורים 
## לכן העיף אותם , פעילות ראשית משאיר גם 
## משאיר גם פעילויות שמטרתם בית 
activity_df=activity_df[((activity_df['NewTaz_x']!=activity_df['HomeNewTaz'])&(activity_df['prime']==0))|(activity_df['prime']==1)|(activity_df['mainActivity_x']==1)]
activity_df=activity_df[((activity_df['NewTaz_x']!=activity_df['PrimeNewTaz'])&(activity_df['prime']==0))|(activity_df['prime']==1)|(activity_df['mainActivity_x']==1)]

print(activity_df['finalWF'].sum())


#### פילטר 5- על האמצעים שאין אצלנו 

activity_df.loc[activity_df['numAccomp']=='--' , 'numAccomp'] = 0
activity_df.loc[activity_df['numAccomp']=='5+' , 'numAccomp'] = 5
activity_df['numAccomp'] = pd.to_numeric(activity_df['numAccomp'])

activity_df =activity_df[activity_df['Mode_x']!='--']
activity_df['Mode_x'] = pd.to_numeric(activity_df['Mode_x'])
activity_df=activity_df[activity_df['Mode_x'].isin([1,2,3,4,7,8,9,10])==True]
activity_df.loc[activity_df['Mode_x']==7 , 'Mode_x'] = 'Car Driver'
activity_df.loc[(activity_df['Mode_x']==8)&(activity_df['numAccomp']<2) , 'Mode_x'] = 'Car Sharing 2'
activity_df.loc[(activity_df['Mode_x']==8)&(activity_df['numAccomp']>1) , 'Mode_x'] = 'Car Sharing 3'
activity_df.loc[activity_df['Mode_x']==9 , 'Mode_x'] = 'Motorcycle'
activity_df.loc[activity_df['Mode_x']==10 , 'Mode_x'] = 'PrivateBus'
activity_df.loc[activity_df['Mode_x']==1 , 'Mode_x'] = 'Walk'
activity_df.loc[activity_df['Mode_x']==2 , 'Mode_x'] = 'Bike'
activity_df.loc[activity_df['Mode_x']==3 , 'Mode_x'] = 'BusTravel'
activity_df.loc[activity_df['Mode_x']==4 , 'Mode_x'] = 'Taxi'

print(activity_df['finalWF'].sum())


activity_df['Activity'] = np.where(activity_df['mainActivity_x'].isin([2,5]),'Work',np.where(activity_df['mainActivity_x']==3,'Education',np.where(activity_df['mainActivity_x']==4,'Shopping', 'Other')))
activity_df['Activity'] = np.where(activity_df['mainActivity_x'].isin([0,1]),'Home',activity_df['Activity'])


activity_df['NewTaz_x'] = pd.to_numeric(activity_df['NewTaz_x'])

activity_df['NewTazSource'] = pd.to_numeric(activity_df['NewTazSource'])

#%%

h=pd.read_csv(r"C:\Lab\Data_base\Hagorot.csv",header='infer')



activity_df=activity_df.merge(h,left_on='NewTazSource',right_on='TAZV41',how='left')
activity_df=activity_df.rename(columns={"FLAG": "From"})

activity_df=activity_df.merge(h,left_on='NewTaz_x',right_on='TAZV41',how='left')
activity_df=activity_df.rename(columns={"FLAG": "To"})






activity_df=activity_df.merge(fare,left_on='NewTazSource',right_on='zone',how='left')
activity_df=activity_df.rename(columns={"ring": "From fare"})

activity_df=activity_df.merge(fare,left_on='NewTaz_x',right_on='zone',how='left')
activity_df=activity_df.rename(columns={"ring": "To fare"})


#%% Anlysis of all 

from datetime import datetime, timedelta, time

activity_df['Arrive_x'] = activity_df.Arrive_x.apply(pd.to_datetime) 
activity_df['Round_Time']= pd.to_datetime(activity_df['Arrive_x']).dt.floor('30T').dt.time

fms_matrix=activity_df[(activity_df['Round_Time']>=time(6,00))&(activity_df['Round_Time']<=time(20,30))]

fms_matrix=fms_matrix.groupby(['From','To'])['finalWF'].sum()
fms_matrix = fms_matrix.unstack(level=-1)

fms_matrix_3=activity_df[(activity_df['Round_Time']>=time(6,00))&(activity_df['Round_Time']<=time(20,30))]
fms_matrix_3=fms_matrix_3.groupby(['From fare','To fare'])['finalWF'].sum()
fms_matrix_3 = fms_matrix_3.unstack(level=-1)


#%%

h=pd.read_csv(r"C:\Lab\Data_base\Hagorot.csv",header='infer')

das=das[(das['prev_stop_departure_time']>5.75)&(das['prev_stop_departure_time']<20.25)]
das=das.merge(h,left_on='prev_stop_location',right_on='TAZV41',how='left')
das=das.rename(columns={"FLAG": "From"})

das=das.merge(h,left_on='stop_location',right_on='TAZV41',how='left')
das=das.rename(columns={"FLAG": "To"})




das=das.merge(fare,left_on='prev_stop_location',right_on='zone',how='left')
das=das.rename(columns={"ring": "From fare"})

das=das.merge(fare,left_on='stop_location',right_on='zone',how='left')
das=das.rename(columns={"ring": "To fare"})




das_gruped=das.groupby(['From','To'])['model'].count()
matrix = das_gruped.unstack(level=-1)

das_gruped_3=das.groupby(['From fare','To fare'])['model'].count()
matrix_3 = das_gruped_3.unstack(level=-1)


#%%  התקשרות אם API של GOOGLE SHEETS כדי שה-CANVA  יעדכו את הגרפים אוטומטית

from google.oauth2 import service_account
from googleapiclient.discovery import build



## הקובץ הרשאות שיצרתי באתר של גוגל
SERVICE_ACCOUNT_FILE = r'C:\Users\gabri\Dropbox\Working_Model_Aimsun_TLV\Das Anlysis April 2022\Modules python - Connected to Sheets\anlysislab-ae811a79b83b.json'
SCOPES = ['https://www.googleapis.com/auth/spreadsheets']


creds = None
creds = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES)

## ה-ID של הגיליון
SAMPLE_SPREADSHEET_ID = '1ENsnRgRXXntSAsLb4WcSEiRfkqhaN5poPnh8iS_TN4A'

service = build('sheets', 'v4', credentials=creds)
 # Call the Sheets API
sheet = service.spreadsheets()


#%%

matrix_list=fms_matrix.reset_index()
matrix_list=matrix_list.values.tolist()
request = service.spreadsheets().values().update(spreadsheetId=SAMPLE_SPREADSHEET_ID, range="All!A4:E7", valueInputOption="USER_ENTERED", body={"values":matrix_list}).execute()


matrix_list=matrix.reset_index()
matrix_list=matrix_list.values.tolist()
request = service.spreadsheets().values().update(spreadsheetId=SAMPLE_SPREADSHEET_ID, range="All!A10:E13", valueInputOption="USER_ENTERED", body={"values":matrix_list}).execute()



#%% Anlysys of car and texi to compere to seker hagurot hizim
 


fms_matrix_2=activity_df[(activity_df['Round_Time']>=time(6,00))&(activity_df['Round_Time']<=time(20,30))]

fms_matrix_2=fms_matrix_2[(fms_matrix_2['Mode_x']=='Car Driver')|(fms_matrix_2['Mode_x']=='Taxi')]
fms_matrix_4=fms_matrix_2.copy()

fms_matrix_2=fms_matrix_2.groupby(['From','To'])['finalWF'].sum()
fms_matrix_2 = fms_matrix_2.unstack(level=-1)


fms_matrix_4=fms_matrix_4.groupby(['From fare','To fare'])['finalWF'].sum()
fms_matrix_4 = fms_matrix_4.unstack(level=-1)


#%%

das2=das[(das['stop_mode']=='Car')|(das['stop_mode']=='Taxi')]
das2_gruped=das2.groupby(['From','To'])['model'].count()
matrix_2 = das2_gruped.unstack(level=-1)

das4_gruped=das2.groupby(['From fare','To fare'])['model'].count()
matrix_4 = das4_gruped.unstack(level=-1)

#%%

matrix_list=fms_matrix_2.reset_index()
matrix_list=matrix_list.values.tolist()
request = service.spreadsheets().values().update(spreadsheetId=SAMPLE_SPREADSHEET_ID, range="All!I4:M7", valueInputOption="USER_ENTERED", body={"values":matrix_list}).execute()


matrix_list=matrix_2.reset_index()
matrix_list=matrix_list.values.tolist()
request = service.spreadsheets().values().update(spreadsheetId=SAMPLE_SPREADSHEET_ID, range="All!I10:M13", valueInputOption="USER_ENTERED", body={"values":matrix_list}).execute()

#%%
matrix_list=fms_matrix_3.reset_index()
matrix_list=matrix_list.values.tolist()
request = service.spreadsheets().values().update(spreadsheetId=SAMPLE_SPREADSHEET_ID, range="All!A36:E39", valueInputOption="USER_ENTERED", body={"values":matrix_list}).execute()


matrix_list=matrix_3.reset_index()
matrix_list=matrix_list.values.tolist()
request = service.spreadsheets().values().update(spreadsheetId=SAMPLE_SPREADSHEET_ID, range="All!A42:E45", valueInputOption="USER_ENTERED", body={"values":matrix_list}).execute()

#%%
matrix_list=fms_matrix_4.reset_index()
matrix_list=matrix_list.values.tolist()
request = service.spreadsheets().values().update(spreadsheetId=SAMPLE_SPREADSHEET_ID, range="All!I36:M39", valueInputOption="USER_ENTERED", body={"values":matrix_list}).execute()


matrix_list=matrix_4.reset_index()
matrix_list=matrix_list.values.tolist()
request = service.spreadsheets().values().update(spreadsheetId=SAMPLE_SPREADSHEET_ID, range="All!I42:M45", valueInputOption="USER_ENTERED", body={"values":matrix_list}).execute()



