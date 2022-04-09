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
## פילטר עודכן ב -29.3 : נשארו תורים למטרת חינוך ועבודה ראשי 
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
activity_df.loc[(activity_df['NewTaz_x']<7000)&(activity_df['NewTaz_x']>5999),'NewTaz_x']=41
activity_df.loc[(activity_df['NewTaz_x']<8000)&(activity_df['NewTaz_x']>6999),'NewTaz_x']=42
activity_df.loc[(activity_df['NewTaz_x']<9000)&(activity_df['NewTaz_x']>7999),'NewTaz_x']=43


activity_df['NewTazSource'] = pd.to_numeric(activity_df['NewTazSource'])
activity_df.loc[(activity_df['NewTazSource']<7000)&(activity_df['NewTazSource']>5999),'NewTazSource']=41
activity_df.loc[(activity_df['NewTazSource']<8000)&(activity_df['NewTazSource']>6999),'NewTazSource']=42
activity_df.loc[(activity_df['NewTazSource']<9000)&(activity_df['NewTazSource']>7999),'NewTazSource']=43

#%%

activity_df['TrvlDist']= np.where((activity_df['TrvlDist']<=1.5), 1.5, activity_df['TrvlDist'])
activity_df['TrvlDist']= np.where((activity_df['TrvlDist']>1.5)&(activity_df['TrvlDist']<=3), 3, activity_df['TrvlDist'])
activity_df['TrvlDist']= np.where((activity_df['TrvlDist']>3)&(activity_df['TrvlDist']<=10), 10, activity_df['TrvlDist'])
activity_df['TrvlDist']= np.where((activity_df['TrvlDist']>10)&(activity_df['TrvlDist']<=20), 20, activity_df['TrvlDist'])
activity_df['TrvlDist']= np.where((activity_df['TrvlDist']>20)&(activity_df['TrvlDist']<=30), 30, activity_df['TrvlDist'])
activity_df['TrvlDist']= np.where((activity_df['TrvlDist']>30)&(activity_df['TrvlDist']<=40), 40, activity_df['TrvlDist'])
activity_df['TrvlDist']= np.where((activity_df['TrvlDist']>40)&(activity_df['TrvlDist']<=50), 50, activity_df['TrvlDist'])
activity_df['TrvlDist']= np.where((activity_df['TrvlDist']>50)&(activity_df['TrvlDist']<=60), 60, activity_df['TrvlDist'])
activity_df['TrvlDist']= np.where((activity_df['TrvlDist']>60)&(activity_df['TrvlDist']<=70), 70, activity_df['TrvlDist'])



#%%
### עצירות ביניים 

## נסנן פעילות ראשית ל - 0 ונעיף פעילות בית 

stop_anlysis=activity_df.copy()
stop_anlysis=stop_anlysis[stop_anlysis['Activity']!='Home']


## לפני שאמשיך הוציא את הסטטיסטיקה עבור DPS
anlysis_act_dps = stop_anlysis.groupby('Activity')['finalWF'].sum()
print("dps by numbers=")
print(anlysis_act_dps)

### מסנן על בית בפעילויות משניות. 
stop_anlysis=stop_anlysis[stop_anlysis['mainActivity_x']!=1]
stop_anlysis=stop_anlysis[stop_anlysis['prime']==0]

stop_anlysis
print("isg")
anlysis_i_stops_act = stop_anlysis.groupby('Activity')['finalWF'].sum()
print(anlysis_i_stops_act)
print("     ")

print("imd")
anlysis_i_stops_mode = stop_anlysis.groupby('Mode_x')['finalWF'].sum()
print(anlysis_i_stops_mode)
print("imd-total")
print(stop_anlysis['finalWF'].sum())



#%%%  עצירות ביניים - לפי פעילות

stop_i_anlysis=stop_anlysis.copy()
del stop_anlysis

stop_i_other=stop_i_anlysis[stop_i_anlysis['Activity']=='Other']
print(stop_i_other['finalWF'].sum())
stop_i_shop=stop_i_anlysis[stop_i_anlysis['Activity']=='Shopping']
print(stop_i_shop['finalWF'].sum())
stop_i_work=stop_i_anlysis[stop_i_anlysis['Activity']=='Work']
print(stop_i_work['finalWF'].sum())
stop_i_education=stop_i_anlysis[stop_i_anlysis['Activity']=='Education']
print(stop_i_education['finalWF'].sum())

#%%



am_dist=am[['origin_zone','destination_zone','distance']]
das=das.merge(am_dist,left_on=['prev_stop_zone','stop_zone'],right_on=['origin_zone','destination_zone'],how="left")
#%%
del am_dist
del am


#%%



das_i_stops=das[das['primary_stop']==False]
das_i_stops=das_i_stops[das_i_stops['stop_type']!='Home']

#%% For the Das Actvitiy that  Nane in Distance , Becuse dont have match in cost tables we know the only the  trips inside the zone
#### Assume 500m insidt zones trips distance

values = {"distance": 0.5}
das_i_stops=das_i_stops.fillna(value=values)

#%%%

das_i_stops['distance']= np.where((das_i_stops['distance']<=1.5), 1.5, das_i_stops['distance'])
das_i_stops['distance']= np.where((das_i_stops['distance']>1.5)&(das_i_stops['distance']<=3), 3, das_i_stops['distance'])
das_i_stops['distance']= np.where((das_i_stops['distance']>3)&(das_i_stops['distance']<=10), 10, das_i_stops['distance'])
das_i_stops['distance']= np.where((das_i_stops['distance']>10)&(das_i_stops['distance']<=20), 20, das_i_stops['distance'])
das_i_stops['distance']= np.where((das_i_stops['distance']>20)&(das_i_stops['distance']<=30), 30, das_i_stops['distance'])
das_i_stops['distance']= np.where((das_i_stops['distance']>30)&(das_i_stops['distance']<=40), 40, das_i_stops['distance'])
das_i_stops['distance']= np.where((das_i_stops['distance']>40)&(das_i_stops['distance']<=50), 50, das_i_stops['distance'])
das_i_stops['distance']= np.where((das_i_stops['distance']>50)&(das_i_stops['distance']<=60), 60, das_i_stops['distance'])
das_i_stops['distance']= np.where((das_i_stops['distance']>60)&(das_i_stops['distance']<=70), 70, das_i_stops['distance'])


#%%

das_i_stops_work=das_i_stops[das_i_stops['stop_type']=='Work']
das_i_stops_shop=das_i_stops[das_i_stops['stop_type']=='Shop']
das_i_stops_edu=das_i_stops[das_i_stops['stop_type']=='Education']
das_i_stops_other=das_i_stops[das_i_stops['stop_type']=='Other']

#%% work


das_i_stops_work=das_i_stops_work.groupby(['distance'])['person_id'].count().to_frame()
das_i_stops_work=das_i_stops_work.rename(index={1.5: "<1.5",3: "1.5-3",10: "3-10", 20: "10-20", 30: "20-30", 40: "30-40", 50: "40-50", 60: "50-60", 70: "60-70"})
das_i_stops_work=das_i_stops_work.rename(columns={'person_id': "Simuleted"})



stop_i_work=stop_i_work.groupby(['TrvlDist'])['finalWF'].sum().to_frame()
stop_i_work=stop_i_work.rename(index={1.5: "<1.5",3: "1.5-3",10: "3-10", 20: "10-20", 30: "20-30", 40: "30-40", 50: "40-50", 60: "50-60", 70: "60-70"})
stop_i_work=stop_i_work.rename(columns={'finalWF': "FMS"})
stop_i_work=stop_i_work.reindex(["<1.5", "1.5-3", "3-10",  "10-20", "20-30", "30-40", "40-50", "50-60", "60-70"])

merge_stops_work=pd.merge(das_i_stops_work, stop_i_work, right_index = True,  left_index = True,how='left')


#%% shop

das_i_stops_shop=das_i_stops_shop.groupby(['distance'])['person_id'].count().to_frame()
das_i_stops_shop=das_i_stops_shop.rename(index={1.5: "<1.5",3: "1.5-3",10: "3-10", 20: "10-20", 30: "20-30", 40: "30-40", 50: "40-50", 60: "50-60", 70: "60-70"})
das_i_stops_shop=das_i_stops_shop.rename(columns={'person_id': "Simuleted"})



stop_i_shop=stop_i_shop.groupby(['TrvlDist'])['finalWF'].sum().to_frame()
stop_i_shop=stop_i_shop.rename(index={1.5: "<1.5",3: "1.5-3",10: "3-10", 20: "10-20", 30: "20-30", 40: "30-40", 50: "40-50", 60: "50-60", 70: "60-70"})
stop_i_shop=stop_i_shop.rename(columns={'finalWF': "FMS"})
stop_i_shop=stop_i_shop.reindex(["<1.5", "1.5-3", "3-10",  "10-20", "20-30", "30-40", "40-50", "50-60", "60-70"])

merge_stops_shop=pd.merge(das_i_stops_shop, stop_i_shop, right_index = True,  left_index = True,how='left')


#%% edu

das_i_stops_edu=das_i_stops_edu.groupby(['distance'])['person_id'].count().to_frame()
das_i_stops_edu=das_i_stops_edu.rename(index={1.5: "<1.5",3: "1.5-3",10: "3-10", 20: "10-20", 30: "20-30", 40: "30-40", 50: "40-50", 60: "50-60", 70: "60-70"})
das_i_stops_edu=das_i_stops_edu.rename(columns={'person_id': "Simuleted"})



stop_i_education=stop_i_education.groupby(['TrvlDist'])['finalWF'].sum().to_frame()
stop_i_education=stop_i_education.rename(index={1.5: "<1.5",3: "1.5-3",10: "3-10", 20: "10-20", 30: "20-30", 40: "30-40", 50: "40-50", 60: "50-60", 70: "60-70"})
stop_i_education=stop_i_education.rename(columns={'finalWF': "FMS"})
stop_i_education=stop_i_education.reindex(["<1.5", "1.5-3", "3-10",  "10-20", "20-30", "30-40", "40-50", "50-60", "60-70"])

merge_stops_edu=pd.merge(das_i_stops_edu, stop_i_education, right_index = True,  left_index = True,how='left')


#%% other

das_i_stops_other=das_i_stops_other.groupby(['distance'])['person_id'].count().to_frame()
das_i_stops_other=das_i_stops_other.rename(index={1.5: "<1.5",3: "1.5-3",10: "3-10", 20: "10-20", 30: "20-30", 40: "30-40", 50: "40-50", 60: "50-60", 70: "60-70"})
das_i_stops_other=das_i_stops_other.rename(columns={'person_id': "Simuleted"})


stop_i_other=stop_i_other.groupby(['TrvlDist'])['finalWF'].sum().to_frame()
stop_i_other=stop_i_other.rename(index={1.5: "<1.5",3: "1.5-3",10: "3-10", 20: "10-20", 30: "20-30", 40: "30-40", 50: "40-50", 60: "50-60", 70: "60-70"})
stop_i_other=stop_i_other.rename(columns={'finalWF': "FMS"})
stop_i_other=stop_i_other.reindex(["<1.5", "1.5-3", "3-10",  "10-20", "20-30", "30-40", "40-50", "50-60", "60-70"])

merge_stops_other=pd.merge(das_i_stops_other, stop_i_other, right_index = True,  left_index = True,how='left')

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
SAMPLE_SPREADSHEET_ID = '1oYkBrJcC4bzzU1aMJCSkU2ZdfDDz_ZVTITQReed4fOs'

service = build('sheets', 'v4', credentials=creds)
 # Call the Sheets API
sheet = service.spreadsheets()


#%%

merge_stops_work=merge_stops_work.fillna(0)
merge_stops_shop=merge_stops_shop.fillna(0)
merge_stops_edu=merge_stops_edu.fillna(0)
merge_stops_other=merge_stops_other.fillna(0)




merge_stops_work_list=merge_stops_work.reset_index()
merge_stops_work_list=merge_stops_work_list.values.tolist()

merge_stops_shop_list=merge_stops_shop.reset_index()
merge_stops_shop_list=merge_stops_shop_list.values.tolist()

merge_stops_edu_list=merge_stops_edu.reset_index()
merge_stops_edu_list=merge_stops_edu_list.values.tolist()

merge_stops_other_list=merge_stops_other.reset_index()
merge_stops_other_list=merge_stops_other_list.values.tolist()



#%%

request = service.spreadsheets().values().update(spreadsheetId=SAMPLE_SPREADSHEET_ID, range="imd!A2:D1000", valueInputOption="USER_ENTERED", body={"values":merge_stops_work_list}).execute()

request = service.spreadsheets().values().update(spreadsheetId=SAMPLE_SPREADSHEET_ID, range="imd!K2:M1000", valueInputOption="USER_ENTERED", body={"values":merge_stops_shop_list}).execute()


request = service.spreadsheets().values().update(spreadsheetId=SAMPLE_SPREADSHEET_ID, range="imd!U2:W1000", valueInputOption="USER_ENTERED", body={"values":merge_stops_edu_list}).execute()

request = service.spreadsheets().values().update(spreadsheetId=SAMPLE_SPREADSHEET_ID, range="imd!AE2:AG1000", valueInputOption="USER_ENTERED", body={"values":merge_stops_other_list}).execute()


#%%%







