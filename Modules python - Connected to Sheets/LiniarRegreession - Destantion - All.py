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

Rings = pd.DataFrame({'superZone':[1,2,3,4,5,6,7,8], 'ring': ['Core','Inner','Middle','Middle','Middle','Outer','Outer','Outer']})

#%% פילטרים על ה - FMS

## לפי ההגדרות שקבענו 

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


#%%

### עצירות ביניים 

## נסנן פעילות ראשית ל - 0 ונעיף פעילות בית 

stop_anlysis=activity_df.copy()


#%%%  עצירות ביניים - רגרסה נכנסים לפי פעילות 


stop_i_other=stop_anlysis[stop_anlysis['Activity']=='Other']
print(stop_i_other['finalWF'].sum())
stop_i_shop=stop_anlysis[stop_anlysis['Activity']=='Shopping']
print(stop_i_shop['finalWF'].sum())
stop_i_work=stop_anlysis[stop_anlysis['Activity']=='Work']
print(stop_i_work['finalWF'].sum())
stop_i_education=stop_anlysis[stop_anlysis['Activity']=='Education']
print(stop_i_education['finalWF'].sum())



origins_map_other=stop_i_other.groupby('NewTaz_x')['finalWF'].sum()
origins_map_shop=stop_i_shop.groupby('NewTaz_x')['finalWF'].sum()
origins_map_work=stop_i_work.groupby('NewTaz_x')['finalWF'].sum()
origins_map_education=stop_i_education.groupby('NewTaz_x')['finalWF'].sum()

#%%

das_work=das[das['stop_type']=='Work']
das_shop=das[das['stop_type']=='Shop']
das_edu=das[das['stop_type']=='Education']
das_other=das[das['stop_type']=='Other']


origins_map_work_das=das_work.groupby('stop_zone')['person_id'].count()
origins_map_shop_das=das_shop.groupby('stop_zone')['person_id'].count()
origins_map_edu_das=das_edu.groupby('stop_zone')['person_id'].count()
origins_map_other_das=das_other.groupby('stop_zone')['person_id'].count()




merge_stops_work=pd.merge(origins_map_work_das, origins_map_work, right_index = True,  left_index = True,how='left')
merge_stops_shop=pd.merge(origins_map_shop_das, origins_map_shop, right_index = True,  left_index = True,how='left')
merge_stops_edu=pd.merge(origins_map_edu_das, origins_map_education, right_index = True,  left_index = True,how='left')
merge_stops_other=pd.merge(origins_map_other_das, origins_map_other, right_index = True,  left_index = True,how='left')


merge_stops_work=merge_stops_work.fillna(0)
merge_stops_shop=merge_stops_shop.fillna(0)
merge_stops_edu=merge_stops_edu.fillna(0)
merge_stops_other=merge_stops_other.fillna(0)


#%% ISTOPS-work
fig, axe = plt.subplots(nrows=1, ncols=1)



X1 = merge_stops_work.iloc[:, 0].values.reshape(-1, 1)  
Y1 = merge_stops_work.iloc[:, 1].values.reshape(-1, 1) 
linear_regressor1 = LinearRegression()  
linear_regressor1.fit(X1, Y1)  
Y1_pred = linear_regressor1.predict(X1)  

##Plot
axe.scatter(X1, Y1, color='navy')
axe.plot(X1, Y1_pred, color='blue')
axe.scatter(X1[[0,1,2]], Y1[[0,1,2]], color='orange')
axe.set_title('Linear regression -ALLL - Dest  stops to Work  ' ,fontsize=25)
axe.set_xlabel('SimMobility ' , fontsize=25,labelpad=15)
axe.set_ylabel('THS ', fontsize=25,labelpad=15)
axe.set_xlim(0, 17500)
axe.set_ylim(0, 17500)
axe.text(1000, 10000, 'R-squared = %0.2f' % r2_score(Y1, Y1_pred),size=17)
axe.text(1000, 12000, 'Coefficients = %0.2f' % linear_regressor1.coef_,size=17)
axe.tick_params(axis='x', rotation=0,labelsize=18)
axe.tick_params(axis='y', rotation=0,labelsize=18)

fig.show()



#%% ISTOPS-eduction
fig2, axe2 = plt.subplots(nrows=1, ncols=1)



X1 = merge_stops_edu.iloc[:, 0].values.reshape(-1, 1)  
Y1 = merge_stops_edu.iloc[:, 1].values.reshape(-1, 1) 
linear_regressor1 = LinearRegression()  
linear_regressor1.fit(X1, Y1)  
Y1_pred = linear_regressor1.predict(X1)  

##Plot
axe2.scatter(X1, Y1, color='navy')
axe2.plot(X1, Y1_pred, color='blue')
axe2.scatter(X1[[0,1,2]], Y1[[0,1,2]], color='orange')
axe2.set_title('Linear regression-ALLL - Dest  \ stops to Education  ' ,fontsize=25)
axe2.set_xlabel('SimMobility ' , fontsize=25,labelpad=15)
axe2.set_ylabel('THS ', fontsize=25,labelpad=15)
axe2.set_xlim(0, 10000)
axe2.set_ylim(0, 10000)
axe2.text(200, 8000, 'R-squared = %0.2f' % r2_score(Y1, Y1_pred),size=17)
axe2.text(200, 6000, 'Coefficients = %0.2f' % linear_regressor1.coef_,size=17)
axe2.tick_params(axis='x', rotation=0,labelsize=18)
axe2.tick_params(axis='y', rotation=0,labelsize=18)

fig2.show()

#%% ISTOPS-other
fig3, axe3 = plt.subplots(nrows=1, ncols=1)



X1 = merge_stops_other.iloc[:, 0].values.reshape(-1, 1)  
Y1 = merge_stops_other.iloc[:, 1].values.reshape(-1, 1) 
linear_regressor1 = LinearRegression()  
linear_regressor1.fit(X1, Y1)  
Y1_pred = linear_regressor1.predict(X1)  

##Plot
axe3.scatter(X1, Y1, color='navy')
axe3.plot(X1, Y1_pred, color='blue')
axe3.scatter(X1[[0,1,2]], Y1[[0,1,2]], color='orange')
axe3.set_title('Linear regression  -ALLL - Dest  stops to Other ' ,fontsize=25)
axe3.set_xlabel('SimMobility ' , fontsize=25,labelpad=15)
axe3.set_ylabel('THS ', fontsize=25,labelpad=15)
axe3.set_xlim(0, 22000)
axe3.set_ylim(0, 22000)
axe3.text(1000, 10000, 'R-squared = %0.2f' % r2_score(Y1, Y1_pred),size=17)
axe3.text(1000, 12000, 'Coefficients = %0.2f' % linear_regressor1.coef_,size=17)
axe3.tick_params(axis='x', rotation=0,labelsize=18)
axe3.tick_params(axis='y', rotation=0,labelsize=18)

fig3.show()

#%% ISTOPS-shop
fig4, axe4 = plt.subplots(nrows=1, ncols=1)



X1 = merge_stops_shop.iloc[:, 0].values.reshape(-1, 1)  
Y1 = merge_stops_shop.iloc[:, 1].values.reshape(-1, 1) 
linear_regressor1 = LinearRegression()  
linear_regressor1.fit(X1, Y1)  
Y1_pred = linear_regressor1.predict(X1)  

##Plot
axe4.scatter(X1, Y1, color='navy')
axe4.plot(X1, Y1_pred, color='blue')
axe4.scatter(X1[[0,1,2]], Y1[[0,1,2]], color='orange')
axe4.set_title('Linear regression-ALLL - Dest  stops to Shop  ' ,fontsize=25)
axe4.set_xlabel('SimMobility ' , fontsize=25,labelpad=15)
axe4.set_ylabel('THS ', fontsize=25,labelpad=15)
axe4.set_xlim(0, 7500)
axe4.set_ylim(0, 7500)
axe4.text(1000, 6000, 'R-squared = %0.2f' % r2_score(Y1, Y1_pred),size=17)
axe4.text(1000, 7000, 'Coefficients = %0.2f' % linear_regressor1.coef_,size=17)
axe4.tick_params(axis='x', rotation=0,labelsize=18)
axe4.tick_params(axis='y', rotation=0,labelsize=18)

fig4.show()




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
SAMPLE_SPREADSHEET_ID = '1GTLgwMnfO3HEELMujVObgUCutxC68CjTVbArAdIu3nQ'

service = build('sheets', 'v4', credentials=creds)
 # Call the Sheets API
sheet = service.spreadsheets()


#%%
merge_stops_work_list=merge_stops_work.reset_index()
merge_stops_work_list=merge_stops_work_list.values.tolist()

merge_stops_shop_list=merge_stops_shop.reset_index()
merge_stops_shop_list=merge_stops_shop_list.values.tolist()

merge_stops_edu_list=merge_stops_edu.reset_index()
merge_stops_edu_list=merge_stops_edu_list.values.tolist()

merge_stops_other_list=merge_stops_other.reset_index()
merge_stops_other_list=merge_stops_other_list.values.tolist()

#%%

request = service.spreadsheets().values().update(spreadsheetId=SAMPLE_SPREADSHEET_ID, range="All-Stops-Destantion!A2:D1000", valueInputOption="USER_ENTERED", body={"values":merge_stops_work_list}).execute()

request = service.spreadsheets().values().update(spreadsheetId=SAMPLE_SPREADSHEET_ID, range="All-Stops-Destantion!K2:M1000", valueInputOption="USER_ENTERED", body={"values":merge_stops_shop_list}).execute()


request = service.spreadsheets().values().update(spreadsheetId=SAMPLE_SPREADSHEET_ID, range="All-Stops-Destantion!U2:W1000", valueInputOption="USER_ENTERED", body={"values":merge_stops_edu_list}).execute()

request = service.spreadsheets().values().update(spreadsheetId=SAMPLE_SPREADSHEET_ID, range="All-Stops-Destantion!AE2:AG1000", valueInputOption="USER_ENTERED", body={"values":merge_stops_other_list}).execute()



























