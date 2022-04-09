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



#%%%   ניתוח תורים

tours_only_fms=activity_df.copy()
tours_only_fms['NewTaz_x']=round(tours_only_fms['NewTaz_x']/100,0)

tours_only_fms=tours_only_fms[tours_only_fms['prime']==1]

tours_only_fms['Tour Mode'] = pd.to_numeric(tours_only_fms['Tour Mode'])


### העמודים של המוד של התורים שונה מהעמודה של המוד של הסטופ לכן נסדר אותה
tours_only_fms=tours_only_fms[tours_only_fms['Tour Mode'].isin([1,2,3,4,7,8,9,10])==True]


tours_only_fms.loc[tours_only_fms['Tour Mode']==7 , 'Tour Mode'] = 'Car Driver'
tours_only_fms.loc[(tours_only_fms['Tour Mode']==8)&(tours_only_fms['numAccomp']<2) , 'Tour Mode'] = 'Car Sharing 2'
tours_only_fms.loc[(tours_only_fms['Tour Mode']==8)&(tours_only_fms['numAccomp']>1) , 'Tour Mode'] = 'Car Sharing 3'

tours_only_fms.loc[tours_only_fms['Tour Mode']==9 , 'Tour Mode'] = 'Motorcycle'
tours_only_fms.loc[tours_only_fms['Tour Mode']==10 , 'Tour Mode'] = 'PrivateBus'

tours_only_fms.loc[tours_only_fms['Tour Mode']==1 , 'Tour Mode'] = 'Walk'
tours_only_fms.loc[tours_only_fms['Tour Mode']==2 , 'Tour Mode'] = 'Bike'
tours_only_fms.loc[tours_only_fms['Tour Mode']==3 , 'Tour Mode'] = 'BusTravel'
tours_only_fms.loc[tours_only_fms['Tour Mode']==4 , 'Tour Mode'] = 'Taxi'


#%% נחלק את התורים לקטגוריות
## הסוג פעילות של התור נמצא בעמודה אחרת מסוג הפעילות בעצירה לשים לב 



tours_only_education=tours_only_fms[tours_only_fms['Tour Activity name']=='Education']
tours_only_other=tours_only_fms[tours_only_fms['Tour Activity name']=='Other']
tours_only_shop=tours_only_fms[tours_only_fms['Tour Activity name']=='Shopping']


## תור למטרת עבודה ושהפעילות הראשית שלו היא עבודה 
tours_only_work_alpha=tours_only_fms[(tours_only_fms['Tour Activity name']=='Work')&(tours_only_fms['mainActivity_x']==2)]

## תור למטרת עבודה ושהפעילות הראשית שלו היא נסיעה שקשורה לעבודה 
tours_only_work_beta=tours_only_fms[(tours_only_fms['Tour Activity name']=='Work')&(tours_only_fms['mainActivity_x']==5)]




#%%%  עצירות ביניים - רגרסה נכנסים לפי פעילות 


print(tours_only_education['finalWF'].sum())
print(tours_only_other['finalWF'].sum())
print(tours_only_work_alpha['finalWF'].sum())
print(tours_only_work_beta['finalWF'].sum())
print(tours_only_shop['finalWF'].sum())



origins_map_other=tours_only_other.groupby('NewTaz_x')['finalWF'].sum()
origins_map_shop=tours_only_shop.groupby('NewTaz_x')['finalWF'].sum()

origins_map_work_alpha=tours_only_work_alpha.groupby('NewTaz_x')['finalWF'].sum()
origins_map_work_beta=tours_only_work_beta.groupby('NewTaz_x')['finalWF'].sum()

origins_map_education=tours_only_education.groupby('NewTaz_x')['finalWF'].sum()

#%%

path_pps=r'C:\Users\gabri\Dropbox\Working_Model_Aimsun_TLV\Das Anlysis April 2022\DataBase\table_individual_by_id_for_preday_5.csv'
ind = pd.read_csv(path_pps)

a1 = das["person_id"].str.split("-", n = 1, expand = True)
das['individual_id']= a1[0]
das["individual_id"] = pd.to_numeric(das["individual_id"])


das = das.merge(ind[['individual_id','work_sla']],left_on ='individual_id',right_on= 'individual_id',how = 'left')


#%%%

das['stop_zone']=round(das['stop_zone']/100,0)


das_tours=das[das['primary_stop']==True]

das_tours_work_alpha=das_tours[(das_tours['tour_type']=='Work')&(das_tours['work_sla']==das_tours['stop_location'])]
das_tours_work_beta=das_tours[(das_tours['tour_type']=='Work')&(das_tours['work_sla']!=das_tours['stop_location'])]

das_tours_shop=das_tours[das_tours['tour_type']=='Shop']
das_tours_edu=das_tours[das_tours['tour_type']=='Education']
das_tours_other=das_tours[das_tours['tour_type']=='Other']


origins_map_work_alpha_das=das_tours_work_alpha.groupby('stop_zone')['person_id'].count()
origins_map_work_beta_das=das_tours_work_beta.groupby('stop_zone')['person_id'].count()
origins_map_shop_das=das_tours_shop.groupby('stop_zone')['person_id'].count()
origins_map_edu_das=das_tours_edu.groupby('stop_zone')['person_id'].count()
origins_map_other_das=das_tours_other.groupby('stop_zone')['person_id'].count()




merge_stops_work_alpha=pd.merge(origins_map_work_alpha_das, origins_map_work_alpha, right_index = True,  left_index = True,how='left')
merge_stops_work_beta=pd.merge(origins_map_work_beta_das, origins_map_work_beta, right_index = True,  left_index = True,how='left')

merge_stops_shop=pd.merge(origins_map_shop_das, origins_map_shop, right_index = True,  left_index = True,how='left')
merge_stops_edu=pd.merge(origins_map_edu_das, origins_map_education, right_index = True,  left_index = True,how='left')
merge_stops_other=pd.merge(origins_map_other_das, origins_map_other, right_index = True,  left_index = True,how='left')


merge_stops_work_alpha=merge_stops_work_alpha.fillna(0)
merge_stops_work_beta=merge_stops_work_beta.fillna(0)
merge_stops_shop=merge_stops_shop.fillna(0)
merge_stops_edu=merge_stops_edu.fillna(0)
merge_stops_other=merge_stops_other.fillna(0)


#%% tours-work alpha
fig, axe = plt.subplots(nrows=1, ncols=1)



X1 = merge_stops_work_alpha.iloc[:, 0].values.reshape(-1, 1)  
Y1 = merge_stops_work_alpha.iloc[:, 1].values.reshape(-1, 1) 
linear_regressor1 = LinearRegression()  
linear_regressor1.fit(X1, Y1)  
Y1_pred = linear_regressor1.predict(X1)  

##Plot
axe.scatter(X1, Y1, color='navy')
axe.plot(X1, Y1_pred, color='blue')
axe.scatter(X1[[0,1,2]], Y1[[0,1,2]], color='orange')
axe.set_title('Linear regression - Tours , Alpha Work - Destenation ' ,fontsize=25)
axe.set_xlabel('SimMobility ' , fontsize=25,labelpad=15)
axe.set_ylabel('THS ', fontsize=25,labelpad=15)
axe.set_xlim(0, 100000)
axe.set_ylim(0, 100000)
axe.text(80000, 70000, 'R-squared = %0.2f' % r2_score(Y1, Y1_pred),size=17)
axe.text(80000, 60000, 'Coefficients = %0.2f' % linear_regressor1.coef_,size=17)
axe.tick_params(axis='x', rotation=0,labelsize=18)
axe.tick_params(axis='y', rotation=0,labelsize=18)

fig.show()



#%% tours-eduction
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
axe2.set_title('Linear regression - Tours , Education- Destenation  ' ,fontsize=25)
axe2.set_xlabel('SimMobility ' , fontsize=25,labelpad=15)
axe2.set_ylabel('THS ', fontsize=25,labelpad=15)
axe2.set_xlim(0, 150000)
axe2.set_ylim(0, 150000)
axe2.text(10000, 100000, 'R-squared = %0.2f' % r2_score(Y1, Y1_pred),size=17)
axe2.text(10000, 120000, 'Coefficients = %0.2f' % linear_regressor1.coef_,size=17)
axe2.tick_params(axis='x', rotation=0,labelsize=18)
axe2.tick_params(axis='y', rotation=0,labelsize=18)

fig2.show()

#%% tours-other
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
axe3.set_title('Linear regression -Tours , Other- Destenation   ' ,fontsize=25)
axe3.set_xlabel('SimMobility ' , fontsize=25,labelpad=15)
axe3.set_ylabel('THS ', fontsize=25,labelpad=15)
axe3.set_xlim(0, 100000)
axe3.set_ylim(0, 100000)
axe3.text(80000, 10500, 'R-squared = %0.2f' % r2_score(Y1, Y1_pred),size=17)
axe3.text(80000, 4500, 'Coefficients = %0.2f' % linear_regressor1.coef_,size=17)
axe3.tick_params(axis='x', rotation=0,labelsize=18)
axe3.tick_params(axis='y', rotation=0,labelsize=18)

fig3.show()

#%%tours-shop
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
axe4.set_title('Linear regression -Tours , Shop- Destenation   ' ,fontsize=25)
axe4.set_xlabel('SimMobility ' , fontsize=25,labelpad=15)
axe4.set_ylabel('THS ', fontsize=25,labelpad=15)
axe4.set_xlim(0, 30000)
axe4.set_ylim(0, 30000)
axe4.text(25000, 15000, 'R-squared = %0.2f' % r2_score(Y1, Y1_pred),size=17)
axe4.text(25000, 10000, 'Coefficients = %0.2f' % linear_regressor1.coef_,size=17)
axe4.tick_params(axis='x', rotation=0,labelsize=18)
axe4.tick_params(axis='y', rotation=0,labelsize=18)

fig4.show()



#%% tours-work beta
fig, axe = plt.subplots(nrows=1, ncols=1)



X1 = merge_stops_work_beta.iloc[:, 0].values.reshape(-1, 1)  
Y1 = merge_stops_work_beta.iloc[:, 1].values.reshape(-1, 1) 
linear_regressor1 = LinearRegression()  
linear_regressor1.fit(X1, Y1)  
Y1_pred = linear_regressor1.predict(X1)  

##Plot
axe.scatter(X1, Y1, color='navy')
axe.plot(X1, Y1_pred, color='blue')
axe.scatter(X1[[0,1,2]], Y1[[0,1,2]], color='orange')
axe.set_title('Linear regression - Tours , Beta Work - Destenation ' ,fontsize=25)
axe.set_xlabel('SimMobility ' , fontsize=25,labelpad=15)
axe.set_ylabel('THS ', fontsize=25,labelpad=15)
axe.set_xlim(0, 20000)
axe.set_ylim(0, 20000)
axe.text(10000, 4000, 'R-squared = %0.2f' % r2_score(Y1, Y1_pred),size=17)
axe.text(10000, 3000, 'Coefficients = %0.2f' % linear_regressor1.coef_,size=17)
axe.tick_params(axis='x', rotation=0,labelsize=18)
axe.tick_params(axis='y', rotation=0,labelsize=18)

fig.show()









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
SAMPLE_SPREADSHEET_ID = '1UoS00Fwiw7JGRWbg_vVjsZiyonwuaKN97Fwmywmerww'

service = build('sheets', 'v4', credentials=creds)
 # Call the Sheets API
sheet = service.spreadsheets()


#%%
merge_stops_work__alpha_list=merge_stops_work_alpha.reset_index()
merge_stops_work__alpha_list=merge_stops_work__alpha_list.values.tolist()

merge_stops_shop_list=merge_stops_shop.reset_index()
merge_stops_shop_list=merge_stops_shop_list.values.tolist()

merge_stops_edu_list=merge_stops_edu.reset_index()
merge_stops_edu_list=merge_stops_edu_list.values.tolist()

merge_stops_other_list=merge_stops_other.reset_index()
merge_stops_other_list=merge_stops_other_list.values.tolist()

merge_stops_work__beta_list=merge_stops_work_beta.reset_index()
merge_stops_work__beta_list=merge_stops_work__beta_list.values.tolist()


#%%

request = service.spreadsheets().values().update(spreadsheetId=SAMPLE_SPREADSHEET_ID, range="Destenation-Tour by zones!A2:D1000", valueInputOption="USER_ENTERED", body={"values":merge_stops_work__alpha_list}).execute()

request = service.spreadsheets().values().update(spreadsheetId=SAMPLE_SPREADSHEET_ID, range="Destenation-Tour by zones!K2:M1000", valueInputOption="USER_ENTERED", body={"values":merge_stops_shop_list}).execute()


request = service.spreadsheets().values().update(spreadsheetId=SAMPLE_SPREADSHEET_ID, range="Destenation-Tour by zones!U2:W1000", valueInputOption="USER_ENTERED", body={"values":merge_stops_edu_list}).execute()

request = service.spreadsheets().values().update(spreadsheetId=SAMPLE_SPREADSHEET_ID, range="Destenation-Tour by zones!AE2:AG1000", valueInputOption="USER_ENTERED", body={"values":merge_stops_other_list}).execute()


request = service.spreadsheets().values().update(spreadsheetId=SAMPLE_SPREADSHEET_ID, range="Destenation-Tour by zones!AO2:AQ1000", valueInputOption="USER_ENTERED", body={"values":merge_stops_work__beta_list}).execute()

































