import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor

st.write("""
# Cost Prediction App
This app predicts the **Unit Price**!
""")
st.write('---')

# Loads the Boston House Price Dataset
path=r"C:\Users\praya\Documents\Project EIL\AL1111.csv"
data=pd.read_csv(path,parse_dates=[0],infer_datetime_format= True, dayfirst=True,)

data = data[(data[['ORDER_QTY','DELIVERY_PERIOD']] != 0).all(axis=1)]
data.dropna(subset=["ACT_ISS_DT"],inplace=True)
data['ACT_ISS_DT'] = pd.to_datetime(data['ACT_ISS_DT'], format='%d-%m-%y')
data=data.sort_values(by='ACT_ISS_DT')
data.reset_index(inplace=True)
data.drop(['index'],inplace=True,axis=1)
data.drop(['UOM'],inplace=True,axis=1)
data['TOTAL PRICE']=data['ORDER_QTY']*data['UNIT_PRICE']
data['ACT_ISS_DT_YR'] = data['ACT_ISS_DT'].dt.year
data['ACT_ISS_DT_M'] = data['ACT_ISS_DT'].dt.month
data['ACT_ISS_DT_D'] = data['ACT_ISS_DT'].dt.day

#function to set all the Delivery period to days
def DP(dataframe,column,column2,new_column_name):
    n=dataframe[column].count()
    j=[]
    for i in range(0,n):
        if dataframe[column2][i] == "Weeks":
            j.append(dataframe[column][i]*7)
        elif dataframe[column2][i] =="Months":
            j.append(dataframe[column][i]*30)
        else:
            j.append(dataframe[column][i])
    dataframe.insert(9,new_column_name,j)
DP(data,'DELIVERY_PERIOD','DP_TYPE','Delivery_Period_Days')
data.drop(['DELIVERY_PERIOD'],inplace=True,axis=1)
data.drop(['DP_TYPE'],inplace=True,axis=1)    


#separate the other attributes from the predicting attribute
X = data.drop('UNIT_PRICE',axis=1)
#separte the predicting attribute into Y for model training 
Y = data['UNIT_PRICE']
# handle categorical variable
items=pd.get_dummies(X,drop_first=True)
# dropping extra column
X= X.drop('ITEM',axis=1)
# concatation of independent variables and new cateorical variable.
X=pd.concat([X,items],axis=1)
X=X.drop('VENDOR_CODE',axis=1)
X=X.loc[:,~X.columns.duplicated()]
X=X.drop('TOTAL PRICE',axis=1)
X=X.drop('ACT_ISS_DT',axis=1)



# Sidebar
# Header of Specify Input Parameters
st.sidebar.header('Specify Input Parameters')

def user_input_features():
    Item = st.sidebar.selectbox('ITEM',('ITEM1','ITEM2','ITEM3','ITEM4','ITEM5'))
    Vendor_code = st.sidebar.selectbox('VENDOR_CODE',('T040', 'T050', 'D111', 'S265', 'C041', 'E009', 'T040A', 'T019',
       'A225', 'N003', '~5121', 'S420', 'P686', 'M057', 'G125', 'E188',
       'C209', 'S046', 'P275', 'I594', 'P315', 'T226', 'H644', 'T221',
       '~1228', 'W615', 'S102', 'Y634', 'P340', '~0985', 'P325', 'K218',
       '~0991', '80293', '80290', '3637', '80566', 'J614', 'F190', 'S859',
       'T556', 'G616', '81587', '81641', '27489', '3747', 'U144', '3806'))
    quantity = st.sidebar.slider('ORDER_QTY', X.ORDER_QTY.min(),X.ORDER_QTY.max())
    year = st.sidebar.slider('ACT_ISS_DT_YR', X.ACT_ISS_DT_YR.min(),(X.ACT_ISS_DT_YR.max()+20),X.ACT_ISS_DT_YR.max())
    time_period= st.sidebar.slider('Delivery_Period_Days', X.Delivery_Period_Days.min(),X.Delivery_Period_Days.max())
    month = st.sidebar.slider('ACT_ISS_DT_M',X.ACT_ISS_DT_M.min(),X.ACT_ISS_DT_M.max())
    day = st.sidebar.slider('ACT_ISS_DT_D', X.ACT_ISS_DT_D.min(),X.ACT_ISS_DT_D.max())
       
    data = {'ORDER_QTY': quantity,
                'ACT_ISS_DT_YR': year,
                'Delivery_Period_Days': time_period,
                'ACT_ISS_DT_M' : month,
                'ACT_ISS_DT_D' : day}
    vendor_list=['T040', 'T050', 'D111', 'S265', 'C041', 'E009', 'T040A', 'T019',
           'A225', 'N003', '~5121', 'S420', 'P686', 'M057', 'G125', 'E188',
           'C209', 'S046', 'P275', 'I594', 'P315', 'T226', 'H644', 'T221',
           '~1228', 'W615', 'S102', 'Y634', 'P340', '~0985', 'P325', 'K218',
           '~0991', '80293', '80290', '3637', '80566', 'J614', 'F190', 'S859',
           'T556', 'G616', '81587', '81641', '3747', 'U144', '3806']
    n=len(vendor_list)
    venlis=[]
    
    for i in range(0,n):
        if Vendor_code == vendor_list[i]:
             venlis.append(1)
        else:
            venlis.append(0)
    for i in range(0,n):
        data[vendor_list[i]]=venlis[i]
    item_list=['ITEM2','ITEM3','ITEM4','ITEM5']
    n=len(item_list)
    itelis=[]
         
    for i in range(0,n):
        if Item == item_list[i]:
             itelis.append(1)
        else:
             itelis.append(0)
    for i in range(0,n):
        data[item_list[i]]=itelis[i]
    
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

# Main Panel

# Print specified input parameters
st.header('Specified Input parameters')
st.write(df)
st.write('---')

# importing train_test_split from sklearn
from sklearn.model_selection import train_test_split
# splitting the data
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)
# importing module
from sklearn.linear_model import LinearRegression
# creating an object of LinearRegression class
LR = LinearRegression()
# fitting the training data
LR.fit(x_train,y_train)
# Apply Model to Make Prediction
prediction = LR.predict(df)

st.header('Prediction of PRICE')
st.write(prediction)
st.write('---')



