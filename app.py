import streamlit as st
import pandas as pd
import numpy as np

# Visualization Package
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots

pd.options.display.float_format = '{:,.2f}'.format
import warnings
warnings.filterwarnings('ignore')

# Preprocessing Package
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import RobustScaler, OneHotEncoder, StandardScaler
from category_encoders import BinaryEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error , r2_score

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score, accuracy_score, precision_score, recall_score
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# Pipeline Package
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Model Deployment Package
import pickle
import joblib

# Deployment Package
import streamlit as st

# Read Data
# ______________________________________________________________________________________________________________________________________________________________________________________________________
df_new = pd.read_csv('Deployment_df.csv',index_col=0)
df_org = pd.read_csv('df_final.csv',index_col=0)
pd.options.display.float_format ='{:,.2f}'.format
st.set_page_config (page_title = 'Breast Cancer' , layout = "wide" , page_icon = '🎗️')
st.title("🎗️ Breast Cancer Wisconsin (Diagnostic) 🎗️")

# Sidebar
brief = st.sidebar.checkbox(":red[Brief about Project]")
Planning = st.sidebar.checkbox(":green[About Project]")
About_me = st.sidebar.checkbox(":green[About me]")

if brief:
    st.sidebar.header(":red[Brief about Project]")
    st.sidebar.write("""
    * The Breast Cancer Wisconsin (Diagnostic) 🎗️ dataset is a valuable resource for data science projects related to health and medicine:
        * Dataset Overview:
        * The Breast Cancer Wisconsin (Diagnostic) dataset, also known as the WBC dataset, is a classification dataset.
        * It records measurements from digitized images of fine needle aspirates (FNAs) of breast masses.
        * The primary task associated with this dataset is classification—specifically, predicting whether a breast cancer case is benign or malignant.
    * :red[So let us see the insights 👀.]
    """)
# Planning
if Planning :
    st.sidebar.header(":green[About Project]")
    st.sidebar.subheader ('🎗️ Breast Cancer Wisconsin (Diagnostic) 🎗️')
    st.sidebar.write("""
    * This project during my Bootcamp @ Zerograd (https://zero-grad.com/). 
    * In this Project we have 3 Tabes:
        * Expolration : Exploratory Data Analysis.
        * Custom EDA for Users Expoloration use.
        * Deployment : Model Prediction & Deployment (Web App').

    * Data Source:
        1) Kaggle : https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data
        2) UCT : https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
        

    """)
    st.sidebar.write("""
    * Data Details:
        * Columns : 32 Features
        * Instance : 569 Instance

    * Features selected in the deployment model:
        * Radius_mean
        * Perimeter_mean
        * Area_mean
        * Concavity_mean 
        * Concave_points_mean
        * Radius_worst 
        * Perimeter_worst
        * Area_worst
        * Concavity_worst
    """)
# Aboutme
if About_me :
    st.sidebar.header(":green[About me]")
    st.sidebar.write("""
    - Osama SAAD
    - Certified Data Science - Epsilon AI
    - Infor ERP (EAM/M3) key Business.User | Infor DMS, Assets and System Control Section Head @Ibnsina Pharma
    - LinkedIn: 
        https://www.linkedin.com/in/ossama-ahmed-saad-525785b2
    - Github : 
        https://github.com/OsamaSamnudi
    """)

# Tabs :
Exploration, Custom_EDA, Prediction = st.tabs(['🔎Exploration🔎' , '🔎Custom EDA for User💡' , '🧠Prediction🤖'])

with Exploration:
    with st.container():
        st.header("Exploration 🔎")
        About_Data = st.expander("✅ About Data")
        About_Data.write(f"""
        * **Data Shape:**
            * Instance : {df_org.shape[0]}
            * Features : {df_org.shape[1]}
            """)
        About_Data.dataframe(data = pd.read_csv('AboutData.csv',index_col=0) , use_container_width=True)
        
        Desc_Data = st.expander("✅ Describe Date")
        Desc_Data.dataframe(data = df_org.describe().T , use_container_width=True)
        
        Corr_Data = st.expander("✅ Correlation")
        fig_1 = px.imshow(df_org.select_dtypes(include=np.number).corr() , text_auto=True , color_continuous_scale='RdBu_r' ,width=1300 , height=1300)
        Corr_Data.plotly_chart(fig_1 , use_container_width=True, theme="streamlit")

    
with Custom_EDA:
    with st.container():
        st.header("🔎Custom EDA for User💡")
        with st.container():
            col1, cols1 , col2 = st.columns([50,10,50])
            with col1:
                st.subheader("Bar Chart (Count of Bins & Categorization)📈")
                i = st.selectbox("Select Feature" , ["Select Feature"] +df_org.drop('diagnosis' , axis=1).columns.tolist() )
                if i == 'Select Feature':
                    st.warning("Please Select Feature")
                else:
                    Prep_msk = df_org[['diagnosis',i]]
                    Prep_msk['desc'] = pd.cut(Prep_msk[i], bins=10)
                    msk_cat = Prep_msk.groupby(['diagnosis','desc'])[i].count().reset_index()
                    msk_cat['desc'] = msk_cat['desc'].apply(lambda x: str(x).replace('(','').replace(']','').replace(', ',' to '))
                    fig_1 = px.bar(msk_cat, x='desc', y=i, color='diagnosis', barmode='group', width=800,height=500, title=f'{i}',text_auto=True)
                    st.plotly_chart(fig_1 , use_container_width=True, theme="streamlit")
                
            with col2:
                st.subheader("Bar Chart (Mean of Bins & Categorization)📈")
                i_mean = st.selectbox("Select Feature Mean" , ["Select Feature Mean"] +df_org.drop('diagnosis' , axis=1).columns.tolist() )
                if i_mean == 'Select Feature Mean':
                    st.warning("Please Select Feature")
                else:
                    Prep_msk_1 = df_org[['diagnosis',i_mean]]
                    Prep_msk_1['desc'] = pd.cut(Prep_msk_1[i_mean], bins=10)
                    msk_cat_1 = Prep_msk_1.groupby(['diagnosis','desc'])[i_mean].mean().reset_index()
                    msk_cat_1['desc'] = msk_cat_1['desc'].apply(lambda x: str(x).replace('(','').replace(']','').replace(', ',' to '))
                    fig_2 = px.bar(msk_cat_1, x='desc', y=i_mean, color='diagnosis', barmode='group',width=800,height=500, title=f'{i_mean}',text_auto=True)
                    st.plotly_chart(fig_2 , use_container_width=True, theme="streamlit")
            
            
        with st.container():
            col3, cols2 , col4 = st.columns([50,10,50])
            with col3:
                st.subheader("Histogram📈")
                i_hist = st.selectbox("Select Feature Histogram" , ["Select Feature Histogram"] +df_org.drop('diagnosis' , axis=1).columns.tolist() )
                if i_hist == 'Select Feature Histogram':
                    st.warning("Please Select Feature")
                else:
                    fig_3 = px.histogram(df_org, x=i_hist, color='diagnosis', barmode='group',width=800,height=500, title=f'{i_hist}',text_auto=True)
                    st.plotly_chart(fig_3 , use_container_width=True, theme="streamlit")
            with col4:
                st.subheader("Histogram📈")
                i_box = st.selectbox("Select Feature box" , ["Select Feature box"] +df_org.drop('diagnosis' , axis=1).columns.tolist() )
                if i_box == 'Select Feature box':
                    st.warning("Please Select Feature")
                else:
                    fig_4 = px.box(df_org, x=i_hist, color='diagnosis',width=800,height=500, title=f'{i_box}')
                    st.plotly_chart(fig_4 , use_container_width=True, theme="streamlit")

        with st.container():
            st.subheader("Scatter 📈")
            col5, cols3 , col6 = st.columns([50,10,50])
            with col5:
                x = st.selectbox("Select x Scatter" , ["Select"] +df_org.drop('diagnosis' , axis=1).columns.tolist() )
            with col6:
                y = st.selectbox("Select y Scatter" , ["Select"] +df_org.drop('diagnosis' , axis=1).columns.tolist() )
            if x == y or y == x or  x == 'Select' or y =='Select':
                st.warning("Please Select Feature")
            else:
                fig_5 = px.scatter(df_org,x=x, y=y, color='diagnosis',width=800,height=500, title=f'{x} vs {y}')
                st.plotly_chart(fig_5 , use_container_width=True, theme="streamlit")

            
            
            
    
with Prediction:
    with st.container():
        st.header("🧠 Prediction🤖")
        col1, col2 , col3 = st.columns([50,50,50])
    # Data Collection
        with col1:
            radius_mean = float(st.number_input("radius_mean",0.0,10000.0))
            perimeter_mean =float(st.number_input("perimeter_mean",0.0,10000.0))
            area_mean = float(st.number_input("area_mean",0.0,10000.0))
            concavity_mean = float(st.number_input("concavity_mean",0.0,10000.0))
            concave_points_mean = float(st.number_input("concave_points_mean",0.0,10000.0))

        with col2:
            radius_worst = float(st.number_input("radius_worst",0.0,10000.0))
            perimeter_worst = float(st.number_input("perimeter_worst",0.0,10000.0))
            area_worst = float(st.number_input("area_worst",0.0,10000.0))
            concavity_worst = float(st.number_input("concavity_worst",0.0,10000.0))
            concave_points_worst = float(st.number_input("points_worst",0.0,10000.0))
        with col3:
            st.write("""
                * <span style="font-size:larger; color:pink">**Radius_mean**</span> : معلومات حول حجم الورم
                * <span style="font-size:larger; color:pink">**Perimeter_mean**</span> : متوسط محيط الورم. يوفر رؤى حول شكل الورم ومدى انتشاره
                * <span style="font-size:larger; color:pink">**Area_mean**</span> : يُمثل المنطقة المتوسطة للورم. القيم الأكبر تشير إلى أحجام أكبر للورم
                * <span style="font-size:larger; color:pink">**Concavity_mean**</span> : يقيس شدة المناطق المقعرة (المنخفضة) داخل حدود الورم
                    * القيم الأعلى تشير إلى تجاويف أكثر وضوحًا
                * <span style="font-size:larger; color:pink">**Concave_points_mean**</span> : تشير إلى عدد المناطق المقعرة داخل الورم
                    * هذه النقاط هي التي ينحني فيها حد الورم
                * <span style="font-size:larger; color:pink">**Radius_worst**</span> : ُمثل القيمة الأسوأ (الأكبر) لنصف القطر. يوفر رؤى حول أكبر حجم للورم
                * <span style="font-size:larger; color:pink">**Perimeter_worst**</span> : أسوأ قيمة للمحيط. يعكس الحدود الأكثر انتشارًا للورم
                * <span style="font-size:larger; color:pink">**Area_worst**</span> : أسوأ قيمة للمنطقة. تشير إلى أكبر منطقة للورم
                * <span style="font-size:larger; color:pink">**Concavity_worst**</span> : أسوأ قيمة للتقعر. يشير إلى أكثر المناطق المقعرة وضوحًا
            """, unsafe_allow_html=True)

 
                
        with st.container():
            st.write('📌 Your Select Data:')
            col4, col5, col6 = st.columns([5,100,2])
            with col5:
                N_data = pd.DataFrame({'radius_mean' : [radius_mean],
                                        'perimeter_mean' : [perimeter_mean],
                                        'area_mean' : [area_mean],
                                        'concavity_mean' : [concavity_mean],
                                        'concave_points_mean' : [concave_points_mean],
                                        'radius_worst' : [radius_worst],
                                        'perimeter_worst' : [perimeter_worst],
                                        'area_worst' : [area_worst],
                                        'concavity_worst' : [concavity_worst],
                                        'concave_points_worst' : [concave_points_worst]})
                st.dataframe(N_data)
            if st.button('Predict'):
                Processor_Model = pickle.load(open('Processor.pkl' , 'rb'))
                RF_model = pickle.load(open('RF_Model.pkl' , 'rb'))
                N_test = Processor_Model.transform(N_data)
                Test_Pred = RF_model.predict(N_test)
                if Test_Pred == 'B':
                    Result = 'Benign - ورم حميد'
                    st.balloons()
                    st.markdown(f"""<span style="font-size:larger; color:green">**Patient Result : {Result}**</span>""" , unsafe_allow_html=True)
                else:
                    Result = 'Malignant - ورم غير حميد'
                    st.markdown(f"""<span style="font-size:larger; color:red">**Patient Result : {Result}**</span>""" , unsafe_allow_html=True)

        
        
