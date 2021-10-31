import streamlit as st
from sklearn.tree import DecisionTreeClassifier
import joblib
import pandas as pd

model = joblib.load(open('model/model.pkl', 'rb'))
codes = {0:'У Вас нет COVID-19',
         1:'У Вас COVID-19'}


st.image('site/Coronavirus.jpg')
st.title('Проверка симптомов на COVID-19')
st.subheader('Выберите Ваши симптомы')

st.sidebar.info('Это приложение создано в образовательных целях.')

fever = st.selectbox('Жар', [0,1])
sore_throat = st.selectbox('Горло болит?', [0,1])
breathing_problem = st.selectbox('Затрудненное дыхание', [0,1])
dry_cough = st.selectbox('Сухой кашель', [0,1])
covid_patient = st.selectbox('Был контакт с COVID больным?', [0,1])
public_place = st.selectbox('Работаете в общественных местах?', [0,1])
abroad_travel= st.selectbox('Были за рубежом?', [0,1])
Attended_Large_Gathering = st.selectbox('Посещали массовые собрания?', [0,1])

input_dict = {'abroad_travel' : abroad_travel, 
              'sore_throat' : sore_throat,
              'breathing_problem' : breathing_problem, 
              'Attended_Large_Gathering' : Attended_Large_Gathering,
              'dry_cough' : dry_cough,
              'covid_patient' : covid_patient,
              'fever' : fever,
              'public_place' : public_place
              
}

input_df = pd.DataFrame([input_dict])

if st.sidebar.button("Predict"):
    pred = model.predict(input_df)
    st.sidebar.write(codes[pred[0]])


# st.success('The output is {}'.format())