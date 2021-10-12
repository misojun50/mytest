import streamlit as st
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

iris = load_iris()
st.title("붓꽃 예측 웹 앱")

st.sidebar.header("입력값")

def user_input_features():
    sepal_length = st.sidebar.slider('꽃받침(Sepal)길이', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('꽃받침(Sepal) 넓이', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('꽃잎(Petal) length', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('꽃잎(Petal) width', 0.1, 2.5, 0.2)
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader("사용자 입력 파라미터")
st.write(df)

iris = load_iris()
X = iris.data
y = iris.target

model = RandomForestClassifier()
model.fit(X, y)

pred = model.predict(df)
pred_proba = model.predict_proba(df)

st.subheader('클래스 레이블 및 해당 색인 번호')
st.write(iris.target_names)

st.subheader('예측')
st.write(iris.target_names[pred])

st.subheader('예측 확률')
st.write(pred_proba)