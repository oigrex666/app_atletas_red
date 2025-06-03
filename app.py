# archivo 1
import streamlit as st 
import pandas as pd 
import matplotlib.pyplot as plt  
import seaborn as sns
import pickle  
import os 
from sklearn.model_selection import train_test_split 
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

st.set_page_config(page_title="Clasificador de Atletas", page_icon="👨‍🦽")

st.markdown("""
<style>
    .stApp {
        background-color: #17202a;
    }
</style>
""", unsafe_allow_html=True)

def cargar_datos():
    try:
        df = pd.read_csv('data/atletas.csv')
        return df
    except:
        st.error("No se pudo cargar el archivo de datos")
        return None

st.sidebar.title("Menú de Navegación")
pagina = st.sidebar.selectbox("Selecciona una opción:", ["home","Preprocesamiento","Predicción", "Modelo", "Datos", "Métricas"])

st.sidebar.subheader("Variables de entrada")
edad = st.sidebar.slider("Edad", 15, 60, 25)
frecuencia = st.sidebar.slider("Frecuencia Cardíaca (lpm)", 40, 100, 70)
volumen = st.sidebar.slider("Volumen Sistólico (ml)", 50, 200, 75)

st.sidebar.subheader("Hiperparámetros del modelo")
epochs = st.sidebar.slider("Número de Epochs (Iteraciones)", 1, 200, 50, step=1)

# Carga de datos
df = cargar_datos()

if pagina == "Datos":
    st.header("Datos de Atletas")
    if df is not None:
        st.write("Vista previa de los datos:")
        st.dataframe(df)
        st.subheader("Distribución por clase")
        fig, ax = plt.subplots()
        df['Clasificación'].value_counts().plot(kind='bar', ax=ax)
        st.pyplot(fig)

elif pagina == "Preprocesamiento":
    st.header("Preprocesamiento de Datos")
    if df is not None:
        st.write("Datos originales:")
        st.dataframe(df.head())

        X = df[['Edad', 'Frecuencia Cardiaca Basal (lpm)', 'Volumen Sistólico (ml)']]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        df_scaled = pd.DataFrame(X_scaled, columns=X.columns)

        st.write("Datos después de la normalización:")
        st.dataframe(df_scaled.head())
    else:
        st.warning("No hay datos disponibles para preprocesar.")

elif pagina == "Modelo":
    st.header("Entrenamiento del Modelo")
    if df is not None:
        X = df[['Edad', 'Frecuencia Cardiaca Basal (lpm)', 'Volumen Sistólico (ml)']]
        y = df['Clasificación']
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        if st.button("Entrenar Modelo"):
            with st.spinner("Entrenando..."):
                modelo = MLPClassifier(hidden_layer_sizes=(10,), max_iter=epochs, random_state=42)
                modelo.fit(X_train, y_train)
                os.makedirs('modelo', exist_ok=True)
                with open('modelo/clasificador.pkl', 'wb') as f:
                    pickle.dump(modelo, f)
                with open('modelo/scaler.pkl', 'wb') as f:
                    pickle.dump(scaler, f)
                y_pred = modelo.predict(X_test)
                precision = accuracy_score(y_test, y_pred)
                st.success(f"¡Modelo entrenado! Precisión: {precision:.2f}")

elif pagina == "Predicción":
    st.header("Hacer Predicción")
    if os.path.exists('modelo/clasificador.pkl'):
        with open('modelo/clasificador.pkl', 'rb') as f:
            modelo = pickle.load(f)
        with open('modelo/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)

        dato = [[edad, frecuencia, volumen]]
        dato_scaled = scaler.transform(dato)
        prediccion = modelo.predict(dato_scaled)[0]
        st.success(f"Predicción: {prediccion}")

        probabilidades = modelo.predict_proba(dato_scaled)[0]
        st.write("Probabilidad por clase:")
        for i, prob in enumerate(probabilidades):
            st.write(f"Clase {i}: {prob:.2f}")
    else:
        st.warning("No hay modelo entrenado. Ve a la página 'Modelo' para entrenarlo primero.")

elif pagina == "Métricas":
    st.header("Métricas del Modelo")
    if df is not None:
        X = df[['Edad', 'Frecuencia Cardiaca Basal (lpm)', 'Volumen Sistólico (ml)']]
        y = df['Clasificación']
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        modelo = MLPClassifier(hidden_layer_sizes=(10,), max_iter=epochs, random_state=42)
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)

        st.write(f"Precisión: {accuracy_score(y_test, y_pred):.2f}")
        st.write(f"Precisión (Precision Score): {precision_score(y_test, y_pred, average='weighted'):.2f}")
        st.write(f"Recall: {recall_score(y_test, y_pred, average='weighted'):.2f}")
        st.write(f"F1-Score: {f1_score(y_test, y_pred, average='weighted'):.2f}")

        cm = confusion_matrix(y_test, y_pred)
        labels = sorted(y.unique())
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm, xticklabels=labels, yticklabels=labels)
        ax_cm.set_xlabel("Predicción")
        ax_cm.set_ylabel("Real")
        st.pyplot(fig_cm)

elif pagina=='home':
    st.title('Inicio')
    st.write('Esta app te permite predecir si alguien es fondista o velocista en función de las variables edad, Frecuencia Cardíaca y Volumen Sistólico de la persona.')
