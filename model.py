import numpy as np 
import pandas as pd  
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import StandardScaler, LabelEncoder  
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import pickle 
import os 

def carga_datos():
    """
    Función para cargar los datos desde un archivo CSV.
    """
    try:
        df = pd.read_csv('data/atletas.csv')
        return df
    except FileNotFoundError:
        print("Error: No se encontró el archivo 'data/atletas.csv'")
        return None
    except Exception as e:
        print(f"Error al cargar los datos: {e}")
        return None

def crear_modelo(df, hidden_layer_sizes=(100,), max_iter=300):
    """
    Crea, entrena y evalúa una red neuronal MLPClassifier.

    Args:
        df (DataFrame): Datos de entrada.
        hidden_layer_sizes (tuple): Estructura de la red neuronal.
        max_iter (int): Número máximo de iteraciones (epochs).

    Returns:
        tuple: (modelo, escalador)
    """
    if df is None:
        return None, None
        
    try:
        print("Valores únicos en 'Clasificación':", df['Clasificación'].unique())

        if df['Clasificación'].dtype == 'object':
            label_encoder = LabelEncoder()
            df['Clasificación_num'] = label_encoder.fit_transform(df['Clasificación'])
            y = df['Clasificación_num']
        else:
            y = df['Clasificación']

        X = df[['Edad', 'Frecuencia Cardiaca Basal (lpm)', 'Volumen Sistólico (ml)']]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"\nPrecisión del modelo: {accuracy:.4f}")

        input_dict = {'Edad': 25, 'Frecuencia Cardiaca Basal (lpm)': 70, 'Volumen Sistólico (ml)': 75}
        X_in = np.array(list(input_dict.values())).reshape(1, -1)
        X_in = scaler.transform(X_in)
        prediccion = model.predict(X_in)
        probabilidades = model.predict_proba(X_in)

        if df['Clasificación'].dtype == 'object':
            categoria_predicha = label_encoder.inverse_transform(prediccion)[0]
            print(f"Predicción para {input_dict}: {categoria_predicha}")
        else:
            print(f"Predicción para {input_dict}: {prediccion[0]}")

        return model, scaler
    except Exception as e:
        print(f"Error al crear el modelo: {e}")
        return None, None

def main():
    print("Cargando datos...")
    df = carga_datos()

    if df is not None:
        print(f"Datos cargados correctamente. Shape: {df.shape}")
        print("\nCreando y evaluando modelo...")
        model, scaler = crear_modelo(df, hidden_layer_sizes=(50, 50), max_iter=500)

        if model is not None and scaler is not None:
            try:
                os.makedirs('app_1v', exist_ok=True)
                with open('app_1v/model_lv.pkl', 'wb') as f:
                    pickle.dump(model, f)
                with open('app_1v/scaler_lv.pkl', 'wb') as g:
                    pickle.dump(scaler, g)
                print("\nModelo y scaler guardados correctamente en la carpeta 'app_1v'")
            except Exception as e:
                print(f"Error al guardar el modelo: {e}")

if __name__ == "__main__":
    main()
