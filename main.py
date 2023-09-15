
from flask import Flask, render_template, request
import pandas as pd
import joblib
import pickle

# Importacion de librerias
from langchain_experimental.sql import SQLDatabaseChain
from langchain.sql_database import SQLDatabase # Agente encargado de interacturar con BBDD SQL
from langchain.chat_models import ChatOpenAI # Cargamos el modelo de chat

import pandas as pd
from sqlalchemy import create_engine

import VarEntorno
import os
from google.cloud import bigquery  # Para interactuar con BigQuery

app = Flask(__name__, static_folder='Imagenes')

# Configurar la variable de entorno 'GOOGLE_APPLICATION_CREDENTIALS' para apuntar al archivo de credenciales 'key.json'
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'key.json'

# Establecer una conexión con el servicio BigQuery utilizando el cliente de BigQuery
client = bigquery.Client()



# Carga de base de datos
db = SQLDatabase.from_uri("sqlite:///datos.db")
engine = create_engine('sqlite:///datos.db')
df = pd.read_sql_query('SELECT * FROM sampled_data LIMIT 2', engine)

# Cargan las variables de entorno
os.environ["OPENAI_API_KEY"] = VarEntorno.OPENAI_API_KEY

# Creamos el LLM, usando a OpenAI
llm = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo')

# Creamos la consulta autocorrectiva de SQL usando el modelo de OpenAI
cadena = SQLDatabaseChain(llm=llm, database=db, verbose=False)

# Formateamos la consulta
formato = """
Take on the role of expert engineer in SQLite3. You are now in the DataDriven role. The name of the columns of our data tables is: id = OCID = identificacion ; quantity = cantidad = cantidad ganada ; method = metodo ;locality = localidad ;region = region suppliers =proveedor; buyer = comprador; year = año ;month = mes ; internal_type = tipo de contrato ;budget = presupuesto. Now given this relevant information, which you must necessarily consider, interpret the following question from the user and generate a Query for SQLite3. It must be a valid query, without errors and well created. User question:
#{question}
"""

# Creamos una funcion para hacer las consultas
def consulta(input_usuario):
    consulta = formato.format(question=input_usuario)
    print(consulta)

    
    resultado = cadena.run(consulta)
    print(resultado)
    return resultado

@app.route('/ChatbotSQL', methods=['GET', 'POST'])
def chatbot_sql():
    if request.method == 'POST':
        pregunta = request.form['pregunta']
        respuesta = consulta(pregunta)
        return render_template('ChatbotSQL.html', pregunta=pregunta, respuesta=respuesta)
    return render_template('ChatbotSQL.html')






# Cargar los preprocesadores

# Cargar el archivo 'preprocessor.pkl' que contiene el preprocesador
with open('preprocessor.pkl', 'rb') as pkl_file:
    preprocessor = pickle.load(pkl_file)

@app.route('/')
def presentacion():
    return render_template('presentacion.html')

@app.route('/EDA')
def EDA():
    return render_template('EDA.html')


# Ruta para la página principal
@app.route('/ANOMALIAS', methods=['GET', 'POST'])
def ANOMALIAS():
    if request.method == 'POST':
        # Obtener los datos del formulario
        amount = float(request.form['amount'])
        budget = float(request.form['budget'])
        locality = request.form['locality']
        region = request.form['region']
        year = int(request.form['year'])
        month = int(request.form['month'])

        # Crear un DataFrame con los datos proporcionados
        nuevos_datos = {
            'amount': [amount],
            'budget': [budget],
            'locality': [locality],
            'region': [region],
            'year': [year],
            'month': [month]
        }
        nuevos_df = pd.DataFrame(nuevos_datos)
        print(nuevos_df)

        # Crear una nueva columna 'date' en 'nuevos_df' utilizando las columnas 'year' y 'month'
        nuevos_df['date'] = pd.to_datetime(nuevos_df[['year', 'month']].assign(day=1))

        # Utilizar el preprocesador guardado para normalizar 'amount' y 'budget'
        nuevos_df[['amount_normalized', 'budget_normalized']] = preprocessor['scaler'].transform(nuevos_df[['amount', 'budget']])

        # Utilizar el preprocesador guardado para codificar 'locality' y 'region'
        nuevos_df['locality_ec'] = preprocessor['le_locality'].transform(nuevos_df['locality'])
        nuevos_df['region_ec'] = preprocessor['le_region'].transform(nuevos_df['region'])


        # Imprimir las columnas seleccionadas en 'nuevos_df'
        print(nuevos_df)

        # Lista de variables utilizadas en el modelo Isolation Forest
        variables = ['amount_normalized', 'locality_ec', 'region_ec', 'year', 'month']

        valores_unicos =["Menor Cuantía","Contratacion directa","Catálogo electrónico - Mejor oferta","Catálogo electrónico - Compra directa","Catálogo electrónico - Gran compra puja","Comunicación Social – Contratación Directa","Catálogo electrónico - Gran compra mejor oferta","Contratos entre Entidades Públicas o sus subsidiarias","Cotización","Licitación","Lista corta","Licitación de Seguros","Repuestos o Accesorios","Bienes y Servicios únicos","Asesoría y Patrocinio Jurídico","Obra artística, científica o literaria","Transporte de correo interno o internacional","Comunicación Social – Proceso de Selección","Contrataciones con empresas públicas internacionales","Concurso publico","Contratación de Seguros","Contratación Directa por Terminación Unilateral","Concurso Público por Lista Corta Desierta"]

    # Iterar a través de los tipos únicos en 'internal_type'
        for tipo in valores_unicos:

            # Construir el nombre del archivo del modelo correspondiente a 'tipo'
            filename = f'./models/model_{tipo}.pkl'

            # Cargar el modelo Isolation Forest desde el archivo
            iforest_fit = joblib.load(filename)

            # Realizar predicciones de anomalía para los nuevos datos utilizando el modelo
            predictions = iforest_fit.predict(nuevos_df[variables])

            # Crear una nueva columna en 'nuevos_df' para almacenar las predicciones de anomalía
            # El nombre de la columna es {tipo}_is_anomaly_prediction
            nuevos_df[f'{tipo}'] = ['ANOMALIA' if pred == -1 else 'NORMAL' for pred in predictions]
            result_data = nuevos_df[nuevos_df.columns[:]]
            


        # return render_template('result.html', data=nuevos_df[nuevos_df.columns[7:]])
        return render_template('result.html', data=result_data)


    # Si la solicitud es GET, mostrar el formulario
    return render_template('index.html')




if __name__ == '__main__':
    app.run(debug=True)
      

