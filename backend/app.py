from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from prophet import Prophet

# Crear la aplicación Flask
app = Flask(__name__)

# Habilitar CORS para permitir llamadas desde Flutter (o cualquier frontend)
CORS(app)

# Ruta principal para hacer la predicción
@app.route("/predecir", methods=["POST"])
def predecir():
    # Verifica si se ha subido un archivo
    if 'file' not in request.files:
        return jsonify({"error": "No se ha subido ningún archivo"}), 400

    archivo = request.files['file']

    try:
        # Leer el CSV en un DataFrame de Pandas
        datos = pd.read_csv(archivo)

        # Crear y entrenar el modelo Prophet
        modelo = Prophet()
        modelo.fit(datos)

        # Crear un DataFrame con 7 días futuros
        futuro = modelo.make_future_dataframe(periods=7)
        
        # Generar la predicción
        prediccion = modelo.predict(futuro)

        # Seleccionar solo las columnas que queremos mostrar
        resultado = prediccion[['ds', 'yhat']].tail(7).to_dict(orient='records')

        # Devolver la predicción como JSON
        return jsonify(resultado)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Ejecutar la aplicación
if __name__ == "__main__":
    app.run(debug=True)

