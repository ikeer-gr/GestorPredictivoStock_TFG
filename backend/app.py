from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from prophet import Prophet

# Crear la aplicación Flask
app = Flask(__name__)
CORS(app)  # Permitir peticiones CORS desde cualquier origen (ajusta si necesitas más seguridad)

@app.route("/predecir", methods=["POST"])
def predecir():
    # Verificar si se subió el archivo
    if 'file' not in request.files:
        return jsonify({"error": "No se ha subido ningún archivo"}), 400

    archivo = request.files['file']

    try:
        # Leer el archivo CSV
        datos = pd.read_csv(archivo)

        # Verificar columnas requeridas
        if not {'ds', 'y'}.issubset(datos.columns):
            return jsonify({"error": "El CSV debe contener las columnas 'ds' (fecha) y 'y' (valor)"}), 400

        # Convertir 'ds' a formato datetime y manejar errores
        datos['ds'] = pd.to_datetime(datos['ds'], errors='coerce')
        if datos['ds'].isnull().any():
            return jsonify({"error": "La columna 'ds' contiene fechas inválidas"}), 400

        # Verificar que no haya valores nulos en 'y'
        if datos['y'].isnull().any():
            return jsonify({"error": "La columna 'y' contiene valores nulos"}), 400

        # Obtener la cantidad de días a predecir (opcional)
        periodos = request.form.get("periods", default=7, type=int)
        if periodos <= 0 or periodos > 365:
            return jsonify({"error": "El número de días debe estar entre 1 y 365"}), 400

        # Crear y entrenar el modelo
        modelo = Prophet()
        modelo.fit(datos)

        # Crear fechas futuras y predecir
        futuro = modelo.make_future_dataframe(periods=periodos)
        prediccion = modelo.predict(futuro)

        # Devolver los últimos 'periodos' días predichos
        resultado = prediccion[['ds', 'yhat']].tail(periodos).to_dict(orient='records')
        return jsonify(resultado)

    except Exception as e:
        return jsonify({"error": f"Error en la predicción: {str(e)}"}), 500

# Ejecutar la app
if __name__ == "__main__":
    app.run(debug=True)


