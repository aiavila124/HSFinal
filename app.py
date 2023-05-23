#!/usr/bin/env python
import os
from flask import flash
import pandas as pd
import json
from flask import Flask, redirect, send_file, request, render_template, send_from_directory, jsonify
import jinja2.exceptions
from clase.modelo import Modelo



app = Flask(__name__)
app.secret_key = 'clave_secreta'
modelo = Modelo(app)

@app.route('/')
def index():
    return render_template('forms.html')

@app.route('/<pagename>')
def admin(pagename):
        return render_template(f"{pagename}.html")    

@app.route('/ruta-de-resultados', methods=['POST'])
def procesar_archivo():
    archivo = request.files['mi_archivo'] 
    if archivo.filename == '' or not archivo.filename.endswith('.xlsx'):
        return render_template('forms.html')   
    resultados_json = modelo.codigo_modelo(archivo)
    os.remove(f"temp/{archivo.filename}")
    return render_template('forms.html', resultados_json=resultados_json)

@app.route('/descargar-archivo')
def descargar_archivo():
    ruta_archivo = os.path.join('temp', 'Lista Clientes.xlsx')
    return send_file(ruta_archivo, as_attachment=True)
   

@app.route('/<path:resource>')
def serveStaticResource(resource):
	return send_from_directory('static/', resource)

@app.route('/health')
def health_check():
    return jsonify(status='OK')

@app.errorhandler(jinja2.exceptions.TemplateNotFound)
def template_not_found(e):
    return not_found(e)

@app.errorhandler(404)
def not_found(e):
    return '<strong>Page Not Found!</strong>', 404

if __name__ == '__main__':
    app.run(debug = True, port= 5000)
