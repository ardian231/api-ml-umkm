from flask import Flask, jsonify, request
from analisis_umkm import analisis
from rekomendasi_umkm import rekomendasi

app = Flask(__name__)

@app.route('/api/analisis_umkm', methods=['POST'])
def analisis_umkm():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No input data provided"}), 400
    
    # Proses data menggunakan fungsi analisis dari modul analisis_umkm
    hasil_analisis = analisis(data)
    
    return jsonify(hasil_analisis), 200

@app.route('/api/rekomendasi_umkm', methods=['POST'])
def rekomendasi_umkm():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No input data provided"}), 400
    
    # Proses rekomendasi menggunakan fungsi rekomendasi dari modul rekomendasi_umkm
    hasil_rekomendasi = rekomendasi(data)
    
    return jsonify(hasil_rekomendasi), 200

if __name__ == '__main__':
    app.run(debug=True)
