import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans

def analisis(data):
    # Membuat dataframe dari data yang dikirimkan
    df_input = pd.DataFrame([data])

    # Membaca data dari dataset
    df = pd.read_csv('./dataset-umkm.csv', delimiter=';') # dari database umkm

    # Menambahkan data baru ke dataset
    df = pd.concat([df, df_input], ignore_index=True)

    # Inisialisasi LabelEncoder untuk setiap kolom
    encoders = {}
    for column in ['nama_pemilik', 'nama_usaha', 'jenis_usaha', 'kategori_usaha', 'lama_usaha', 'jumlah_karyawan', 'omset_tahunan', 'jenis_investasi', 'modal_diinginkan', 'lokasi']:
        encoders[column] = LabelEncoder()
        df[column] = encoders[column].fit_transform(df[column])

    # Feature selection
    features = ['lama_usaha','omset_tahunan','modal_diinginkan']

    # Normalisasi
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[features])

    # K-Means Clustering
    kmeans = KMeans(n_clusters=2, n_init=10, max_iter=300, random_state=0)

    df['status'] = kmeans.fit_predict(scaled_features)
    # Mengubah label cluster menjadi "Ya" dan "Tidak"
    df['status'] = df['status'].map({0: "ditolak_sistem", 1: "diterima_sistem"})

    # Mengonversi kembali data menjadi bentuk asli semula
    for column in encoders:
        df[column] = encoders[column].inverse_transform(df[column])

    last_row = df.tail(1)

    # merubah menjadi array objek
    # Menampilkan lima baris pertama dari dataframe hasil akhir
    last_row_dict = last_row.to_dict(orient='records')
    
    return last_row_dict
