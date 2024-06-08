import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import joblib

# Load datasets
df_umkm = pd.read_csv('dataset-umkm.csv', delimiter=';')
df_investor = pd.read_csv('dataset-investor.csv', delimiter=';')

# Drop 'status' column from df_umkm and check for missing values
df_umkm = df_umkm.drop(columns=['status'])

# Transform data using LabelEncoder
investor_encoders = {}
for column in ['jenis_investasi']:
    investor_encoders[column] = LabelEncoder()
    df_investor[column] = investor_encoders[column].fit_transform(df_investor[column])

umkm_encoders = {}
for column in ['jenis_investasi']:
    umkm_encoders[column] = LabelEncoder()
    df_umkm[column] = umkm_encoders[column].fit_transform(df_umkm[column])

# Extract features
features_investor = df_investor[['jenis_investasi']]
features_umkm = df_umkm[['jenis_investasi']]

# Normalize data using MinMaxScaler
scaler = MinMaxScaler()

features_umkm_scaled = scaler.fit_transform(features_umkm)
features_investor_scaled = scaler.transform(features_investor)

# Save models using joblib
joblib.dump(umkm_encoders, 'model-umkm-encoders.joblib')
joblib.dump(investor_encoders, 'model-investor_encoders.joblib')
joblib.dump(scaler, 'model-scaler.joblib')

# Compute cosine similarity matrix
similarity_matrix = cosine_similarity(features_umkm_scaled, features_investor_scaled)

# Function to recommend investors based on UMKM index
def recommend_investors(umkm_index):
    umkm_investasi = df_umkm.iloc[umkm_index]['jenis_investasi']
    matching_investors = df_investor[df_investor['jenis_investasi'] == umkm_investasi]
    
    if matching_investors.empty:
        return "Tidak ada investor yang cocok dengan jenis investasi UMKM ini."
    
    similarity_scores = similarity_matrix[umkm_index][matching_investors.index]
    top_indices = similarity_scores.argsort()[::-1]
    recommended_investors = matching_investors.iloc[top_indices][['nama_investor', 'email', 'no_telepon']]
    return recommended_investors

# Function to get actual recommendations based on UMKM index
def get_actual_recommendations(umkm_index):
    umkm_investasi = df_umkm.iloc[umkm_index]['jenis_investasi']
    matching_investors = df_investor[df_investor['jenis_investasi'] == umkm_investasi]
    actual_recommendations = matching_investors['nama_investor'].tolist()
    return actual_recommendations

# Function to get predicted recommendations for all UMKM
def get_predicted_recommendations():
    predicted_recommendations_list = []
    for umkm_index in range(len(df_umkm)):
        recommendations = recommend_investors(umkm_index)
        recommendations_list = recommendations['nama_investor'].tolist()
        predicted_recommendations_list.append(recommendations_list)
    return predicted_recommendations_list

# Get predicted recommendations
predicted_recommendations = get_predicted_recommendations()

# Create DataFrames for recommendations
dfs_recommendations = []
for umkm_index, recommendations in enumerate(predicted_recommendations):
    df_umkm_recommendations = pd.DataFrame(columns=["Nama Investor", "Email", "No Telepon"])
    for investor_index, investor in enumerate(recommendations):
        investor_info = df_investor[df_investor['nama_investor'] == investor].iloc[0]
        df_umkm_recommendations = pd.concat([df_umkm_recommendations, pd.DataFrame({
            "Nama Investor": [investor],
            "Email": [investor_info['email']],
            "No Telepon": [investor_info['no_telepon']]
        })], ignore_index=True)
    dfs_recommendations.append(df_umkm_recommendations)

# Display recommendations for the last UMKM
last_umkm_index = len(dfs_recommendations) - 1
last_umkm_recommendations = dfs_recommendations[last_umkm_index]

# Save recommendation model using joblib
joblib.dump(recommendations, 'model-sistem_rekomendasi.joblib')

# Load models using joblib
umkm_encoders = joblib.load('model-umkm-encoders.joblib')
investor_encoders = joblib.load('model-investor_encoders.joblib')
scaler = joblib.load('model-scaler.joblib')

import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

def rekomendasi(data):
    # Load datasets
    df_umkm = pd.read_csv('dataset-umkm.csv', delimiter=';')
    df_investor = pd.read_csv('dataset-investor.csv', delimiter=';')

    # Drop status column
    df_umkm = df_umkm.drop(columns=['status'])

    # Append new UMKM data
    df_new_umkm = pd.DataFrame([data])
    df_umkm = pd.concat([df_umkm, df_new_umkm], ignore_index=True)

    # Encode categorical data for investors
    investor_encoders = {}
    for column in ['jenis_investasi']:
        investor_encoders[column] = LabelEncoder()
        df_investor[column] = investor_encoders[column].fit_transform(df_investor[column])

    # Encode categorical data for UMKM
    umkm_encoders = {}
    for column in ['jenis_investasi']:
        umkm_encoders[column] = LabelEncoder()
        df_umkm[column] = umkm_encoders[column].fit_transform(df_umkm[column])

    # Select features for similarity computation
    features_investor = df_investor[['jenis_investasi']]
    features_umkm = df_umkm[['jenis_investasi']]

    # Normalize the data (optional)
    scaler = MinMaxScaler()
    features_umkm_scaled = scaler.fit_transform(features_umkm)
    features_investor_scaled = scaler.transform(features_investor)

    # Compute cosine similarity
    similarity_matrix = cosine_similarity(features_umkm_scaled, features_investor_scaled)

    # Generate recommendations for the new UMKM entry (last row in df_umkm)
    umkm_index = len(df_umkm) - 1

    def recommend_investors(umkm_index):
        umkm_investasi = df_umkm.iloc[umkm_index]['jenis_investasi']
        # Filter investors based on UMKM's investment type
        matching_investors = df_investor[df_investor['jenis_investasi'] == umkm_investasi]
        if matching_investors.empty:
            return "Tidak ada investor yang cocok dengan jenis investasi UMKM ini."
        similarity_scores = similarity_matrix[umkm_index][matching_investors.index]
        top_indices = similarity_scores.argsort()[::-1]
        recommended_investors = matching_investors.iloc[top_indices][['nama_investor', 'email', 'no_telepon']]
        return recommended_investors

    recommendations = recommend_investors(umkm_index)
    df_recommendations = pd.DataFrame(recommendations)
    recommendations = df_recommendations.to_dict(orient='records')

    return recommendations
