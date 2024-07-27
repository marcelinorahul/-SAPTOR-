import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
import networkx as nx
import gym
from gym import spaces
from transformers import pipeline
import matplotlib.pyplot as plt

class LingkunganInventori(gym.Env):
    def __init__(self):
        super(LingkunganInventori, self).__init__()
        self.action_space = spaces.Discrete(10)
        self.observation_space = spaces.Box(low=0, high=100, shape=(2,), dtype=np.float32)
        self.state = np.array([50, 10], dtype=np.float32)
        self.inventori_maksimal = 100
        
    def step(self, action):
        inventori, permintaan = self.state
        inventori = min(inventori - permintaan + action, self.inventori_maksimal)
        permintaan = np.random.randint(0, 20)
        reward = -abs(inventori - 50)
        self.state = np.array([inventori, permintaan], dtype=np.float32)
        done = False
        return self.state, reward, done, {}
    
    def reset(self):
        self.state = np.array([50, 10], dtype=np.float32)
        return self.state

class LingkunganHarga(gym.Env):
    def __init__(self):
        super(LingkunganHarga, self).__init__()
        self.action_space = spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=100, shape=(2,), dtype=np.float32)
        self.state = np.array([50, 10], dtype=np.float32)
        
    def step(self, action):
        harga, permintaan = self.state
        harga = action[0]
        permintaan = max(0, 100 - harga + np.random.normal(0, 10))
        reward = harga * permintaan
        self.state = np.array([harga, permintaan], dtype=np.float32)
        done = False
        return self.state, reward, done, {}
    
    def reset(self):
        self.state = np.array([50, 10], dtype=np.float32)
        return self.state

class AgenSederhana:
    def __init__(self, action_space):
        self.action_space = action_space

    def prediksi(self, state):
        if isinstance(self.action_space, spaces.Discrete):
            return np.random.randint(0, self.action_space.n)
        else:
            return self.action_space.sample()

class SAPTOR:
    def __init__(self):
        self.model_permintaan = self._bangun_model_permintaan()
        self.lingkungan_inventori = LingkunganInventori()
        self.agen_inventori = AgenSederhana(self.lingkungan_inventori.action_space)
        self.lingkungan_harga = LingkunganHarga()
        self.agen_harga = AgenSederhana(self.lingkungan_harga.action_space)
        self.penganalisis_sentimen = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        self.perekomendasi_pemasok = self._bangun_perekomendasi_pemasok()
        self.graf_rantai_pasok = nx.Graph()
        self.scaler = StandardScaler()

    def _bangun_model_permintaan(self):
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
        nn = Sequential([
            LSTM(50, activation='relu', input_shape=(1, 8)),
            Dense(1)
        ])
        nn.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return [rf, gb, nn]

    def _bangun_perekomendasi_pemasok(self):
        return RandomForestRegressor(n_estimators=100, random_state=42)

    def latih_model(self, data_permintaan, data_pemasok):
        self.data_permintaan = data_permintaan
        X = data_permintaan.drop('permintaan', axis=1)
        y = data_permintaan['permintaan']

        X_encoded = pd.get_dummies(X, columns=['musim'])
        self.X_encoded = X_encoded

        X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.model_permintaan[0].fit(X_train_scaled, y_train)
        self.model_permintaan[1].fit(X_train_scaled, y_train)
        
        X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
        self.model_permintaan[2].fit(X_train_lstm, y_train, epochs=50, batch_size=32, verbose=0)
        
        X_pemasok = data_pemasok.drop('peringkat', axis=1)
        y_pemasok = data_pemasok['peringkat']
        self.perekomendasi_pemasok.fit(X_pemasok, y_pemasok)

    def prediksi_permintaan(self, fitur):
        fitur_df = pd.DataFrame([fitur])
        fitur_encoded = pd.get_dummies(fitur_df, columns=['musim'])
        fitur_encoded = fitur_encoded.reindex(columns=self.X_encoded.columns, fill_value=0)
        fitur_scaled = self.scaler.transform(fitur_encoded)
        
        rf_pred = self.model_permintaan[0].predict(fitur_scaled)
        gb_pred = self.model_permintaan[1].predict(fitur_scaled)
        nn_pred = self.model_permintaan[2].predict(fitur_scaled.reshape((1, 1, fitur_scaled.shape[1])))
        return np.mean([rf_pred, gb_pred, nn_pred.flatten()])

    def optimasi_inventori(self, state):
        return self.agen_inventori.prediksi(state)

    def tetapkan_harga_dinamis(self, state):
        return self.agen_harga.prediksi(state)

    def analisis_sentimen(self, teks):
        return self.penganalisis_sentimen(teks)[0]

    def rekomendasi_pemasok(self, kriteria):
        return self.perekomendasi_pemasok.predict(kriteria)

    def visualisasi_rantai_pasok(self):
        pos = nx.spring_layout(self.graf_rantai_pasok)
        nx.draw(self.graf_rantai_pasok, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=10, font_weight='bold')
        plt.title("Jaringan Rantai Pasok")
        plt.show()

    def bangun_graf_rantai_pasok(self, nodes, edges):
        self.graf_rantai_pasok.add_nodes_from(nodes)
        self.graf_rantai_pasok.add_edges_from(edges)

if __name__ == "__main__":
    saptor = SAPTOR()

    np.random.seed(42)
    n_samples = 100
    data_permintaan = pd.DataFrame({
        'musim': np.random.choice(['semi', 'panas', 'gugur', 'dingin'], n_samples),
        'harga': np.random.uniform(10, 100, n_samples),
        'ekonomi': np.random.uniform(0, 1, n_samples),
        'lalu_lintas_online': np.random.uniform(1000, 10000, n_samples),
        'peringkat_produk': np.random.uniform(1, 5, n_samples),
        'permintaan': np.random.uniform(50, 500, n_samples)
    })

    data_pemasok = pd.DataFrame({
        'harga': np.random.uniform(10, 100, n_samples),
        'kualitas': np.random.uniform(1, 5, n_samples),
        'waktu_pengiriman': np.random.uniform(1, 30, n_samples),
        'keandalan': np.random.uniform(0, 1, n_samples),
        'kapasitas': np.random.uniform(1000, 10000, n_samples),
        'peringkat': np.random.uniform(1, 5, n_samples)
    })

    saptor.latih_model(data_permintaan, data_pemasok)

    nodes = ['Pemasok A', 'Pemasok B', 'Gudang 1', 'Gudang 2', 'Toko 1', 'Toko 2', 'Toko 3']
    edges = [('Pemasok A', 'Gudang 1'), ('Pemasok B', 'Gudang 1'), 
             ('Pemasok B', 'Gudang 2'), ('Gudang 1', 'Toko 1'), 
             ('Gudang 1', 'Toko 2'), ('Gudang 2', 'Toko 2'), 
             ('Gudang 2', 'Toko 3')]
    saptor.bangun_graf_rantai_pasok(nodes, edges)

    print("Prediksi Permintaan:")
    sampel_fitur = data_permintaan.drop('permintaan', axis=1).iloc[0].to_dict()
    permintaan_diprediksi = saptor.prediksi_permintaan(sampel_fitur)
    print(f"Prediksi permintaan: {permintaan_diprediksi:.2f}")

    print("\nOptimisasi Inventori:")
    state_inventori = np.array([50, 10], dtype=np.float32)
    pesanan_optimal = saptor.optimasi_inventori(state_inventori)
    print(f"Jumlah optimal untuk dipesan: {pesanan_optimal}")

    print("\nPenetapan Harga Dinamis:")
    state_harga = np.array([50, 10], dtype=np.float32)
    harga_optimal = saptor.tetapkan_harga_dinamis(state_harga)
    print(f"Harga optimal: Rp{harga_optimal[0]:.2f}")

    print("\nAnalisis Sentimen:")
    sentimen = saptor.analisis_sentimen("Pemasok mengirimkan produk berkualitas tinggi tepat waktu.")
    print(f"Sentimen: {sentimen['label']} (Skor: {sentimen['score']:.2f})")

    print("\nRekomendasi Pemasok:")
    kriteria_pemasok = data_pemasok.drop('peringkat', axis=1).iloc[0].values.reshape(1, -1)
    peringkat_pemasok = saptor.rekomendasi_pemasok(kriteria_pemasok)
    print(f"Peringkat pemasok yang direkomendasikan: {peringkat_pemasok[0]:.2f}")

    print("\nVisualisasi Rantai Pasokan:")
    saptor.visualisasi_rantai_pasok()
