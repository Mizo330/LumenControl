import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

def find_diagonal_patterns(ssm, threshold=0.8, min_length=8):
    """Keres párhuzamos átlókat az SSM-ben."""
    diagonals = []
    n = ssm.shape[0]
    for i in range(n):
        for j in range(i+1, n):  # Csak a főátló felett
            if ssm[i, j] > threshold:
                length = 0
                while i+length < n and j+length < n and ssm[i+length, j+length] > threshold:
                    length += 1
                if length >= min_length:
                    diagonals.append((i, j, length))
    return diagonals


def format_time(seconds):
    """Másodpercek perc:másodperc formátumba alakítása."""
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes}:{seconds:02}"

# Audio betöltése
y, sr = librosa.load("/home/appuser/lumenai/segment_t.wav", sr=None)
n = 1
# Kromagram kiszámítása
chroma = librosa.feature.chroma_stft(y=y, sr=sr,n_fft=4096*n, hop_length=2048*n, win_length=4096*n)
# Meghatározzuk az SSM maximális értékét
# SSM számítása
ssm = librosa.segment.recurrence_matrix(chroma, metric='cosine', sym=True, mode='affinity')
ssm_max = np.max(ssm)
print(ssm.dtype)  # Ellenőrizd, hogy a mátrix elemei float típusúak-e
print(np.max(ssm))  # Nézd meg a maximális értéket

# Megjelenítés
plt.figure(figsize=(10, 10))
plt.imshow(ssm, origin='lower', cmap='coolwarm', aspect='auto')
plt.colorbar(label='Hasonlóság')
plt.title("Self-Similarity Matrix")
# Tengelycímkék másodpercekkel
n_frames = ssm.shape[0]
times = librosa.frames_to_time(range(n_frames), sr=sr, hop_length=2028*n)
plt.xticks(range(0, n_frames, n_frames // 10), [format_time(t) for t in times[::n_frames // 10]])
plt.yticks(range(0, n_frames, n_frames // 10), [format_time(t) for t in times[::n_frames // 10]])
plt.xlabel("Idő (perc:másodperc)")
plt.ylabel("Idő (perc:másodperc)")
plt.show()