# ==========================================
#               CriptoAudio v3
#         By Alexandre Mazzei -2026
#        Com a ajuda da IA CLAUDE 4.6
#
#  AO RODAR A PRIMEIRA VEZ:
#  pip install numpy scipy pillow
# ==========================================

import numpy as np
import scipy.io.wavfile as wav
from PIL import Image, ImageDraw, ImageFont

# ==========================================
# CONFIGURATION
# ==========================================
SECRET_WORDS = "abandon ability able about above absent"

BACKGROUND_MUSIC_PATH = "input_music3.wav"
OUTPUT_AUDIO_PATH = "steganography_output.wav"

SAMPLE_RATE = 44100
DURATION    = 20          # 20s = mais espaço horizontal para as palavras
MIN_FREQ    = 2000
MAX_FREQ    = 10000

# Resolução da imagem calibrada para Window=1024 no Audacity
# img_w=600 → cada coluna = 33ms > janela FFT de 23ms → sem borrão temporal
# img_h=180 → passo de frequência ≈ 44Hz, que a FFT de 1024 resolve bem
img_w, img_h = 600, 180

MIX_VOLUME = 0.05

# ==========================================
# STEP 1: GERAR IMAGEM COM O TEXTO
# ==========================================
print("Gerando imagem de texto...")

img   = Image.new("L", (img_w, img_h), color=0)
draw  = ImageDraw.Draw(img)
words = SECRET_WORDS.split()

# Layout: 2 palavras por linha → 3 linhas (mais espaço horizontal)
lines = [" ".join(words[i:i+2]) for i in range(0, len(words), 2)]

# Fontes candidatas (Windows e Linux)
FONT_CANDIDATES = [
    "arialbd.ttf",
    "Arial Bold.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf",
]

def find_font(size):
    for path in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    print("[Aviso] Nenhuma fonte TrueType encontrada. O texto ficará ilegível.")
    return ImageDraw.Draw(Image.new("L", (1,1))).getfont()

# Auto-escala: encontra o maior tamanho que cabe na imagem
def best_font_size(lines, max_w, max_h, margin=0.92):
    for size in range(80, 8, -1):
        font = find_font(size)
        widths  = [draw.textbbox((0,0), l, font=font)[2] - draw.textbbox((0,0), l, font=font)[0] for l in lines]
        heights = [draw.textbbox((0,0), l, font=font)[3] - draw.textbbox((0,0), l, font=font)[1] for l in lines]
        if max(widths) <= max_w * margin and sum(heights) <= max_h * margin:
            return size, font
    return 10, find_font(10)

size, font = best_font_size(lines, img_w, img_h)
print(f"[Fonte] Tamanho escolhido: {size}px para {len(lines)} linhas")

# Distribui as linhas verticalmente de forma uniforme
n   = len(lines)
ys  = [(img_h * (i + 0.5) / n) for i in range(n)]  # centros de cada linha

for line, cy in zip(lines, ys):
    bbox = draw.textbbox((0,0), line, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = (img_w - w) / 2 - bbox[0]
    y = cy - h / 2 - bbox[1]
    draw.text((x, y), line, fill=255, font=font)

lit = np.count_nonzero(np.array(img) > 10)
print(f"[Checagem] Pixels acesos: {lit} ({100*lit/(img_w*img_h):.1f}%)")

# Flip vertical: espectrograma plota grave embaixo, agudo em cima
img = img.transpose(Image.FLIP_TOP_BOTTOM)
pixel_matrix = np.array(img) / 255.0

# ==========================================
# STEP 2: PIXELS → FREQUÊNCIAS DE ÁUDIO
# ==========================================
print("Sintetizando pixels em frequências...")
num_samples     = int(SAMPLE_RATE * DURATION)
time_axis       = np.linspace(0, DURATION, num_samples, endpoint=False)
secret_signal   = np.zeros(num_samples)
frequencies     = np.linspace(MIN_FREQ, MAX_FREQ, img_h)
samples_per_col = num_samples / img_w

for col in range(img_w):
    t0 = int(col * samples_per_col)
    t1 = int((col + 1) * samples_per_col)
    t_slice = time_axis[t0:t1]
    n = len(t_slice)
    if n == 0:
        continue

    active_rows = np.where(pixel_matrix[:, col] > 0.1)[0]
    if len(active_rows) == 0:
        continue

    col_sig = np.zeros(n)
    for row in active_rows:
        col_sig += pixel_matrix[row, col] * np.sin(2 * np.pi * frequencies[row] * t_slice)

    # Janela de Hann para eliminar cliques nas bordas de cada coluna
    col_sig *= np.hanning(n)
    secret_signal[t0:t1] = col_sig

if np.max(np.abs(secret_signal)) > 0:
    secret_signal /= np.max(np.abs(secret_signal))

# ==========================================
# STEP 3: MISTURAR COM A MÚSICA DE FUNDO
# ==========================================
print("Misturando com a música de fundo...")
try:
    bg_rate, bg_data = wav.read(BACKGROUND_MUSIC_PATH)
    bg_mono = bg_data[:, 0] if len(bg_data.shape) > 1 else bg_data
    if bg_mono.dtype == np.int16:
        bg_mono = bg_mono / 32768.0
    elif bg_mono.dtype == np.int32:
        bg_mono = bg_mono / 2147483648.0
    else:
        bg_mono = bg_mono.astype(np.float64)
except FileNotFoundError:
    print(f"⚠️ {BACKGROUND_MUSIC_PATH} não encontrado. Usando silêncio como base.")
    bg_rate = SAMPLE_RATE
    bg_mono = np.zeros(num_samples)

if len(bg_mono) < num_samples:
    bg_mono = np.concatenate((bg_mono, np.zeros(num_samples - len(bg_mono))))

mixed = bg_mono.copy()
mixed[:num_samples] += secret_signal * MIX_VOLUME
mixed = np.clip(mixed, -1.0, 1.0)

wav.write(OUTPUT_AUDIO_PATH, bg_rate, (mixed * 32767).astype(np.int16))

print(f"🎉 Salvo em: {OUTPUT_AUDIO_PATH}")
print()
print("═══════════════════════════════════════")
print(" CONFIGURAÇÕES DO AUDACITY (Spectrogram)")
print("  Escala(scale)   → Linear")
print("  Min frequency   → 5000 Hz")
print("  Max frequency   → 16000 Hz")
print("  Algoritmo       → Frequencies")
print("  Tamanho da Janela → 1024")
print("  Tipo de janela  → Hann")
print("  Fator zero-pad  → 2")
print("  Ganho(gain)     → 20 dB")
print("═══════════════════════════════════════")
print("Exiftoll")
