import pandas as pd
import librosa
import numpy as np
import os
from joblib import Parallel, delayed
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.callbacks import ModelCheckpoint

# ----------------------------------------------------------------------
# STEP 1: Carica e prepara il dataset
# ----------------------------------------------------------------------

# Carica i dati
annotations = pd.read_csv('data/annotations/annotations averaged per song/song_level/static_annotations_averaged_songs_1_2000.csv')

# Crea una funzione per assegnare le emozioni in base a valence e arousal
def assign_emotion(row):
    valence = row[' valence_mean']
    arousal = row[' arousal_mean']
    if valence > 5 and arousal > 5:
        return 'happiness'
    elif valence <= 5 and arousal <= 5:
        return 'sadness'
    elif valence <= 5 and arousal > 5 and arousal < 7:
        return 'fear'
    elif valence <= 5 and arousal >= 7:
        return 'anger'
    else:
        return None  # Esclude altri stati emotivi

# Applica la funzione per assegnare le emozioni
annotations['emotion'] = annotations.apply(assign_emotion, axis=1)

# Accorpa 'anger' in 'sadness'
annotations['emotion'] = annotations['emotion'].replace({'anger': 'sadness'})

# Esclude le canzoni con song_id 3 e 4
annotations = annotations[~annotations['song_id'].isin([3, 4])]

# Filtra le righe con emozioni assegnate
dataset = annotations[annotations['emotion'].notna()]

# Visualizza le prime righe del dataset
print(dataset.head())
print(dataset['emotion'].value_counts())

# ----------------------------------------------------------------------
# STEP 2: Estrazione delle caratteristiche audio
# ----------------------------------------------------------------------

def extract_features(file_name):
    try:
        audio_data, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        # Estrazione delle MFCC
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40)
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        return mfccs_scaled
    except Exception as e:
        print(f"Errore nell'elaborazione del file {file_name}: {e}")
        return None

def process_file(row):
    file_name = os.path.join(os.getcwd(), 'data', 'MEMD_audio', f"{row['song_id']}.mp3")
    class_label = row['emotion']
    if not os.path.isfile(file_name):
        print(f"File non trovato: {file_name}")
        return None
    data = extract_features(file_name)
    if data is not None:
        return [data, class_label]
    else:
        return None

results = Parallel(n_jobs=-1)(delayed(process_file)(row) for index, row in tqdm(dataset.iterrows(), total=dataset.shape[0]))
features = [result for result in results if result is not None]

print(f'Numero di campioni con caratteristiche estratte: {len(features)}')

# ----------------------------------------------------------------------
# STEP 3: Prepara i dati per la rete neurale
# ----------------------------------------------------------------------

# Crea un DataFrame con le caratteristiche e le etichette
features_df = pd.DataFrame(features, columns=['feature', 'label'])
print(features_df.head())

# Converte le caratteristiche in array numpy
X = np.array(features_df['feature'].tolist())

# Codifica le etichette
y = np.array(features_df['label'].tolist())
le = LabelEncoder()
yy = to_categorical(le.fit_transform(y))

print(f'Forma di X: {X.shape}')
print(f'Forma di yy: {yy.shape}')

x_train, x_test, y_train, y_test = train_test_split(X, yy, test_size=0.2, random_state=42)

print(f'Numero di campioni nel training set: {x_train.shape[0]}')
print(f'Numero di campioni nel testing set: {x_test.shape[0]}')

# ----------------------------------------------------------------------
# STEP 4: Costruisci la rete neurale
# ----------------------------------------------------------------------

num_labels = yy.shape[1]
model = Sequential()
model.add(Dense(256, input_shape=(40,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_labels))
model.add(Activation('softmax'))

model.summary()
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

# ----------------------------------------------------------------------
# STEP 5: Addestra il modello
# ----------------------------------------------------------------------

if not os.path.exists('saved_models'):
    os.makedirs('saved_models')

checkpointer = ModelCheckpoint(filepath='saved_models/audio_classification.keras', verbose=1, save_best_only=True)

history = model.fit(x_train, y_train, batch_size=32, epochs=20, validation_data=(x_test, y_test), callbacks=[checkpointer], verbose=1)

# ----------------------------------------------------------------------
# STEP 6: Valuta il modello
# ----------------------------------------------------------------------

model.load_weights('saved_models/audio_classification.keras')
score = model.evaluate(x_test, y_test, verbose=0)
print(f'Accuracy sul test set: {score[1]*100:.2f}%')

# ----------------------------------------------------------------------
# STEP 7: Testa il modello
# ----------------------------------------------------------------------

# Testiamo canzone felice
new_file1 = 'test/4.mp3'
new_features1 = extract_features(new_file1)
if new_features1 is not None:
    new_features1 = np.expand_dims(new_features1, axis=0)
    predicted_vector1 = model.predict(new_features1)
    predicted_class1 = le.inverse_transform(np.argmax(predicted_vector1, axis=1))
    print(f"L'emozione predetta per '{new_file1}' è: {predicted_class1[0]}")
    print(f'Forma di new_features: {new_features1.shape}')
else:
    print(f"Errore nell'elaborazione della canzone {new_file1}")

# Testiamo canzone triste
new_file2 = 'test/3.mp3'
new_features2 = extract_features(new_file2)
if new_features2 is not None:
    new_features2 = np.expand_dims(new_features2, axis=0)
    predicted_vector2 = model.predict(new_features2)
    predicted_class2 = le.inverse_transform(np.argmax(predicted_vector2, axis=1))
    print(f"L'emozione predetta per '{new_file2}' è: {predicted_class2[0]}")
    print(f'Forma di new_features: {new_features2.shape}')
else:
    print(f"Errore nell'elaborazione della canzone {new_file2}")