import os
import pickle
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (TimeDistributed, Conv1D, MaxPooling1D, Flatten, LSTM,
                                     Dense, Dropout)
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam

# ---------------------------
# Global Constants
# ---------------------------
MAX_PAD_LEN = 100         # Numero fisso di time frames
HOP_LENGTH = 512
N_FFT = 2048
TARGET_COUNT = 800        # Numero target di campioni per etichetta

# ---------------------------
# Utility: Z-score normalization
# ---------------------------
def normalize_feature(feature):
    mean = np.mean(feature, axis=1, keepdims=True)
    std = np.std(feature, axis=1, keepdims=True)
    return (feature - mean) / (std + 1e-6)

# ---------------------------
# STEP 1: Load and Prepare the Dataset
# ---------------------------
annotations = pd.read_csv('data/annotations/annotations averaged per song/song_level/static_annotations_averaged_songs_1_2000.csv')

def assign_emotion(row):
    valence = row[' valence_mean']
    arousal = row[' arousal_mean']
    # Quadrante 1: valence > 5, arousal > 5  --> "happiness"
    if valence > 5 and arousal > 5:
        return 'happiness'
    # Quadrante 2: valence < 5, arousal < 5  --> "sadness"
    elif valence < 5 and arousal < 5:
        return 'sadness'
    # Quadrante 3: valence < 5, arousal > 5  --> "anger"
    elif valence < 5 and arousal > 5:
        return 'anger'
    # Quadrante 4: valence > 5, arousal < 5  --> "calm"
    elif valence > 5 and arousal < 5:
        return 'calm'
    else:
        return 'neutral'

annotations['emotion'] = annotations.apply(assign_emotion, axis=1)
print("Emotion distribution (before filtering):")
print(annotations['emotion'].value_counts())

# Filtra per escludere "neutral" e song_id 3 e 4
dataset = annotations[(annotations['emotion'].notna()) & (annotations['emotion'] != 'neutral')]
dataset = dataset[~dataset['song_id'].isin([3, 4])]
print("Final emotion distribution in dataset:")
print(dataset['emotion'].value_counts())

# ---------------------------
# STEP 4: Build or Load the Model
# ---------------------------
if not os.path.exists('saved_models'):
    os.makedirs('saved_models')

MODEL_PATH = 'saved_models/audio_classification.keras'
ENCODER_PATH = 'saved_models/label_encoder.pkl'

if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH):
    print("Loading existing model...")
    model = load_model(MODEL_PATH)
    with open(ENCODER_PATH, 'rb') as f:
        le = pickle.load(f)
else:
    print("No existing model found. Training a new model...")

    # ---------------------------
    # STEP 2: Extract Audio Features (without augmentation)
    # ---------------------------
    def extract_features(file_name, max_pad_len=MAX_PAD_LEN, augment=False):
        try:
            # Usa un sample rate fisso per evitare problematiche di resampling
            audio_data, sample_rate = librosa.load(file_name, sr=22050)
            if augment:
                if np.random.rand() < 0.5:
                    audio_data = librosa.effects.pitch_shift(audio_data, sr=sample_rate,
                                                             n_steps=np.random.uniform(-2, 2))
                if np.random.rand() < 0.5:
                    audio_data = librosa.effects.time_stretch(audio_data, rate=np.random.uniform(0.8, 1.2))
            # Estrai MFCC (40)
            mfcc = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40,
                                        hop_length=HOP_LENGTH, n_fft=N_FFT)
            mfcc = normalize_feature(mfcc)
            # Estrai Mel-spectrogram (128)
            mel_spec = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate, n_mels=128,
                                                      hop_length=HOP_LENGTH, n_fft=N_FFT)
            mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
            mel_spec = normalize_feature(mel_spec)
            # Estrai Chroma (12)
            chroma = librosa.feature.chroma_stft(y=audio_data, sr=sample_rate,
                                                  hop_length=HOP_LENGTH, n_fft=N_FFT)
            chroma = normalize_feature(chroma)
            # Estrai Spectral Contrast (7)
            spec_contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sample_rate,
                                                              hop_length=HOP_LENGTH, n_fft=N_FFT)
            spec_contrast = normalize_feature(spec_contrast)
            # Estrai Tonnetz (6)
            tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(audio_data), sr=sample_rate)
            tonnetz = normalize_feature(tonnetz)
            # Estrai il tempo (BPM) e replicalo per ogni frame (numero di colonne dei MFCC)
            tempo, _ = librosa.beat.beat_track(y=audio_data, sr=sample_rate)
            T_frames = mfcc.shape[1]
            tempo_array = np.full((1, T_frames), tempo)
            # Aggiungi ulteriori feature:
            # Zero Crossing Rate (1)
            zcr = librosa.feature.zero_crossing_rate(y=audio_data, hop_length=HOP_LENGTH)
            zcr = normalize_feature(zcr)
            # Spectral Centroid (1)
            spec_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate,
                                                              hop_length=HOP_LENGTH, n_fft=N_FFT)
            spec_centroid = normalize_feature(spec_centroid)
            # Concatenazione verticale: dimensione totale = 40 + 128 + 12 + 7 + 6 + 1 + 1 + 1 = 196
            combined = np.concatenate([mfcc, mel_spec, chroma, spec_contrast, tonnetz, tempo_array, zcr, spec_centroid], axis=0)
            combined = combined.T  # forma: (T, 196)
            T_current = combined.shape[0]
            if T_current < max_pad_len:
                pad_width = max_pad_len - T_current
                combined = np.pad(combined, pad_width=((0, pad_width), (0, 0)), mode='constant')
            else:
                combined = combined[:max_pad_len, :]
            return combined
        except Exception as e:
            print(f"Error processing {file_name}: {e}")
            return None

    # Processa ogni file in modo sequenziale
    results = []
    for index, row in tqdm(dataset.iterrows(), total=dataset.shape[0]):
        file_name = os.path.join(os.getcwd(), 'data', 'MEMD_audio', f"{row['song_id']}.mp3")
        class_label = row['emotion']
        if not os.path.isfile(file_name):
            print(f"File not found: {file_name}")
        else:
            features_orig = extract_features(file_name, max_pad_len=MAX_PAD_LEN, augment=False)
            if features_orig is not None:
                results.append([features_orig, class_label, file_name])
    features = [res for res in results if res is not None]
    print(f"Number of samples with extracted features: {len(features)}")

    # ---------------------------
    # STEP 2.5: Balance the Dataset via Data Augmentation (sequenziale)
    # ---------------------------
    def generate_augmented_sample(label, file_names):
        chosen_file = np.random.choice(file_names)
        aug_features = extract_features(chosen_file, max_pad_len=MAX_PAD_LEN, augment=True)
        if aug_features is not None:
            return {'feature': aug_features, 'label': label, 'file_name': chosen_file}
        else:
            return None

    def balance_dataset(df, target_count):
        counts = df['label'].value_counts()
        new_rows = []
        for label, count in counts.items():
            if count < target_count:
                file_names = df[df['label'] == label]['file_name'].unique()
                num_to_generate = target_count - count
                print(f"Generating {num_to_generate} augmented samples for label '{label}'")
                for _ in range(num_to_generate):
                    sample = generate_augmented_sample(label, file_names)
                    if sample is not None:
                        new_rows.append(sample)
        if new_rows:
            df_new = pd.DataFrame(new_rows)
            df_balanced = pd.concat([df, df_new], ignore_index=True)
            return df_balanced
        else:
            return df

    features_df = pd.DataFrame(features, columns=['feature', 'label', 'file_name'])
    print(features_df.head())
    
    current_counts = features_df['label'].value_counts()
    print("Current label counts:")
    print(current_counts)
    print(f"Target count per class: {TARGET_COUNT}")
    
    features_df_balanced = balance_dataset(features_df, TARGET_COUNT)
    print("Label counts after balancing:")
    print(features_df_balanced['label'].value_counts())
    
    # ---------------------------
    # STEP 3: Prepare Data for the Neural Network
    # ---------------------------
    X = np.array(features_df_balanced['feature'].tolist())  # Forma: (n_samples, MAX_PAD_LEN, 196)
    X = np.expand_dims(X, axis=-1)  # Forma: (n_samples, MAX_PAD_LEN, 196, 1)
    
    y = np.array(features_df_balanced['label'].tolist())
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    yy = to_categorical(y_encoded)
    
    print(f"Shape of X: {X.shape}")
    print(f"Shape of yy: {yy.shape}")
    
    x_train, x_test, y_train, y_test = train_test_split(X, yy, test_size=0.2, random_state=42)
    print(f"Number of training samples: {x_train.shape[0]}")
    print(f"Number of testing samples: {x_test.shape[0]}")
    
    # ---------------------------
    # STEP 4: Build a Hybrid CNN-LSTM Model
    # ---------------------------
    num_labels = yy.shape[1]
    model = Sequential()
    model.add(TimeDistributed(Conv1D(filters=16, kernel_size=3, activation='relu'),
                              input_shape=(MAX_PAD_LEN, 196, 1)))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    model.add(TimeDistributed(Dropout(0.25)))
    
    model.add(TimeDistributed(Conv1D(filters=32, kernel_size=3, activation='relu')))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    model.add(TimeDistributed(Dropout(0.25)))
    
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(64, return_sequences=False, recurrent_dropout=0.2))
    model.add(Dropout(0.5))
    model.add(Dense(num_labels, activation='softmax'))
    
    model.summary()
    custom_lr = 0.0005
    optimizer = Adam(learning_rate=custom_lr)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    checkpointer = ModelCheckpoint(filepath=MODEL_PATH, verbose=1, save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True)
    
    history = model.fit(
        x_train, y_train, batch_size=32, epochs=50, validation_data=(x_test, y_test),
        callbacks=[checkpointer, early_stopping], verbose=1
    )
    
    with open(ENCODER_PATH, 'wb') as f:
        pickle.dump(le, f)
    
    score = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test set accuracy: {score[1]*100:.2f}%")
    
# ---------------------------
# STEP 7: Feature Extraction for Prediction
# ---------------------------
def extract_features_for_prediction(file_name, max_pad_len=MAX_PAD_LEN):
    try:
        audio_data, sample_rate = librosa.load(file_name, sr=22050)
        mfcc = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40,
                                    hop_length=HOP_LENGTH, n_fft=N_FFT)
        mfcc = normalize_feature(mfcc)
        mel_spec = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate, n_mels=128,
                                                  hop_length=HOP_LENGTH, n_fft=N_FFT)
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        mel_spec = normalize_feature(mel_spec)
        chroma = librosa.feature.chroma_stft(y=audio_data, sr=sample_rate,
                                              hop_length=HOP_LENGTH, n_fft=N_FFT)
        chroma = normalize_feature(chroma)
        spec_contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sample_rate,
                                                          hop_length=HOP_LENGTH, n_fft=N_FFT)
        spec_contrast = normalize_feature(spec_contrast)
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(audio_data), sr=sample_rate)
        tonnetz = normalize_feature(tonnetz)
        # Estrai il tempo (BPM) e replicalo per ogni frame (usando le colonne dei MFCC)
        tempo, _ = librosa.beat.beat_track(y=audio_data, sr=sample_rate)
        T_frames = mfcc.shape[1]
        tempo_array = np.full((1, T_frames), tempo)
        # Aggiungi anche le feature aggiuntive (ZCR e Spectral Centroid)
        zcr = librosa.feature.zero_crossing_rate(y=audio_data, hop_length=HOP_LENGTH)
        zcr = normalize_feature(zcr)
        spec_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate,
                                                          hop_length=HOP_LENGTH, n_fft=N_FFT)
        spec_centroid = normalize_feature(spec_centroid)
        # Concatenazione verticale: MFCC (40) + Mel (128) + Chroma (12) + SpecContrast (7) + Tonnetz (6) + Tempo (1) + ZCR (1) + SpecCentroid (1) = 196
        combined = np.concatenate([mfcc, mel_spec, chroma, spec_contrast, tonnetz, tempo_array, zcr, spec_centroid], axis=0)
        combined = combined.T  # forma: (T, 196)
        T_current = combined.shape[0]
        if T_current < max_pad_len:
            pad_width = max_pad_len - T_current
            combined = np.pad(combined, pad_width=((0, pad_width), (0, 0)), mode='constant')
        else:
            combined = combined[:max_pad_len, :]
        return combined
    except Exception as e:
        print(f"Error processing {file_name}: {e}")
        return None

def predict_emotion(file_path):
    features = extract_features_for_prediction(file_path, max_pad_len=MAX_PAD_LEN)
    if features is not None:
        new_features = np.expand_dims(features, axis=-1)
        new_features = np.expand_dims(new_features, axis=0)
        predicted_vector = model.predict(new_features)
        predicted_class = le.inverse_transform(np.argmax(predicted_vector, axis=1))
        print(f"Predicted emotion for '{file_path}': {predicted_class[0]}")
        print(f"Shape of new_features: {new_features.shape}")
    else:
        print(f"Error processing {file_path}")

# Test the model on new files
new_file1 = 'test/4.mp3'
predict_emotion(new_file1)

new_file2 = 'test/3.mp3'
predict_emotion(new_file2)


# OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES python3 code1.py