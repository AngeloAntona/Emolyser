# Importazione librerie
import numpy as np
import glob
import os
from music21 import converter, instrument, note, chord, stream
from tqdm import tqdm
from collections import Counter
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.preprocessing import LabelEncoder

# ============================================
# ============ PARAMETRI GLOBALI =============
# ============================================

EMOTIONS = ['happiness', 'sadness', 'fear']
SEQUENCE_LENGTH = 50
MAX_PATTERNS = 50000
MODEL_PATH = 'saved_models/audio_generation.keras'
MODEL_DIR = 'saved_models'
RESULTS_DIR = 'generation_results'

# ============================================
# ========== CREAZIONE DELLE CARTELLE ========
# ============================================

# Crea le cartelle se non esistono
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

# ============================================
# ======== FUNZIONI PER LA PREPARAZIONE ======
# ============================================

# Funzione per estrarre le note da una singola emozione
def get_notes(emotion):
    notes = []
    files = glob.glob(f"data_midi/{emotion}/*.mid")
    for file in tqdm(files, desc=f"Elaborazione files {emotion}"):
        midi = converter.parse(file)

        notes_to_parse = None

        try:
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse()
        except Exception as e:
            notes_to_parse = midi.flat.notes

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                # Gestione degli accordi con una sola nota
                if len(element.pitches) == 1:
                    notes.append(str(element.pitches[0]))
                else:
                    notes.append('.'.join(str(n) for n in element.normalOrder))
    return notes

# Funzione per ottenere tutte le note e le emozioni
def get_all_notes_and_emotions():
    all_notes = []
    all_emotions = []
    for emotion in EMOTIONS:
        notes = get_notes(emotion)
        all_notes.extend(notes)
        all_emotions.extend([emotion]*len(notes))
    return all_notes, all_emotions

# ============================================
# ============ PREPARAZIONE DEI DATI =========
# ============================================

# Ottieni tutte le note e le emozioni
all_notes, all_emotions = get_all_notes_and_emotions()

# Tokenizzazione delle note
unique_notes = sorted(set(all_notes))
note_to_int = {note: number for number, note in enumerate(unique_notes)}
notes_as_int = [note_to_int[note] for note in all_notes]

# Codifica delle emozioni
label_encoder = LabelEncoder()
encoded_emotions = label_encoder.fit_transform(all_emotions)

# Creazione delle sequenze di input e output
network_input = []
network_output = []
network_emotion = []

for i in range(len(notes_as_int) - SEQUENCE_LENGTH):
    seq_in = notes_as_int[i:i + SEQUENCE_LENGTH]
    seq_out = notes_as_int[i + SEQUENCE_LENGTH]
    emotion = encoded_emotions[i + SEQUENCE_LENGTH]
    network_input.append(seq_in)
    network_output.append(seq_out)
    network_emotion.append(emotion)

# Limitare il numero di pattern
network_input = network_input[:MAX_PATTERNS]
network_output = network_output[:MAX_PATTERNS]
network_emotion = network_emotion[:MAX_PATTERNS]

# Converti l'input in array numpy
network_input = np.array(network_input)
network_output = np.array(network_output)
network_emotion = np.array(network_emotion)

# Normalizzazione delle note
n_vocab = len(unique_notes)
network_input = network_input / float(n_vocab)
network_input = network_input.reshape((network_input.shape[0], SEQUENCE_LENGTH, 1))

# Normalizzazione e ripetizione dell'emozione
emotion_normalized = network_emotion / float(max(network_emotion))
emotion_input = emotion_normalized.reshape(-1, 1, 1)
emotion_input = np.repeat(emotion_input, SEQUENCE_LENGTH, axis=1)

# Concatenazione delle note e delle emozioni come caratteristiche
network_input = np.concatenate((network_input, emotion_input), axis=2)

# Conversione dell'output in categoriale
network_output = to_categorical(network_output, num_classes=n_vocab)

# Verifica delle dimensioni degli array
print("Dimensioni input:", network_input.shape)  # Dovrebbe essere (numero_di_campioni, SEQUENCE_LENGTH, 2)
print("Dimensioni output:", network_output.shape)

# ============================================
# ============= DEFINIZIONE DEL MODELLO ======
# ============================================

def create_network(input_shape, n_vocab):
    inputs = Input(shape=(input_shape[1], input_shape[2]))
    x = GRU(256, return_sequences=True)(inputs)
    x = GRU(256)(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(n_vocab, activation='softmax')(x)

    model = Model(inputs, outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

# ============================================
# ========= CONTROLLO E CARICAMENTO MODELLO ==
# ============================================

if os.path.exists(MODEL_PATH):
    print("Caricamento del modello esistente...")
    model = load_model(MODEL_PATH)
else:
    print("Nessun modello esistente trovato. Addestramento del nuovo modello...")
    # Creazione del modello
    model = create_network(network_input.shape, n_vocab)
    model.summary()

    # Definizione dei callback
    checkpoint = ModelCheckpoint(
        MODEL_PATH,
        monitor='loss',
        verbose=1,
        save_best_only=True,
        mode='min'
    )
    early_stopping = EarlyStopping(
        monitor='loss',
        patience=5,
        verbose=1,
        mode='min'
    )
    callbacks_list = [checkpoint, early_stopping]

    # Aumentiamo il numero di epoche per migliorare l'apprendimento
    history = model.fit(network_input, network_output, epochs=50, batch_size=32, callbacks=callbacks_list)

# ============================================
# ============= GENERAZIONE MELODIA ==========
# ============================================

# Creazione del dizionario inverso
int_to_note = {number: note for note, number in note_to_int.items()}

# Funzione per il campionamento con temperatura
def sample_with_temperature(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    if temperature == 0:
        temperature = 1e-10
    preds = np.log(preds + 1e-10) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = preds
    return np.random.choice(len(probas), p=probas)

def generate_notes_by_emotion(model, network_input, int_to_note, n_vocab, desired_emotion, num_notes=100, temperature=1.0):
    # Codifica e normalizza l'emozione desiderata
    emotion_encoded = label_encoder.transform([desired_emotion])[0]
    emotion_normalized = emotion_encoded / float(max(label_encoder.transform(label_encoder.classes_)))

    # Scegli un punto di partenza casuale
    start = np.random.randint(0, len(network_input)-1)
    pattern = network_input[start]
    pattern = pattern.reshape(1, SEQUENCE_LENGTH, 2)

    # Imposta l'emozione desiderata nel pattern
    pattern[0, :, 1] = emotion_normalized

    prediction_output = []

    for note_index in range(num_notes):
        prediction = model.predict(pattern, verbose=0)
        # Utilizza il campionamento con temperatura
        index = sample_with_temperature(prediction[0], temperature)
        result = int_to_note[index]
        prediction_output.append(result)

        # Crea il nuovo input
        new_note = index / float(n_vocab)
        new_input = np.array([[new_note, emotion_normalized]])
        pattern = np.concatenate((pattern[:, 1:, :], new_input.reshape(1, 1, 2)), axis=1)
    return prediction_output

def create_midi(prediction_output, output_filename='output.mid'):
    offset = 0
    output_notes = []

    # Crea note e accordi dal risultato predetto
    for pattern in prediction_output:
        # Se il pattern è un accordo
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                try:
                    new_note = note.Note(int(current_note))
                except:
                    new_note = note.Note(current_note)
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        else:
            # Se il pattern è una nota singola
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)
        # Incrementa l'offset per evitare che le note si sovrappongano
        offset += 0.5

    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp=output_filename)
    print(f"Melodia generata e salvata in {output_filename}")

# ============================================
# ======= GENERAZIONE DELLA MELODIA ==========
# ============================================

# Seleziona l'emozione desiderata
desired_emotion = 'happiness'  # Può essere 'happiness', 'sadness' o 'fear'

# Genera la sequenza
prediction_output = generate_notes_by_emotion(
    model,
    network_input,
    int_to_note,
    n_vocab,
    desired_emotion,
    num_notes=200,  # Puoi modificare questo numero
    temperature=0.8  # Puoi sperimentare con diversi valori di temperatura
)

# Creazione del file MIDI
output_filename = os.path.join(RESULTS_DIR, f'melody_{desired_emotion}.mid')
create_midi(prediction_output, output_filename)