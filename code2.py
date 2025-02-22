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
from tensorflow.keras.layers import Input, LSTM, GRU, Dense, Dropout, Embedding, Concatenate
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
network_input_notes = []
network_input_emotions = []
network_output = []

for i in range(len(notes_as_int) - SEQUENCE_LENGTH):
    seq_in_notes = notes_as_int[i:i + SEQUENCE_LENGTH]
    seq_in_emotions = encoded_emotions[i:i + SEQUENCE_LENGTH]
    seq_out = notes_as_int[i + SEQUENCE_LENGTH]
    network_input_notes.append(seq_in_notes)
    network_input_emotions.append(seq_in_emotions)
    network_output.append(seq_out)

# Limitare il numero di pattern
network_input_notes = network_input_notes[:MAX_PATTERNS]
network_input_emotions = network_input_emotions[:MAX_PATTERNS]
network_output = network_output[:MAX_PATTERNS]

# Converti gli input in array numpy
network_input_notes = np.array(network_input_notes)
network_input_emotions = np.array(network_input_emotions)
network_output = np.array(network_output)

# Numero di note e di emozioni
n_vocab_notes = len(unique_notes)
n_vocab_emotions = len(EMOTIONS)

# Conversione dell'output in categoriale
network_output = to_categorical(network_output, num_classes=n_vocab_notes)

# Verifica delle dimensioni degli array
print("Dimensioni input note:", network_input_notes.shape)  # (numero_di_campioni, SEQUENCE_LENGTH)
print("Dimensioni input emozioni:", network_input_emotions.shape)  # (numero_di_campioni, SEQUENCE_LENGTH)
print("Dimensioni output:", network_output.shape)

# ============================================
# ============= DEFINIZIONE DEL MODELLO ======
# ============================================

def create_network(input_shape_notes, input_shape_emotions, n_vocab_notes, n_vocab_emotions):
    # Input per le note
    notes_input = Input(shape=(input_shape_notes[1],), name='notes_input')
    # Embedding per le note
    notes_embedding = Embedding(input_dim=n_vocab_notes, output_dim=128, input_length=SEQUENCE_LENGTH)(notes_input)

    # Input per le emozioni
    emotions_input = Input(shape=(input_shape_emotions[1],), name='emotions_input')
    # Embedding per le emozioni
    emotions_embedding = Embedding(input_dim=n_vocab_emotions, output_dim=8, input_length=SEQUENCE_LENGTH)(emotions_input)

    # Concatenazione delle embeddings
    x = Concatenate()([notes_embedding, emotions_embedding])

    # Layer ricorrenti
    x = GRU(512, return_sequences=True)(x)
    x = GRU(512)(x)
    x = Dropout(0.4)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(n_vocab_notes, activation='softmax')(x)

    model = Model([notes_input, emotions_input], outputs)
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
    model = create_network(network_input_notes.shape, network_input_emotions.shape, n_vocab_notes, n_vocab_emotions)
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
    history = model.fit(
        [network_input_notes, network_input_emotions],
        network_output,
        epochs=100,
        batch_size=64,  # Puoi aumentare il batch size
        callbacks=callbacks_list
    )

# ============================================
# ============= GENERAZIONE MELODIA ==========
# ============================================

# Creazione del dizionario inverso
int_to_note = {number: note for note, number in note_to_int.items()}

# Funzioni di campionamento avanzate
def sample_with_temperature(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    if temperature == 0:
        temperature = 1e-10
    preds = np.log(preds + 1e-10) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    return np.random.choice(len(preds), p=preds)

def top_k_sampling(preds, k=10):
    preds = np.asarray(preds).astype('float64')
    indices = np.argpartition(preds, -k)[-k:]
    top_preds = preds[indices]
    top_preds = top_preds / np.sum(top_preds)
    return np.random.choice(indices, p=top_preds)

def nucleus_sampling(preds, p=0.9):
    preds = np.asarray(preds).astype('float64')
    sorted_indices = np.argsort(preds)[::-1]
    cumulative_probs = np.cumsum(preds[sorted_indices])
    cutoff = np.where(cumulative_probs >= p)[0][0] + 1
    top_indices = sorted_indices[:cutoff]
    top_preds = preds[top_indices]
    top_preds = top_preds / np.sum(top_preds)
    return np.random.choice(top_indices, p=top_preds)

def generate_notes_by_emotion(model, network_input_notes, network_input_emotions, int_to_note, n_vocab_notes, desired_emotion, num_notes=100, temperature=1.0):
    # Codifica l'emozione desiderata
    emotion_encoded = label_encoder.transform([desired_emotion])[0]

    # Scegli un punto di partenza casuale
    start = np.random.randint(0, len(network_input_notes)-1)
    pattern_notes = network_input_notes[start]
    pattern_emotions = network_input_emotions[start]

    # Imposta l'emozione desiderata nel pattern
    pattern_emotions[:] = emotion_encoded

    prediction_output = []

    for note_index in range(num_notes):
        prediction_input_notes = np.reshape(pattern_notes, (1, len(pattern_notes)))
        prediction_input_emotions = np.reshape(pattern_emotions, (1, len(pattern_emotions)))

        prediction = model.predict([prediction_input_notes, prediction_input_emotions], verbose=0)[0]

        # Utilizza il campionamento con temperatura o altre tecniche
        # index = sample_with_temperature(prediction, temperature)
        # index = top_k_sampling(prediction, k=10)
        index = nucleus_sampling(prediction, p=0.9)

        result = int_to_note[index]
        prediction_output.append(result)

        # Aggiorna il pattern per il prossimo input
        pattern_notes = np.append(pattern_notes[1:], index)
        pattern_emotions = np.append(pattern_emotions[1:], emotion_encoded)
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
    network_input_notes,
    network_input_emotions,
    int_to_note,
    n_vocab_notes,
    desired_emotion,
    num_notes=200,  # Puoi modificare questo numero
    temperature=0.7  # Puoi sperimentare con diversi valori di temperatura
)

# Creazione del file MIDI
output_filename = os.path.join(RESULTS_DIR, f'melody_{desired_emotion}.mid')
create_midi(prediction_output, output_filename)