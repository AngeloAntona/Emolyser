import os
import glob
import numpy as np
from music21 import converter, instrument, note, chord, stream
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, GRU, Dense, Dropout, Embedding, Concatenate, Layer, RepeatVector, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import LabelEncoder

# ============================================
# ============ PARAMETRI GLOBALI =============
# ============================================
# Cartelle originali con le 4 etichette
EMOTIONS = ['happiness', 'sadness', 'calm', 'anger']
# Mappatura in macro etichette:
#   happiness + calm => positive_emotion
#   anger + sadness  => negative_emotion
MACRO_MAPPING = {
    "happiness": "positive_emotion",
    "calm": "positive_emotion",
    "anger": "negative_emotion",
    "sadness": "negative_emotion"
}

SEQUENCE_LENGTH = 50
MAX_PATTERNS = 50000
MODEL_PATH = 'saved_models/audio_generation.keras'
MODEL_DIR = 'saved_models'
RESULTS_DIR = 'generation_results'

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Parametri per tokenizzazione delle note:
EOS_TOKEN = 129
START_TOKEN = 130
PAD_TOKEN = 0
VOCAB_SIZE = 131  # I pitch sono da 0 a 127; +1 per spostare; 0 riservato al PAD

# Parametri per il modello
NOTE_EMB_DIM = 128
EMOTION_EMB_DIM = 8
GRU_UNITS = 512
DROPOUT_RATE = 0.5  # Aumentato leggermente per regolarizzare

EPOCHS = 100  # Puoi aumentare il numero di epoche
BATCH_SIZE = 64

# ============================================
# ======== FUNZIONI DI DATA AUGMENTATION =======
# ============================================
def transpose_sequence(seq, offset):
    """
    Traspone una sequenza (lista di token) di un certo offset.
    I token rappresentano pitch+1. Ritorna None se la trasposizione
    esce dal range [1, 128].
    """
    transposed = []
    for token in seq:
        if token in (PAD_TOKEN, START_TOKEN, EOS_TOKEN):
            transposed.append(token)
        else:
            # token = pitch + 1 → pitch = token - 1
            new_pitch = (token - 1) + offset
            if new_pitch < 0 or new_pitch > 127:
                return None  # Trasposizione non valida
            transposed.append(new_pitch + 1)
    return transposed

def augment_data(sequences, labels, offsets=(-3, -2, -1, 0, 1, 2, 3)):
    """
    Per ogni sequenza, aggiunge le versioni trasposte (esclude l'offset 0 se già presente).
    Ritorna le nuove sequenze e le etichette corrispondenti.
    """
    augmented_seqs = []
    augmented_labels = []
    for seq, lab in zip(sequences, labels):
        for offset in offsets:
            transposed = transpose_sequence(seq, offset)
            if transposed is not None:
                augmented_seqs.append(transposed)
                augmented_labels.append(lab)
    return augmented_seqs, np.array(augmented_labels)

# ============================================
# ========== FUNZIONI PER LA PREPARAZIONE ======
# ============================================
def get_notes(emotion):
    notes = []
    files = glob.glob(f"data_midi/{emotion}/*.mid")
    for file in tqdm(files, desc=f"Elaborazione files {emotion}"):
        try:
            midi = converter.parse(file)
        except Exception as e:
            print(f"Errore nel parsing di {file}: {e}")
            continue
        try:
            s2 = instrument.partitionByInstrument(midi)
            if s2:
                notes_to_parse = s2.parts[0].recurse()
            else:
                notes_to_parse = midi.flat.notes
        except Exception as e:
            notes_to_parse = midi.flat.notes
        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                if len(element.pitches) == 1:
                    notes.append(str(element.pitches[0]))
                else:
                    notes.append('.'.join(str(n) for n in element.normalOrder))
    return notes

def get_all_notes_and_emotions():
    all_notes = []
    all_emotions = []
    for emotion in EMOTIONS:
        notes = get_notes(emotion)
        all_notes.extend(notes)
        all_emotions.extend([MACRO_MAPPING[emotion]] * len(notes))
    return all_notes, all_emotions

# ============================================
# ============ PREPARAZIONE DEI DATI =========
# ============================================
all_notes, all_emotions = get_all_notes_and_emotions()

# Tokenizzazione delle note
unique_notes = sorted(set(all_notes))
note_to_int = {n: i for i, n in enumerate(unique_notes)}
notes_as_int = [note_to_int[n] for n in all_notes]

# Codifica delle macro etichette
label_encoder = LabelEncoder()
encoded_emotions = label_encoder.fit_transform(all_emotions)  # Dovrebbero essere 2 classi

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

# Applica la data augmentation tramite trasposizione
augmented_notes, augmented_emotions = augment_data(network_input_notes, network_input_emotions)
# Combina i dati originali con quelli aumentati
network_input_notes = np.array(network_input_notes + augmented_notes)
network_input_emotions = np.array(list(network_input_emotions) + list(augmented_emotions))
network_output = np.array(network_output + network_output[:len(augmented_notes)])  # Per semplicità, replica output

n_vocab_notes = len(unique_notes)
n_vocab_emotions = len(label_encoder.classes_)  # Dovrebbe essere 2

# Conversione dell'output in formato one-hot
network_output = to_categorical(network_output, num_classes=n_vocab_notes)

print("Dimensioni input note:", network_input_notes.shape)
print("Dimensioni input emozioni:", network_input_emotions.shape)
print("Dimensioni output:", network_output.shape)

# ============================================
# ========== LAYER DI ATTENTION ============
# ============================================
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], input_shape[-1]),
                                 initializer="random_normal", trainable=True)
        self.b = self.add_weight(name="att_bias", shape=(input_shape[-1],),
                                 initializer="zeros", trainable=True)
        self.V = self.add_weight(name="att_var", shape=(input_shape[-1], 1),
                                 initializer="random_normal", trainable=True)
        super(AttentionLayer, self).build(input_shape)
    def call(self, inputs):
        # inputs shape: (batch, time, features)
        score = tf.keras.backend.tanh(tf.keras.backend.dot(inputs, self.W) + self.b)
        attention_weights = tf.keras.backend.softmax(tf.keras.backend.dot(score, self.V), axis=1)
        context_vector = attention_weights * inputs
        context_vector = tf.keras.backend.sum(context_vector, axis=1)
        return context_vector

# ============================================
# ============= DEFINIZIONE DEL MODELLO ======
# ============================================
def create_network(input_shape_notes, input_shape_emotions, n_vocab_notes, n_vocab_emotions):
    # Input per le note
    notes_input = Input(shape=(input_shape_notes[1],), name='notes_input')
    notes_embedding = Embedding(input_dim=n_vocab_notes, output_dim=NOTE_EMB_DIM, input_length=SEQUENCE_LENGTH)(notes_input)
    
    # Input per le emozioni (macro etichetta)
    emotions_input = Input(shape=(input_shape_emotions[1],), name='emotions_input')
    emotions_embedding = Embedding(input_dim=n_vocab_emotions, output_dim=EMOTION_EMB_DIM, input_length=SEQUENCE_LENGTH)(emotions_input)
    
    # Concatenazione degli embedding
    x = Concatenate()([notes_embedding, emotions_embedding])
    
    # Due layer GRU
    x = GRU(GRU_UNITS, return_sequences=True)(x)
    x = GRU(GRU_UNITS, return_sequences=True)(x)
    
    # Layer di Attention per focalizzare il modello
    context = AttentionLayer()(x)
    
    # Passa il contesto al dense
    x = Dropout(DROPOUT_RATE)(context)
    x = Dense(256, activation='relu')(x)
    x = Dropout(DROPOUT_RATE)(x)
    outputs = Dense(n_vocab_notes, activation='softmax')(x)
    
    model = Model([notes_input, emotions_input], outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

# ============================================
# ========= CONTROLLO E CARICAMENTO MODELLO ==
# ============================================
if os.path.exists(MODEL_PATH):
    print("Caricamento del modello esistente...")
    model = load_model(MODEL_PATH, custom_objects={'AttentionLayer': AttentionLayer})
else:
    print("Nessun modello esistente trovato. Addestramento del nuovo modello...")
    model = create_network(network_input_notes.shape, network_input_emotions.shape, n_vocab_notes, n_vocab_emotions)
    model.summary()
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
    lr_reducer = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, verbose=1)
    callbacks_list = [checkpoint, early_stopping, lr_reducer]
    model.fit(
        [network_input_notes, network_input_emotions],
        network_output,
        epochs=100,   # Puoi aumentare questo valore se necessario
        batch_size=BATCH_SIZE,
        callbacks=callbacks_list
    )
    model.save(MODEL_PATH)
    print(f"Modello salvato in: {MODEL_PATH}")

# ============================================
# ============= GENERAZIONE DELLA MELODIA ======
# ============================================
# Dizionario inverso per le note
int_to_note = {i: n for n, i in note_to_int.items()}

def nucleus_sampling(preds, p=0.9):
    preds = np.asarray(preds).astype('float64')
    sorted_indices = np.argsort(preds)[::-1]
    cumulative_probs = np.cumsum(preds[sorted_indices])
    cutoff = np.where(cumulative_probs >= p)[0][0] + 1
    top_indices = sorted_indices[:cutoff]
    top_preds = preds[top_indices]
    top_preds = top_preds / np.sum(top_preds)
    return np.random.choice(top_indices, p=top_preds)

def generate_sequence(model, emotion, max_seq_length, temperature=1.0):
    generated = [START_TOKEN]
    for i in range(max_seq_length - 1):
        current_seq = pad_sequences([generated], maxlen=(max_seq_length-1), padding='post', value=PAD_TOKEN)
        preds = model.predict([current_seq, np.array([[emotion]])], verbose=0)
        next_index = len(generated) - 1
        logits = preds[0, next_index]
        logits = np.asarray(logits).astype('float64') / temperature
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        next_token = np.random.choice(len(probs), p=probs)
        if len(generated) == 1 and next_token == EOS_TOKEN:
            sorted_indices = np.argsort(probs)[::-1]
            for idx in sorted_indices:
                if idx != EOS_TOKEN:
                    next_token = idx
                    break
        if next_token == EOS_TOKEN:
            break
        generated.append(next_token)
    return generated

def generate_notes_by_emotion(model, network_input_notes, network_input_emotions,
                              int_to_note, n_vocab_notes, desired_emotion, num_notes=100, temperature=1.0):
    # Codifica la macro etichetta desiderata
    emotion_encoded = label_encoder.transform([desired_emotion])[0]
    start = np.random.randint(0, len(network_input_notes) - 1)
    pattern_notes = network_input_notes[start]
    pattern_emotions = network_input_emotions[start]
    pattern_emotions[:] = emotion_encoded
    prediction_output = []
    for _ in range(num_notes):
        pred_input_notes = np.reshape(pattern_notes, (1, len(pattern_notes)))
        pred_input_emotions = np.reshape(pattern_emotions, (1, len(pattern_emotions)))
        prediction = model.predict([pred_input_notes, pred_input_emotions], verbose=0)[0]
        index = nucleus_sampling(prediction, p=0.9)
        result = int_to_note[index]
        prediction_output.append(result)
        pattern_notes = np.append(pattern_notes[1:], index)
        pattern_emotions = np.append(pattern_emotions[1:], emotion_encoded)
    return prediction_output

def create_midi(prediction_output, output_filename='output.mid'):
    offset = 0
    output_notes = []
    for pattern in prediction_output:
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes_list = []
            for current_note in notes_in_chord:
                try:
                    new_note = note.Note(int(current_note))
                except:
                    new_note = note.Note(current_note)
                new_note.storedInstrument = instrument.Piano()
                notes_list.append(new_note)
            new_chord = chord.Chord(notes_list)
            new_chord.offset = offset
            output_notes.append(new_chord)
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)
        offset += 0.5
    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp=output_filename)
    print(f"Melodia generata e salvata in {output_filename}")

# ============================================
# ======= GENERAZIONE DELLA MELODIA ==========
# ============================================
# Scegli la macro etichetta desiderata: "positive_emotion" o "negative_emotion"
desired_emotion = 'positive_emotion'
prediction_output = generate_notes_by_emotion(
    model,
    network_input_notes,
    network_input_emotions,
    int_to_note,
    VOCAB_SIZE,
    desired_emotion,
    num_notes=100,
    temperature=0.8
)
output_filename = os.path.join(RESULTS_DIR, f'melody_{desired_emotion}.mid')
create_midi(prediction_output, output_filename)