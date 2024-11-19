This script aims to generate musical melodies conditioned by different emotions using Recurrent Neural Networks (RNN) and the Keras library. The goal is to train a model capable of producing musical sequences that reflect specific emotions such as happiness, sadness, or fear.

## Dependencies
Ensure the following libraries are installed:
- `numpy`
- `music21`
- `tensorflow`
- `sklearn`
- `tqdm`

You can install them using `pip`:
```bash
pip install numpy music21 tensorflow scikit-learn tqdm
```

## Data Preparation
### Note Extraction
For each emotion, we extract the notes from the MIDI files located in the folder `data_midi/<emotion>/`. We use the `music21` library to read the MIDI files and extract notes and chords.
```python
def get_notes(emotion):
    notes = []
    files = glob.glob(f"data_midi/{emotion}/*.mid")
    for file in tqdm(files, desc=f"Processing {emotion} files"):
        midi = converter.parse(file)

        try:
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse()
        except:
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
```

### Collecting All Notes and Emotions
```python
def get_all_notes_and_emotions():
    all_notes = []
    all_emotions = []
    for emotion in EMOTIONS:
        notes = get_notes(emotion)
        all_notes.extend(notes)
        all_emotions.extend([emotion]*len(notes))
    return all_notes, all_emotions

all_notes, all_emotions = get_all_notes_and_emotions()
```

### Tokenization of Notes and Emotions
- **Notes**: Convert the notes into integers using a mapping dictionary.
- **Emotions**: Encode the emotions using `LabelEncoder` from scikit-learn.
```python
# Tokenization of notes
unique_notes = sorted(set(all_notes))
note_to_int = {note: number for number, note in enumerate(unique_notes)}
notes_as_int = [note_to_int[note] for note in all_notes]

# Encoding of emotions
label_encoder = LabelEncoder()
encoded_emotions = label_encoder.fit_transform(all_emotions)
```

### Creating Input and Output Sequences
We create fixed-length sequences (`SEQUENCE_LENGTH`) for the model's input and output.
```python
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
```

### Limiting the Number of Patterns
To manage computational resources, we limit the number of patterns used for training.
```python
# Limit the number of patterns
network_input = network_input[:MAX_PATTERNS]
network_output = network_output[:MAX_PATTERNS]
network_emotion = network_emotion[:MAX_PATTERNS]
```

### Final Data Preparation
- **Normalization of notes**
- **Preparation of emotion input**
- **Concatenation of notes and emotions**
```python
# Convert input to numpy arrays
network_input = np.array(network_input)
network_output = np.array(network_output)
network_emotion = np.array(network_emotion)

# Normalization of notes
n_vocab = len(unique_notes)
network_input = network_input / float(n_vocab)
network_input = network_input.reshape((network_input.shape[0], SEQUENCE_LENGTH, 1))

# Normalization and repetition of emotion
emotion_normalized = network_emotion / float(max(network_emotion))
emotion_input = emotion_normalized.reshape(-1, 1, 1)
emotion_input = np.repeat(emotion_input, SEQUENCE_LENGTH, axis=1)

# Concatenation of notes and emotions as features
network_input = np.concatenate((network_input, emotion_input), axis=2)

# Conversion of output to categorical
network_output = to_categorical(network_output, num_classes=n_vocab)
```

### Verifying Array Dimensions
```python
print("Input dimensions:", network_input.shape)  # Should be (number_of_samples, SEQUENCE_LENGTH, 2)
print("Output dimensions:", network_output.shape)
```

**Output:**
```plaintext
Input dimensions: (50000, 50, 2)
Output dimensions: (50000, 685)
```

## Model Definition
We define a recurrent neural network using GRU layers with dropout to prevent overfitting.
```python
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
```

## Training the Model
### Checking for Existing Model
If a previously trained model exists, we load it; otherwise, we proceed with training.
```python
if os.path.exists(MODEL_PATH):
    print("Loading existing model...")
    model = load_model(MODEL_PATH)
else:
    print("No existing model found. Training a new model...")
    # Create the model
    model = create_network(network_input.shape, n_vocab)
    model.summary()
```

### Model Summary Output
```plaintext
Model: "functional"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape         ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ input_layer (InputLayer)             │ (None, 50, 2)        │               0 │
├──────────────────────────────────────┼──────────────────────┼─────────────────┤
│ gru (GRU)                            │ (None, 50, 256)      │         199,680 │
├──────────────────────────────────────┼──────────────────────┼─────────────────┤
│ gru_1 (GRU)                          │ (None, 256)          │         394,752 │
├──────────────────────────────────────┼──────────────────────┼─────────────────┤
│ dropout (Dropout)                    │ (None, 256)          │               0 │
├──────────────────────────────────────┼──────────────────────┼─────────────────┤
│ dense (Dense)                        │ (None, 128)          │          32,896 │
├──────────────────────────────────────┼──────────────────────┼─────────────────┤
│ dropout_1 (Dropout)                  │ (None, 128)          │               0 │
├──────────────────────────────────────┼──────────────────────┼─────────────────┤
│ dense_1 (Dense)                      │ (None, 685)          │          88,365 │
└──────────────────────────────────────┴──────────────────────┴─────────────────┘
 Total params: 715,693 (2.73 MB)
 Trainable params: 715,693 (2.73 MB)
 Non-trainable params: 0 (0.00 B)
```

### Defining Callbacks and Training
We use `ModelCheckpoint` to save the best model and `EarlyStopping` to halt training if the model doesn't improve.
```python
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

# Training the model
history = model.fit(network_input, network_output, epochs=50, batch_size=32, callbacks=callbacks_list)
```

### Training Output
During training, the model prints the loss for each epoch and saves the model if it improves.
```plaintext
Epoch 1/50
1563/1563 [==============================] - 233s 148ms/step - loss: 5.0591
Epoch 1: loss improved from inf to 5.0591, saving model to saved_models/audio_generation.keras
...
Epoch 50/50
1563/1563 [==============================] - 243s 156ms/step - loss: 3.4806
Epoch 50: loss did not improve from 3.49500
```

## Melody Generation
### Preparation for Generation
We create an inverse dictionary to convert integers back to notes and define a function for temperature sampling.
```python
int_to_note = {number: note for note, number in note_to_int.items()}

def sample_with_temperature(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    if temperature == 0:
        temperature = 1e-10
    preds = np.log(preds + 1e-10) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = preds
    return np.random.choice(len(probas), p=probas)
```

### Function to Generate Notes Based on Emotion
```python
def generate_notes_by_emotion(model, network_input, int_to_note, n_vocab, desired_emotion, num_notes=100, temperature=1.0):
    # Encode and normalize the desired emotion
    emotion_encoded = label_encoder.transform([desired_emotion])[0]
    emotion_normalized = emotion_encoded / float(max(label_encoder.transform(label_encoder.classes_)))

    # Choose a random starting point
    start = np.random.randint(0, len(network_input)-1)
    pattern = network_input[start]
    pattern = pattern.reshape(1, SEQUENCE_LENGTH, 2)

    # Set the desired emotion in the pattern
    pattern[0, :, 1] = emotion_normalized

    prediction_output = []

    for note_index in range(num_notes):
        prediction = model.predict(pattern, verbose=0)
        # Use temperature sampling
        index = sample_with_temperature(prediction[0], temperature)
        result = int_to_note[index]
        prediction_output.append(result)

        # Create new input
        new_note = index / float(n_vocab)
        new_input = np.array([[new_note, emotion_normalized]])
        pattern = np.concatenate((pattern[:, 1:, :], new_input.reshape(1, 1, 2)), axis=1)
    return prediction_output
```

### Creating the MIDI File
```python
def create_midi(prediction_output, output_filename='output.mid'):
    offset = 0
    output_notes = []

    for pattern in prediction_output:
        # If the pattern is a chord
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
            # If the pattern is a single note
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)
        # Increase offset to prevent notes from overlapping
        offset += 0.5

    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp=output_filename)
    print(f"Melody generated and saved to {output_filename}")
```

### Generating the Melody
We select the desired emotion and generate the melody.
```python
# Select the desired emotion
desired_emotion = 'sadness'  # Can be 'happiness', 'sadness', or 'fear'

# Generate the sequence
prediction_output = generate_notes_by_emotion(
    model,
    network_input,
    int_to_note,
    n_vocab,
    desired_emotion,
    num_notes=200,  # Number of notes to generate
    temperature=0.8  # Temperature value for sampling
)

# Create the MIDI file
output_filename = os.path.join(RESULTS_DIR, f'melody_{desired_emotion}.mid')
create_midi(prediction_output, output_filename)
```

**Output:**
```plaintext
Melody generated and saved to generation_results/melody_sadness.mid
```

## Results
The model successfully generated a melody based on the selected emotion. The resulting MIDI file can be found in the `generation_results/` folder with the name `melody_sadness.mid`.