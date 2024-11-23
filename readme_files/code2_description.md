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
    files = glob.glob(f"data_midi/{emotion}/*.mid") # usa glob per individuare tutti i files midi all'interno della cartella {emotions}.
    for file in tqdm(files, desc=f"Processing {emotion} files"): # cicle over all the files
        midi = converter.parse(file) # use a music21 function to convert the midi file in a structured type suitable for processing.

        try:
            # if the midi File has different instruments:
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse() # extract notes and chords
        except:
            # if the midi file has just one instrument:
            notes_to_parse = midi.flat.notes # extract notes and chords

        for element in notes_to_parse: # each element can be a single note or a chord
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                if len(element.pitches) == 1: # single note chord
                    notes.append(str(element.pitches[0]))
                else:
                    notes.append('.'.join(str(n) for n in element.normalOrder)) # multiple note chord
    return notes
```

### Collecting All Notes and Emotions
```python
def get_all_notes_and_emotions():
    all_notes = []
    all_emotions = []
    for emotion in EMOTIONS:
        notes = get_notes(emotion)
        all_notes.extend(notes) # similar to .append, but adds each element of notes individually to all_notes.
        all_emotions.extend([emotion]*len(notes)) # similar to .append, but adds each element of notes individually to all_notes.
    return all_notes, all_emotions

all_notes, all_emotions = get_all_notes_and_emotions()
```

### Tokenization of Notes and Emotions
Convert each unique note or chord into a unique integer ID. This numerical representation is necessary for the neural network, which cannot process textual or symbolic data directl
- **Notes**: Convert the notes into integers using a mapping dictionary.
- **Emotions**: Encode the emotions using `LabelEncoder` from scikit-learn.
```python
# Tokenization of notes
unique_notes = sorted(set(all_notes)) # crea un set di note, contenente ogni nota solo una volta. 
note_to_int = {note: number for number, note in enumerate(unique_notes)} # crea un mapping tra le note presenti in unique_notes ed un semplice set di ID 1,2,3,...
notes_as_int = [note_to_int[note] for note in all_notes] # applica il mapping ottenendo notes_as_int

# Encoding of emotions
label_encoder = LabelEncoder() # trasforma le etichette in formato numerico
encoded_emotions = label_encoder.fit_transform(all_emotions) # trasforma le etichette numeriche in formato one-hot.
```

### Creating Input and Output Sequences
We create fixed-length sequences (`SEQUENCE_LENGTH`) for the model's input and output.
```python
network_input = [] #  Stores sequences of notes (input for the model).
network_output = [] #  Stores the next note (target output for the model).
network_emotion = [] # Stores the emotion associated with each sequence (input for conditioning).

for i in range(len(notes_as_int) - SEQUENCE_LENGTH):
    seq_in = notes_as_int[i:i + SEQUENCE_LENGTH]
    seq_out = notes_as_int[i + SEQUENCE_LENGTH]
    emotion = encoded_emotions[i + SEQUENCE_LENGTH]
    network_input.append(seq_in)
    network_output.append(seq_out)
    network_emotion.append(emotion)
```
The input and output sequences created in this section are used later in the code as the training data for the neural network.

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
network_input = network_input.reshape((network_input.shape[0], SEQUENCE_LENGTH, 1)) # Adds a third dimension to match the expected input format for recurrent neural networks (RNNs like GRUs or LSTMs).

# Normalization and repetition of emotion
emotion_normalized = network_emotion / float(max(network_emotion)) 
emotion_input = emotion_normalized.reshape(-1, 1, 1) # Neural networks process inputs in 2D or 3D arrays (depending on the architecture)
emotion_input = np.repeat(emotion_input, SEQUENCE_LENGTH, axis=1) # Ripetiamo l'emozione per ogni nota della sequenza (ogni sequenza è caratterizzata da una singola emozione)

# Concatenation of notes and emotions as features
network_input = np.concatenate((network_input, emotion_input), axis=2) # concateniamo le strutture contenenti note ed emozioni così che ad ogni nota, alla "cella sotto" corrisponda la sua emozione.

# Conversion of output to categorical
network_output = to_categorical(network_output, num_classes=n_vocab) # converte l'output in formato numerico invece che simbolico per poter essere processato dalla neural network.
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
Where:
* Input dimensions: The model processes 50,000 input sequences. Each sequence is 50 time steps long (notes). Each time step has 2 features: the note (normalized) and the emotion (normalized and repeated across the sequence).
* Output dimensions: The model produces 50,000 target outputs, one for each input sequence. Each output is represented as a vector of length 685, corresponding to the total number of unique notes (vocab size) in the dataset. Each output is a probability distribution over these 685 notes, generated by the Dense layer of the model with a softmax activation.

## Model Definition
This model defines a Recurrent Neural Network (RNN) architecture using Gated Recurrent Units (GRU), which are a type of RNN layer.
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
### Creating an Inverse Dictionary
During training, notes are converted to integers (note_to_int) for numerical processing. Now, during generation, the model predicts integers as outputs. These integers need to be converted back into their corresponding notes or chords to make the predictions understandable.
```python
int_to_note = {number: note for note, number in note_to_int.items()}
```
### Temperature Sampling Function Definition
The *sample_with_temperature* function adds controlled randomness to the model’s predictions. It helps generate more creative melodies by adjusting the influence of less likely notes in the probability distribution.

```
def sample_with_temperature(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64') 
    if temperature == 0:
        temperature = 1e-10
    preds = np.log(preds + 1e-10) / temperature # Smooths the predictions to make smaller probabilities more significant.
    exp_preds = np.exp(preds) # Converts the scaled log values back into regular probabilities.
    preds = exp_preds / np.sum(exp_preds) # Ensures the probabilities sum to 1, forming a valid distribution.
    probas = preds
    return np.random.choice(len(probas), p=probas) # Chooses one of the notes based on the adjusted probability distribution (probas).
```

### Function to Generate Notes Based on Emotion
We define a function that generates a sequence of notes based on a given emotion using a trained model. It combines the randomness of temperature sampling with emotion conditioning to produce expressive melodies.

```python
def generate_notes_by_emotion(model, network_input, int_to_note, n_vocab, desired_emotion, num_notes=100, temperature=1.0):
    # Encode and normalize the desired emotion
    emotion_encoded = label_encoder.transform([desired_emotion])[0]
    emotion_normalized = emotion_encoded / float(max(label_encoder.transform(label_encoder.classes_)))

    # Randomly chooses an existing sequence (pattern) from the input data (network_input) to initialize the melody generation.
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
