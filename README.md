# Introduction
During the Artificial Intelligence course at AMU University (Poznan), we analyzed how AI and intelligent systems can be used to solve logical problems of various kinds. However, when applying these systems to music, an intriguing challenge arises: musical composition is not a purely "logical" task, even though it has a strong mathematical component. This led me to reflect on how complex it is for an AI system to understand and express emotions, a crucial element in the musical domain.

For these reasons, I applied the techniques that we studied to two problems:  
  1. The classification of emotions conveyed by musical tracks.  
  2. The generation of melodies conditioned by specific emotions.  

# System Configuration
To execute the scripts we will discuss later, we need to install several libraries. Use the following command to install them:
```
pip install pandas librosa numpy joblib tqdm scikit-learn tensorflow music21
```
I used the following datasets:
* Dataset for the first script: [DEAM](http://cvml.unige.ch/databases/DEAM/).
* Dataset for the second script: [EMOPIA](https://zenodo.org/records/5090631#.YPPo-JMzZz8).  
*(You can find additional information about the two datasets [here](readme_files/dataset_info.md))*

# CODE1: Song Emotion Analysis

This project implements a classifier of emotions (**happiness**, **sadness**, **calm**, and **anger**) based on audio features extracted from musical tracks.

![Code1 scheme](readme_files/code1_scheme.png)

The dataset is derived from the DEAM annotation set, and the preprocessing includes the extraction of various audio features (MFCC, Mel-spectrogram, Chroma, Spectral Contrast, Tonnetz, Tempo, Zero Crossing Rate, Spectral Centroid, Spectral Rolloff, and RMS Energy). A data augmentation strategy balances the dataset by generating new samples for underrepresented classes.

## Model Architecture

The model is a **hybrid CNN-LSTM** network:
- **TimeDistributed CNN** for local feature extraction on each frame
- **MaxPooling** and **Dropout** layers to improve regularization
- **LSTM** for capturing temporal dependencies across frames
- **Dense (Softmax)** for the final classification into four emotion classes

During training, the model achieved an accuracy of around **70%** on the test set, successfully predicting the correct emotion for various test files.

# CODE2: Song Emotion Generation
This section explains the process of generating melodies conditioned on specific emotions using a neural network trained on MIDI data. The goal is to create a system that can produce emotionally expressive music by analyzing patterns in the dataset.

![Code2 scheme](readme_files/code2_scheme.png)
*(you can read a deeper analysis of the code [here](readme_files/code2_description.md))*

### Extract Notes
We begin by extracting musical notes and chords from the MIDI files. Each file is associated with one of the predefined emotions: happiness, sadness, or fear (each song is contained in one folder between *"happiness", "sadness"* or *"fear"*).

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

This function extracts notes for all emotions and associates each note with its corresponding emotion label.

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

### Tokenize Notes and Emotions
Convert unique notes into integers using a mapping dictionary to create numerical representations of the music data.

```python
# Tokenization of notes
unique_notes = sorted(set(all_notes))
note_to_int = {note: number for number, note in enumerate(unique_notes)}
notes_as_int = [note_to_int[note] for note in all_notes]
```
Encode emotions into numerical labels using a LabelEncoder.
```python
# Encoding of emotions
label_encoder = LabelEncoder()
encoded_emotions = label_encoder.fit_transform(all_emotions)
```
### Create Input and Output Sequences
We create sequences of fixed length (SEQUENCE_LENGTH) to be used as input for the model. Each sequence includes:
	•	A series of notes.
	•	The corresponding emotion.

The model will predict the next note based on the input sequence and emotion.

```python
network_input = []
network_output = []
network_emotion = []

for i in range(len(notes_as_int) - SEQUENCE_LENGTH):
    # la sequenza di input sono tutte le note analizzate fino ad ora
    seq_in = notes_as_int[i:i + SEQUENCE_LENGTH]
    # la sequenza di output dovrebbe essere la nota successiva e la sua emozione
    seq_out = notes_as_int[i + SEQUENCE_LENGTH]
    emotion = encoded_emotions[i + SEQUENCE_LENGTH]
    network_input.append(seq_in)
    network_output.append(seq_out)
    network_emotion.append(emotion)
```

### Normalize Input
Machine learning frameworks like TensorFlow/Keras require input data to be in array format for training, so we convert the arrays in that format:

```python
# Convert input to numpy arrays
network_input = np.array(network_input)
network_output = np.array(network_output)
network_emotion = np.array(network_emotion)
```

We normalize the note sequences to ensure they are in the range [0, 1]:

```
# Normalization of notes
n_vocab = len(unique_notes) # Total number of unique notes in the dataset
network_input = network_input / float(n_vocab)  # Normalize to [0, 1]
network_input = network_input.reshape((network_input.shape[0], SEQUENCE_LENGTH, 1)) # Reshape for GRU
```
We also normalize and replicate emotion labels to match the sequence length, allowing the model to integrate emotional context:

```
# Normalization and repetition of emotion
emotion_normalized = network_emotion / float(max(network_emotion))
emotion_input = emotion_normalized.reshape(-1, 1, 1)
emotion_input = np.repeat(emotion_input, SEQUENCE_LENGTH, axis=1)
```

Concatenate normalized notes and emotions as input features:

```
# Concatenation of notes and emotions as features
network_input = np.concatenate((network_input, emotion_input), axis=2)
```

The inputnetwork will have 50 "notes-emotions" array, each one on lenght 50000, as shown in the terminal output:

```plaintext
Input dimensions: (50000, 50, 2)
```
# Conversion of output to categorical
Converts the network_output (next note to predict) into a one-hot encoded format. The model outputs probabilities for all possible notes. Using one-hot encoding ensures the training process compares the predicted probability distribution to the correct note.
```
network_output = to_categorical(network_output, num_classes=n_vocab)
```

The dimensions of output sequencies are:

```plaintext
Output dimensions: (50000, 685)
```
where 685 is the ocabulary size (number of unique notes).

### Training 
The model uses GRU layers to process sequential data, with dense layers for output prediction. The structure of the model is:
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
We train the model:
```python
history = model.fit(network_input, network_output, epochs=50, batch_size=32, callbacks=callbacks_list)
```

### Evaluation
This section describes how the model performs during training and how the final results are evaluated.

```plaintext
Epoch 1/50
1563/1563 [==============================] - 233s 148ms/step - loss: 5.0591
Epoch 1: loss improved from inf to 5.0591, saving model to saved_models/audio_generation.keras
...
Epoch 50/50
1563/1563 [==============================] - 243s 156ms/step - loss: 3.4806
Epoch 50: loss did not improve from 3.49500
```
Loss: Measures how well the model’s predictions match the expected output during training. Lower loss indicates better performance.

The model saves its weights when the loss improves. This ensures only the best-performing version of the model is kept.

### Melody Generation
The trained model is now used to generate new melodies based on a specified emotion. 

#### Initialization
Firstly we convert the predicted integers (note indices) back into readable note/chord names:

```python
int_to_note = {number: note for note, number in note_to_int.items()}
```
The next function is used to add randomness and creativity to the predictions made by the model when generating music. Without randomness, the model would always choose the most likely note, resulting in repetitive and predictable melodies.
```
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
How randomness improves creativity:
	•	The temperature parameter controls the level of randomness:
	•	High temperature: Introduces more variety by allowing less probable notes to be selected.
	•	Low temperature: Reduces randomness, focusing on the most likely notes and producing safer, more predictable melodies.

##### Generating Notes
The model generates notes based on a given emotion and sequence:

```python
def generate_notes_by_emotion(model, network_input, int_to_note, n_vocab, desired_emotion, num_notes=100, temperature=1.0):
    ...
```
Convert the desired emotion into a normalized numerical format (e.g., “happiness” → 0.5). Ensure the generated sequence aligns with the specified emotion.
```
    ...
    # Encode and normalize the desired emotion
    emotion_encoded = label_encoder.transform([desired_emotion])[0]
    emotion_normalized = emotion_encoded / float(max(label_encoder.transform(label_encoder.classes_)))
    ...
```
Choose a random sequence (pattern) from the training data as the starting point.
```
    ...
    # Choose a random starting point
    start = np.random.randint(0, len(network_input)-1)
    pattern = network_input[start]
    pattern = pattern.reshape(1, SEQUENCE_LENGTH, 2)
    ...
```
Predict the next note by feeding the current pattern to the model.
	•	Use temperature sampling to add creativity.
	•	Append the predicted note to prediction_output.
```
    ...
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
        ...
```
Slide the pattern window forward by adding the predicted note and removing the oldest note.
```
        ...
        pattern = np.concatenate((pattern[:, 1:, :], new_input.reshape(1, 1, 2)), axis=1)
    return prediction_output
```
##### Creation of the MIDI file
We create a midi file:

```python
def create_midi(prediction_output, output_filename='output.mid'):
    offset = 0
    output_notes = []

    for pattern in prediction_output:
        ...
```
Determine whether each pattern is a single note or a chord and convert patterns into MIDI-compatible notes/chords.

```
        ...
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
        ...
```
Adjust the time spacing between notes to avoid overlapping.
```
        ...
        # Increase offset to prevent notes from overlapping
        offset += 0.5
        ...
```
Write the generated sequence into a .mid file for playback:

```
    ...
    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp=output_filename)
    print(f"Melody generated and saved to {output_filename}")
```
##### Running the Generator
We finally can run the generator:
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
```
and save the result in a midi file:
```
# Create the MIDI file
output_filename = os.path.join(RESULTS_DIR, f'melody_{desired_emotion}.mid')
create_midi(prediction_output, output_filename)
```
**Output:**
```plaintext
Melody generated and saved to generation_results/melody_sadness.mid
```
### Results 
The results of the generation are in this [folder](generation_results)