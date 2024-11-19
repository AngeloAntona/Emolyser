# 0. Introduction and Basic Setup
For this "experiment," we will use the dataset available for download [here](http://cvml.unige.ch/databases/DEAM/).  
Additionally, the following libraries need to be installed:
```
pip install pandas librosa numpy joblib tqdm scikit-learn tensorflow
```
The goal is to create and train a model that identifies the primary type of emotion a given song conveys to the listener. For simplicity, we will use a very basic set of emotions, categorizing the entire emotional spectrum into just the labels of **happiness**, **sadness**, and **fear**.

# 1. Data Loading
First, we need to load the dataset and the associated metadata into our program:
``` Python
annotations = pd.read_csv('data/annotations/annotations averaged per song/song_level/static_annotations_averaged_songs_1_2000.csv')
```
The annotations in the DEAM database attempt to define the emotion conveyed by a given song using two measures:
* **Valence:** measures how positive or negative an emotion is. In this database, `valence_mean` ranges from 1 to 10, where:
  * **High values** (above 5) indicate positive emotions (like happiness, excitement).
  * **Low values** (below 5) indicate negative emotions (like sadness, melancholy).
* **Arousal:** measures the intensity or activation of the emotion, representing how energetic or calm an emotion is. `Arousal_mean` also ranges from 1 to 10, where:
  * **High values** (above 5) indicate excited or energetic emotions (like anger, excitement).
  * **Low values** (below 5) indicate calmer or more relaxed emotions (like sadness or serenity).

To associate emotions with pairs of values: ```valence_mean - arousal_mean``` I have defined the function:

``` Python
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
		return None # Esclude altri stati emotivi
```
and applied this function to all rows of the annotations:
``` Python
annotations['emotion'] = annotations.apply(assign_emotion, axis=1)
```
Having done this, the resulting dataset will have a structure like this:
```
    song_id  valence_mean   valence_std   arousal_mean   arousal_std    emotion
0        2            3.1          0.94            3.0          0.63    sadness
3        5            4.4          2.01            5.3          1.85       fear
4        7            5.8          1.47            6.4          1.69  happiness
```
And a distribution of emotions in the dataset that is:
```
emotion
sadness      729
happiness    582
fear         210
```
(*It might be necessary to balance the number of samples across different categories by performing oversampling or subsampling, but for brevity, we will skip this step*).

# 2. Audio Feature Extraction
Using the following function:
``` Python
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
```
we extract the **Mel-frequency Cepstral Coefficients (MFCC)** from the audio files, a common representation of audio features widely used in audio processing and the recognition of emotions, voice, and other acoustic properties. 

The MFCC capture spectral and temporal information from the audio in a compact representation, enabling the neural network to learn meaningful patterns such as tone, intensity, and the articulation of emotions.

The output of the function for each song is an MFCC matrix, where:
* Rows represent the different coefficients (in this case, 40 coefficients for each time segment).
* Columns represent the temporal frames across the audio signal.

We apply the previously mentioned function to all the audio files:
```
100%|████████████████████████████████| 1521/1521 [03:45<00:00, 6.76it/s]
Numero di campioni con caratteristiche estratte: 1521
```
# 3. Preparing Data for Neural Network Input
We now have the **features** matrix and the **labels** vector. We can create a DataFrame, which will serve as the input for our neural network:
``` Python
features_df = pd.DataFrame(features, columns=['feature', 'label'])
```
The data structure will be of the type:
```
	feature                                             label
0  [-144.26477, 123.45465, -21.118523, 36.46806, ...    sadness
1  [-148.90979, 125.012566, -3.8318107, 33.42975,...       fear
2  [-155.02774, 115.9887, 19.196737, 53.457127, 1...  happiness
3  [-294.01443, 138.91287, 71.07505, 36.631863, 8...    sadness
4  [-270.40778, 132.17728, 3.762553, 25.400993, 1...    sadness
...
```
From the previous DataFrame, we derive **X** and **yy**, containing respectively the numerical features of each song and the label of each song (*the labels in ```yy``` are converted into a 2D numerical matrix format where each row is of the type ```100```, ```010```, or ```001```. This format is called "one-hot"*).
```Python
# Otteniamo X
X = np.array(features_df['feature'].tolist())
# Otteniamo yy
y = np.array(features_df['label'].tolist())
le = LabelEncoder()
yy = to_categorical(le.fit_transform(y))
```
From the entire dataset, we partition the elements to obtain a *train set* and a *test set*:
```Python
x_train, x_test, y_train, y_test = train_test_split(X, yy, test_size=0.2, random_state=42)
```
The resulting data structures will be:
```
Forma di X: (1521, 40)
Forma di yy: (1521, 3)
Numero di campioni nel training set: 1216
Numero di campioni nel testing set: 305
```
# 4. Creating the Neural Network
We then move on to the neural network itself. The simple structure chosen for the network is as follows:
```
 Layer (type)                        │ Output Shape                │ Param #
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┿━━━━━━━━━━━
 dense (Dense)                       │ (None, 256)                 │ 10,496
 activation (Activation)             │ (None, 256)                 │ 0
 dropout (Dropout)                   │ (None, 256)                 │ 0
─────────────────────────────────────┼─────────────────────────────┼───────────
 dense_1 (Dense)                     │ (None, 128)                 │ 32,896
 activation_1 (Activation)           │ (None, 128)                 │ 0
 dropout_1 (Dropout)                 │ (None, 128)                 │ 0
─────────────────────────────────────┼─────────────────────────────┼───────────
 dense_2 (Dense)                     │ (None, 3)                   │ 387
 activation_2 (Activation)           │ (None, 3)                   │ 0
```
in which:
* **"Dense" nodes:** are the core of the neural network. In these, each neuron is connected to all neurons in the subsequent layer. These layers allow the network to learn complex relationships between the input features and the target emotions.
* **"Activation" nodes:** implement nonlinear functions. This enables the network to learn complex relationships in the data (**ReLU** transforms negative outputs into 0, **softmax** converts the final layer's outputs into probabilities by normalizing the output to range from 0 to 1).
* **"Dropout" nodes:** randomly deactivate a fraction of neurons in a layer. This prevents overfitting by ensuring the network does not overly rely on specific neuron pathways.

In summary, the neural network structure is as follows:
* **Initial layer:** Receives the input and processes it in 256 neurons, expanding the representation of the original data.
* **Intermediate layer:** Reduces the information to 128 units, retaining only essential features.
* **Final layer:** Outputs a final prediction about which emotion corresponds to the input data, represented in terms of probabilities.

# 5. Training
**Training a deep learning model** is the process where the model learns from the training data to make accurate predictions. Essentially, the model is repeatedly exposed to the training data to improve its ability to recognize patterns. During each training phase, the model optimizes its parameters (weights and biases) to minimize the difference between its predictions and the correct labels.
```Python
checkpointer = ModelCheckpoint(filepath='saved_models/audio_classification.keras', verbose=1, save_best_only=True)

history = model.fit(x_train, y_train, batch_size=32, epochs=20, validation_data=(x_test, y_test), callbacks=[checkpointer], verbose=1)
```
**Epochs** represent the number of times the model iterates over the entire training dataset. Each time the model processes the entire dataset, it completes one epoch:
* **Forward pass:** The model processes the input, calculates the output, and compares the prediction with the actual label.
* **Loss calculation:** **Loss** measures the error between the prediction and the expected result. During training, the goal is to minimize this loss.
* **Backward pass and weight updates:** Using the backpropagation technique, the error is propagated backward through the model to update the weights and reduce loss in subsequent iterations.

With more epochs, the model gradually learns to make better predictions. An optimal number of epochs ensures the model learns sufficiently without "memorizing" the training data, thereby guaranteeing good generalization.

The model is trained for **20 epochs**, and for each epoch, the output provides information on:
* **Accuracy:** The percentage of correct predictions on the training set.
* **Loss:** The value of the loss function on the training data. A lower loss indicates the model is making more accurate predictions.
* **Val_accuracy:** The model's accuracy on the validation data. This value reflects the model's generalization ability.
* **Val_loss:** The loss calculated on the validation data. Minimizing val_loss is crucial to prevent the model from overfitting the training data while maintaining good generalization.

At epoch 17, the val_loss reaches 0.75070, showing continuous improvement. However, from **Epoch 18** onward, no significant improvements are observed. Therefore, I decided to stop the training at 20 epochs.

# 6. Testing
At the end of training, we use the test set
```Python
model.load_weights('saved_models/audio_classification.keras')
score = model.evaluate(x_test, y_test, verbose=0)
```
to evaluate the prediction accuracy. The result obtained is:
```
Accuracy sul test set: 70.49%
```
which seems to indicate a certain ability of the system to recognize the emotions in songs.

# 7. Prediction on New Data
From the initial song dataset, I removed the samples [3.mp3](test/3.mp3) and [4.mp3](test/4.mp3) and placed these samples in a separate test folder, apart from the other dataset elements. Listening to these samples, 3.mp3 conveys sadness to me, whereas 4.mp3 conveys happiness.

We will use these two songs to test if the model's predictions align with the emotions I perceived while listening to them. The terminal output is:
```
L'emozione predetta per 'test/4.mp3' è: happiness

L'emozione predetta per 'test/3.mp3' è: sadness
```
from which we can conclude that, despite the simplicity of this model for recognizing emotions in songs, it has a good categorization ability.