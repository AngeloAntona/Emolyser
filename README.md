# 0. Introduction
```
(0 - 2 minutes: explanation of the branching definitions Artificial Intelligence -> Machine Learning -> Neural Network -> Generative AI).
```
During the course, we analyzed how AI and intelligent systems can be used to solve logical problems of various kinds. However, when applying these systems to music, an intriguing challenge arises: musical composition is not a purely "logical" task, even though it has a strong mathematical component. This led me to reflect on how complex it is for an AI system to understand and express emotions, a crucial element in the musical domain.

Among the various artificial intelligence systems available  
![Ai fields diagram](readme_files/AI_fields_diagram.png)  
*(For an overview of other types of AI, you can visit [here](readme_files/AI_fields.md))*  
I chose to use machine learning systems, with a particular focus on neural networks, to explore the ability of simple learning models to replicate emotions.

The choice is based on the flexibility of neural networks in modeling complex phenomena and their ability to learn rich representations from data. Specifically, I applied these techniques to two problems:  
  1. The classification of emotions conveyed by musical tracks.  
  2. The generation of melodies conditioned by specific emotions.  

This exploration seeks to address a central question: can an AI system, based on relatively simple algorithms, not only "understand" but also "express" emotions through music?

# System Configuration
To execute the scripts we will discuss later, we need to install several libraries. Use the following command to install them:
```
pip install pandas librosa numpy joblib tqdm scikit-learn tensorflow music21
```
The libraries used are:
1. **NumPy:** A fundamental library for fast numerical operations and multidimensional arrays, essential for scientific computation.
2. **Pandas:** Crucial for data manipulation and analysis, such as reading CSV files and managing structured tables.
3. **Scikit-learn:** Provides tools for dataset splitting, label encoding, and other basic machine learning functionalities.
4. **TensorFlow:** An advanced deep learning library used to build, train, and save neural network models.
5. **Librosa:** Used for processing audio files, including feature extraction (e.g., MFCC). It is key for audio analysis.
6. **Music21:** Necessary for handling MIDI files, analyzing notes and chords, and creating melodies.
7. **Joblib:** Used for parallel processing to speed up data handling or complex functions.
8. **TQDM:** Adds a progress bar to monitor iterative processes, especially useful for large datasets.
9. **Pickle:** Part of Python's standard library; no installation required. Used for saving and loading objects, such as label encoders.
10. **OS and Glob:** Both are part of Python's standard library. OS is used for file and directory management, while Glob works with file name patterns.
11. **Collections (Counter):** Another standard Python library, used for operations like counting elements in a list.

Additionally, we will use the following datasets:
* Dataset for the first script: [DEAM](http://cvml.unige.ch/databases/DEAM/).
* Dataset for the second script: [EMOPIA](https://zenodo.org/records/5090631#.YPPo-JMzZz8).  
*(You can find additional information about the two datasets [here](readme_files/dataset_info.md))*

# Song Emotion Analysis Code
The goal is to create and train a model that recognizes the primary type of emotion a given song conveys to the listener. For simplicity, we will use a very basic set of emotions, categorizing the entire emotional spectrum into just the labels of **happiness**, **sadness**, and **fear**.

![Code1 scheme](readme_files/Emotion_analysis_scheme.png)
*(you can read a deeper analysis of the code [here](readme_files/code1_description.md))*
### 1. Data Loading
``` Python
annotations = pd.read_csv('data/annotations/annotations averaged per song/song_level/static_annotations_averaged_songs_1_2000.csv')
```
### 2. Emotion Assignment
We define a function to map the valence and arousal values to the emotions:
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
and we apply that function to each song:
``` Python
annotations['emotion'] = annotations.apply(assign_emotion, axis=1)
```
The terminal output at this point is:
```
    song_id  valence_mean   valence_std   arousal_mean   arousal_std    emotion
0        2            3.1          0.94            3.0          0.63    sadness
3        5            4.4          2.01            5.3          1.85       fear
4        7            5.8          1.47            6.4          1.69  happiness
```
and the distribution of emotions in the dataset is:
```
emotion
sadness      729
happiness    582
fear         210
```
### 3. Features Extraction (MFCC)
We define a function to extract the songs features:
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
We apply the previously mentioned function to all the audio files:
```
100%|████████████████████████████████| 1521/1521 [03:45<00:00, 6.76it/s]
Numero di campioni con caratteristiche estratte: 1521
```
### 3. Data Preparation
We now have the **features** matrix and the **labels** vector. We can create a DataFrame to serve as input for our neural network:
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
We derive **X** and **yy**:
```Python
# Otteniamo X
X = np.array(features_df['feature'].tolist())
# Otteniamo yy
y = np.array(features_df['label'].tolist())
le = LabelEncoder()
yy = to_categorical(le.fit_transform(y))
```
We partition the elements to obtain a *train set* and a *test set*:
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
# 4. Creazione della rete neurale
Si passa poi alla rete neurale vera e propria. La semplice struttura scelta per la rete è la seguente:
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
nella quale:
* **Nodi di tipo *"dense"*:** sono il fulcro della rete neurale. In essi, ogni neurone è connesso a tutti i neuroni della rete successiva. Questi livelli consentono alla rete di apprendere le relazioni complesse tra le caratteristiche in input e le emozioni target.
* **Nodi di tipo *"activation"*:** sono nodi che implementano funzioni non lineari. In tal modo si fa si che la rete apprenda relazioni complesse nei dati (**ReLU** trasforma le uscite negative in 0, **softmax** converte le uscite del livello finale in probabilità normalizzando l'output da 0 a 1).
* **Nodi di tipo *"dropout"*:** disattivano randomicamente una frazione di neuroni di un livello. Questo permette di evitare l'overfitting, impedendo alla rete di fare troppo affidamento su specifici percorsi di neuroni.
In sintesi, la struttura della rete neurale è la seguente:
* **Livello iniziale**: Riceve l’input e lo elabora nei 256 neuroni, ampliando la rappresentazione dei dati originali.
* **Livello intermedio**: Riduce l’informazione a 128 unità, mantenendo solo le caratteristiche essenziali.
* **Livello finale**: Emette una previsione finale su quale emozione appartiene al dato in input, rappresentata in termini di probabilità.
# 5. Addestramento
L’**addestramento di un modello di deep learning** è il processo in cui il modello apprende dai dati di addestramento per fare previsioni accurate. In pratica, il modello viene sottoposto ripetutamente ai dati di addestramento per migliorare la sua capacità di riconoscere pattern. Durante ogni fase di addestramento, il modello ottimizza i suoi parametri (pesi e bias) per ridurre la differenza tra le sue predizioni e le etichette corrette.
```Python
checkpointer = ModelCheckpoint(filepath='saved_models/audio_classification.keras', verbose=1, save_best_only=True)

history = model.fit(x_train, y_train, batch_size=32, epochs=20, validation_data=(x_test, y_test), callbacks=[checkpointer], verbose=1)
```
Le **epochs** rappresentano il numero di volte in cui il modello percorre l’intero dataset di addestramento. Ogni volta che il modello passa attraverso l’intero dataset, completa una epoch:
* **Forward pass**: Il modello elabora l’input, calcola l’output e confronta la predizione con l’etichetta reale.
* **Calcolo della perdita**: La **perdita** (o “loss”) rappresenta la misura dell’errore tra la predizione e il risultato atteso. Durante l’addestramento, l’obiettivo è minimizzare questa perdita.
* **Backward pass e aggiornamento dei pesi**: Usando la tecnica della retropropagazione backpropagation, l’errore viene retropropagato attraverso il modello per aggiornare i pesi e ridurre la perdita nelle successive iterazioni.

Con più epochs, il modello apprende gradualmente a fare predizioni migliori. Un numero ottimale di epochs permette al modello di apprendere abbastanza senza “memorizzare” i dati di addestramento, garantendo così una buona capacità di generalizzazione.

ll modello in questione è addestrato per **20 epochs**, e per ciascuna epoch l’output mostra informazioni su:
* **accuracy**: La percentuale di predizioni corrette sul set di addestramento.
* **loss**: Il valore della funzione di perdita sui dati di addestramento. Una loss più bassa indica che il modello sta facendo previsioni più accurate.
* **val_accuracy**: Accuratezza del modello sui dati di validazione. Questo valore mostra la capacità di generalizzazione del modello.
* **val_loss**: La perdita calcolata sui dati di validazione. Minimizzare val_loss è fondamentale per evitare che il modello si adatti troppo ai dati di addestramento senza generalizzare bene.
Alla epoch 17 la val_loss raggiunge 0.75070, mostrando un miglioramento continuo. Tuttavia, dal **Epoch 18** in poi, non si registrano ulteriori miglioramenti significativi. Per tale motivo ho deciso di fermare l'addestramento a 20 epochs.
# 6. Test
Alla fine dell’addestramento, usiamo il test set
```Python
model.load_weights('saved_models/audio_classification.keras')
score = model.evaluate(x_test, y_test, verbose=0)
```
per valutare l'accuratezza della predizione. Il risultato ottenuto è:
```
Accuracy sul test set: 70.49%
```
che sembra indicare una certa capacità del sistema nel riconoscere le emozioni delle canzoni.
# 7. Predizione su nuovi dati
Dal dataset di canzoni iniziale ho rimosso i campioni [3.mp3](test/3.mp3) e [4.mp3](test/4.mp3) e ho inserito tali campioni in un'apposita cartella test, separata dagli altri elementi del dataset. Ascoltando tali campioni, 3.mp3 mi trasmette tristezza, invece 4.mp3 mi trasmette felicità. 

Useremo tali due canzoni per testare se la predizione del modello coincide con l'emozione da me percepita all'ascolto di tali due canzoni. L'output a terminale è:
```
L'emozione predetta per 'test/4.mp3' è: happiness

L'emozione predetta per 'test/3.mp3' è: sadness
```
da cui possiamo concludere che, per quanto semplificato sia questo modello di riconoscimento delle emozioni delle canzoni, esso ha una buona capacità di categorizzazione.
