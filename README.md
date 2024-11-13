# Intro e configurazione di base
Per questo "esperimento" useremo il dataset scaricabile [qui][http://cvml.unige.ch/databases/DEAM/]. 
E' inoltre necessario installare le seguenti librerie:
```
pip install pandas librosa numpy joblib tqdm scikit-learn tensorflow
```
L'obiettivo è quello di creare e addestrare un modello che riconosca il tipo di emozione principale che una determinata canzone trasmette all'ascoltatore. Per semplicità useremo un set di emozioni molto molto semplice, catalogando l'intero spettro emotivo nelle sole etichette di **felicità**, **tristezza** e **paura**.
# 1. Caricamento dei dati
Prima di tutto dobbiamo caricare all'interno del nostro programma il dataset e i metadati associati:
``` Python
annotations = pd.read_csv('data/annotations/annotations averaged per song/song_level/static_annotations_averaged_songs_1_2000.csv')
```
Le annotation del database DEAM tentano di definire che emozione è trasmessa da una determinata canzone attraverso due misure:
* **Valence:** misura quanto un’emozione sia positiva o negativa. In questo database, valence_mean varia da 1 a 10, dove:
	* Valori **alti** (sopra 5) indicano emozioni positive (come felicità, eccitazione).
	* Valori **bassi** (sotto 5) indicano emozioni negative (come tristezza, malinconia).
* **Arousal:** misura l’intensità o l’attivazione dell’emozione, rappresentando quanto un’emozione sia energica o calma. Anche arousal_mean varia da 1 a 10, dove:
	* Valori **alti** (sopra 5) indicano emozioni eccitate o energiche (come rabbia, eccitazione).
	* Valori **bassi** (sotto 5) indicano emozioni più calme o rilassate (come tristezza o serenità).
Per associare delle emozioni alle coppie di valori ```valence_mean - arousal_mean``` ho definito la funzione:
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
e applicato tale funzione a tutte le righe delle annotation:
``` Python
annotations['emotion'] = annotations.apply(assign_emotion, axis=1)
```
Fatto ciò, il dataset che otteniamo avrà una struttura del tipo:
```
    song_id  valence_mean   valence_std   arousal_mean   arousal_std    emotion
0        2            3.1          0.94            3.0          0.63    sadness
3        5            4.4          2.01            5.3          1.85       fear
4        7            5.8          1.47            6.4          1.69  happiness
```
E una distribuzione delle emozioni nel dataset che è:
```
emotion
sadness      729
happiness    582
fear         210
```
(*Potrebbe essere necessario equilibrare il numero di campioni delle diverse categorie effettuando un oversampling o un subsampling dei campioni, ma per brevità saltiamo questo passaggio*).
# 2. Estrazione delle caratteristiche audio
Attraverso la seguente funzione:
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
estraiamo dai file audio i **Mel-frequency Cepstral Coefficients (MFCC)**, una rappresentazione comune delle caratteristiche audio ampiamente utilizzata nell’elaborazione audio e nel riconoscimento delle emozioni, della voce e di altre proprietà acustiche. 
Gli MFCC sintetizzano le informazioni spettrali e temporali dell’audio in una rappresentazione compatta, consentendo alla rete neurale di apprendere pattern significativi come tono, intensità, e articolazione delle emozioni.

L'output della funzione per ogni canzone è una matrice di MFCC, dove:
* Le righe rappresentano i diversi coefficienti (in questo caso, 40 coefficienti per ciascun segmento temporale).
* Le colonne rappresentano i frame temporali lungo il segnale audio.

Applichiamo la funzione precedentemente citata a tutti i file audio:
```
100%|████████████████████████████████| 1521/1521 [03:45<00:00, 6.76it/s]
Numero di campioni con caratteristiche estratte: 1521
```
# 3. Preparazione dei dati per l'input della rete neurale
Abbiamo adesso la matrice delle **features** e il vettore delle **etichette**. Ora possiamo creare un DataFrame, che sarà l'input della nostra rete neurale:
``` Python
features_df = pd.DataFrame(features, columns=['feature', 'label'])
```
La struttura dati sarà del tipo:
```
	feature                                             label
0  [-144.26477, 123.45465, -21.118523, 36.46806, ...    sadness
1  [-148.90979, 125.012566, -3.8318107, 33.42975,...       fear
2  [-155.02774, 115.9887, 19.196737, 53.457127, 1...  happiness
3  [-294.01443, 138.91287, 71.07505, 36.631863, 8...    sadness
4  [-270.40778, 132.17728, 3.762553, 25.400993, 1...    sadness
...
```
Dal DataFrame precedente ricaviamo **X** e **yy**, contenenti rispettivamente le caratteristiche numeriche di ciascuna canzone e l'etichetta di ciascuna canzone (*le etichette in ```yy```vengono convertite in un formato numerico matriciale 2D in cui ogni riga è del tipo ```100```, ```010``` o ```001```. Tale formato è detto "one-hot"*).
```Python
# Otteniamo X
X = np.array(features_df['feature'].tolist())
# Otteniamo yy
y = np.array(features_df['label'].tolist())
le = LabelEncoder()
yy = to_categorical(le.fit_transform(y))
```
Dall'intero dataset, partizioniamo gli elementi così da ottenere un *train set* e un *test set*:
```Python
x_train, x_test, y_train, y_test = train_test_split(X, yy, test_size=0.2, random_state=42)
```
Le strutture dati ottenute saranno:
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
Dal dataset di canzoni iniziale ho rimosso i campioni 3.mp3 e 4.mp3 e ho inserito tali campioni in un'apposita cartella test, separata dagli altri elementi del dataset. Ascoltando tali campioni, 3.mp3 mi trasmette tristezza, invece 4.mp3 mi trasmette felicità. 
Useremo tali due canzoni per testare se la predizione del modello coincide con l'emozione da me percepita all'ascolto di tali due canzoni. L'output a terminale è:
```
L'emozione predetta per 'test/4.mp3' è: happiness

L'emozione predetta per 'test/3.mp3' è: sadness
```
da cui possiamo concludere che, per quanto semplificato sia questo modello di riconoscimento delle emozioni delle canzoni, esso ha una buona capacità di categorizzazione.