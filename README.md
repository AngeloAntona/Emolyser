# Introduction
During my Artificial Intelligence course at AMU University (Poznan) and my Creative Programming course at Politecnico di Milano, I delved into how AI and intelligent systems can solve various logical problems. Once I tried to apply these approaches to music, though, I realized it’s not a purely “logical” field. Despite music having a solid mathematical structure, it conveys subtle emotions that are challenging to capture computationally. This insight sparked my curiosity about how a small neural network, trained on my MacBook Air M1 with 8 GB of RAM, might learn and replicate these emotions in musical compositions. 

After covering the theoretical concepts in class, I was eager to see if my limited resources could handle such a nuanced task. While my experiments in **classification** turned out relatively well—around 70% accuracy, the results in **generation** left a lot to be desired. I suspect this gap might be due to the modest dataset I used, or perhaps my own inexperience in fine-tuning hyperparameters and network architectures. It could also be that my setup simply wasn’t powerful enough to train complex generative models to the point where they sound convincingly musical and emotionally rich.

# System Configuration
To run the scripts I’ll describe below, you’ll need to install several libraries. I used the following command:
```
pip install pandas librosa numpy joblib tqdm scikit-learn tensorflow music21
```
I worked with two main datasets:
* **DEAM** for the first script (emotion classification).
* **EMOPIA** for the second script (melody generation).

(You can find further details about these datasets [here](readme_files/dataset_info.md).)

# CODE1: Song Emotion Analysis
For my first practical test, I built a system to classify musical emotions (**happiness**, **sadness**, **calm**, and **anger**) based on extracted audio features.

![Code1 scheme](readme_files/code1_scheme.png)

I relied on the DEAM annotation set to label my data. In the preprocessing phase, I extracted a wide range of features (MFCC, Mel-spectrogram, Chroma, Spectral Contrast, Tonnetz, Tempo, Zero Crossing Rate, Spectral Centroid, Spectral Rolloff, and RMS Energy). To address class imbalance, I applied a data augmentation strategy that generated new samples for underrepresented categories.

## Model Architecture
I chose a **hybrid CNN-LSTM** network for this task:

1. A **TimeDistributed CNN** for extracting local features from each frame of audio.  
2. **MaxPooling** and **Dropout** layers for regularization.  
3. An **LSTM** to capture temporal relationships across frames.  
4. A **Dense (Softmax)** layer to classify the output into four emotion classes.

Despite being my first hands-on experiment, the classifier reached about **70% accuracy** on the test set. It correctly predicted the emotional category for most tracks, confirming that even on my modest hardware, some aspect of musical emotion can be captured effectively.

# CODE2: Song Emotion Generation
Encouraged by the classification results, I decided to push my luck and explore **generating** melodies that reflect specific emotional tones. Here, the question was whether a small model could learn not just to *recognize* emotional markers in music but also to *create* them.

![Code1 scheme](readme_files/code2_scheme.png)

### Overall Approach
I gathered MIDI files corresponding to four emotions: **happiness**, **sadness**, **calm**, and **anger**. I then merged them into two broader groups: **positive_emotion** (happiness + calm) and **negative_emotion** (anger + sadness). From each MIDI file, I extracted note sequences and chords, turning them into numerical arrays suitable for my neural network.

### Data Augmentation
To enlarge my training data, I used **transposition**, shifting sequences by a few semitones up or down. Although this trick helped increase variety, I still felt the dataset was too small to provide truly convincing results. That limitation might have been a core reason why generation never felt quite *musically complete*.

### Model Architecture
I tested multiple approaches, but my final setup used a **GRU-based network** with two stacked GRU layers and a **custom Attention** mechanism. By embedding notes and emotions in separate spaces, the model could potentially learn not only melodic progressions but also the subtle emotional transitions. The attention layer was supposed to help the system “focus” on significant parts of the input sequence.

### Training & Model Saving
I trained these networks on my MacBook Air M1, often leaving them running for hours. To avoid overfitting, I employed **ModelCheckpoint**, **EarlyStopping**, and **ReduceLROnPlateau**. Each time I tested an architecture tweak—tuning hyperparameters or adjusting the sequence length—the results were mixed. I did manage to save intermediate models, so I didn’t lose progress whenever I found a slight improvement.

### Melody Generation
Once trained, my model attempted to produce new sequences that matched the target emotion—either positive or negative. I used **nucleus sampling** to introduce randomness without letting the results descend into pure noise. Even though the generated melodies often sounded clumsy, it was interesting to note that they tended to employ more **major scales and chords** under the positive emotion label, and more **minor scales and chords** under the negative one. This suggests the model did learn *something* about emotional musical patterns, even if the overall quality of the melodies remains quite poor.

# Conclusions
Although the classification task worked decently, the melody generation results were less satisfying. I suspect that deeper architectures or a larger dataset could yield better musical pieces. Nevertheless, it was an enlightening journey into how an AI might grasp and replicate human emotions in music—proof that, despite the underwhelming outcomes, the model did manage to capture at least some rudimentary aspects of musical affect. 