# Thesis

Thesis work on the Classification of pre and post eruption seismic events relative to the Klyuchevskoy volcano eruption of April $21^{st}$, 2016.

This project uses the catalog from the [KISS experiment](https://geofon.gfz.de/doi/network/X9/2015) in Kamchatka, recorded by the X9 and D0 networks from July 2015 to August 2016, for this work we consider only the X9 network.

The dataset consists of 11,209 events recorded across 77 different stations at a sampling rate of 50 Hz.

## KISS Dataset Description

The study area is located around Klyuchevskoy Volcano, where the 77 stations are distributed as shown in Figure 1.

<img src="Images/Figure 1 (Real Map of Stations and Volcano).png" alt="Logo" width="70%" height="70%">

This figure shows the location of all stations, each marked as a blue triangle labeled with its respective name, while the volcano is marked as a red triangle.

For every event, the catalog provides detailed information including Latitude, Longitude, Depth, and Magnitude.

Figure 2 shows the distribution of events over time, marking the date of April $21^{st}$ with a vertical line.

<img src="Images/Figure 2 (Number of Events Over Time).png" alt="Logo" width="70%" height="90%">

Figure 3 shows the magnitude of every event.

<img src="Images/Figure 3 (Magnitude of all Events).png" alt="Logo" width="70%" height="90%">
We note that the maximum magnitude value is 3.1, while the mean magnitude is 1.

In order to train a model to classify events as Pre or Post eruption, we first need to label them correctly by separating them according to their date.

The catalog also includes a Phase Data section, indicating which station recorded which event.

Figure 4 shows the recording coverage for each station.

<img src="Images/Figure 4 (Data Availability Plot).png" alt="Logo" width="300" height="500">

In this figure, events registered by stations that also recorded Post eruption events are marked in blue, while those recorded by stations without Post eruption data are marked in orange. Only 41 of the 77 stations in the X9 network actually recorded post-eruption data, so we take data from those stations.

We apply this filtering because the model might otherwise learn that recordings from a particular station always correspond to events occurring before April $21^{st}$. In that case, rather than capturing meaningful seismic features, the model would simply recognize the station and classify the input as a Pre eruption event, basing its prediction solely on the fact that the station has no Post eruption data.

In Figure 5, we show the location of the selected stations.

<img src="Images/Figure 5 (Filtered Stations).png" alt="Logo" width="70%" height="70%">

Figure 5a shows all the stations, as already presented in Figure 1, while Figure 5b highlights in blue the stations of interest (those with post-eruption data) and in orange the stations that were discarded.

After this selection, we can introduce the models and describe the dataset in detail.

## Models

For the classification task, we compare two models, **CNN1D** and **CNN2D**, trained on the same dataset. The first model is trained on the raw waveforms, while the second is trained on their corresponding spectrograms.

The criterion used to train the models is a weighted cross-entropy loss with label smoothing, since, as we will see in the next section, the dataset is highly unbalanced. 

- The weighting compensates for this imbalance by assigning greater importance to the underrepresented Post eruption class
- Label Smoothing discourages the model from becoming overconfident in its predictions and improves generalization.

The optimizer is Adam, with an initial learning rate of $10^{-3}$ that decreases to a minimum of $8 \times 10^{-5}$ through a Reduce-on-Plateau scheduler.

The scheduler halves the learning rate whenever the validation accuracy begins to plateau, allowing the model to take smaller steps and refine its search for the best fit.

## Dataset

As shown in Figure 1, the dataset is highly unbalanced: it contains 10,729 Pre eruption events and 480 Post eruption events. 

In Figure 6, we show the number of events registered for each of the selected stations.

<img src="Images/Figure 6 (Filtered Stations Pre and Post Proportion).png" alt="Logo" width="70%" height="70%">

The blue bars represent the pre-eruption events, while the orange ones represent the post-eruption events. 
Note that the y-axis is on a logarithmic scale.

The complete dataset consists of $114{,}382$ samples from pre-eruption events and $4{,}649$ samples from post-eruption events. 

We define a sample as a 20-second-long waveform of a specific event retrieved from a specific station.
This 20-second window starts two seconds before the P-wave arrival and ends 18 seconds after it. 

The total number of samples is $119{,}031$, of which $96\%$ are Pre eruption ones and $4\%$ are Post eruption.

Because the dataset is so heavily unbalanced, for training we retrieve all the post-eruption samples but only $9{,}300$ pre-eruption samples. 

The pre-eruption portion of the dataset is not selected in a fully random: we draw Pre eruption events at random and keep adding them until the Pre eruption set reaches twice the size of the Post eruption one.

In practice, this means we train the model on roughly $870$ pre-eruption events and $480$ post-eruption events. (The exact number of Pre eruption events may vary, since events contributing few samples may be selected, but the sample count remains fixed at twice the number of post-eruption samples.)

Finally, we split this dataset into training, validation, and test sets following an 80/10/10 ratio.

It is important to note that the splits are not random either, they are built at the event level to avoid data leakage, meaning that all recordings of a given event are assigned to only one of the three sets (train, validation, or test).

Since we are training two different models, we also need to specify how the waveforms are provided to each one.

### 1D CNN Dataset

The CNN1D model is trained on the raw waveforms. As defined earlier, each sample consists of a 20-second recording sampled at 50 Hz, where the recording window for each event starts 2 seconds before the P-wave arrival and ends 18 seconds after it, so we have 1000 data for each sample.

Figure 7 shows an example of a waveform given to the model.

<img src="Images/Figure 7 (Waveform Example).png" alt="Logo" width="60%" height="70%">

### 2D CNN Dataset

The second model cannot process raw waveforms directly, so we first convert each waveform into its time frequency representation by computing a spectrogram.

The spectrogram is computed using a Short Time Fourier Transform (STFT). 

For each trace we apply the STFT with these features:

- A sampling frequency of 50 Hz
- A window length of 64 samples
- An overlap of 48 samples between consecutive windows. 

This configuration yields a hop size of 16 samples, providing a good compromise between time and frequency resolution.

We then take the magnitude of the resulting complex STFT coefficients and apply a logarithmic compression of the form $\log(1 + |Z|)$, which reduces the dynamic range and prevents high-amplitude components from dominating the representation.

Finally, each spectrogram is standardized to zero mean and unit variance, so that all inputs share a consistent scale before being fed to the network.

Figure 8 shows the spectrogram of the same waveform shown in Figure 7, computed as described above and given to the model.

<img src="Images/Figure 8 (Spectrogram Example).png" alt="Logo" width="60%" height="70%">

## Results

_Still work in Progress, this is a previous run with still good results on a balanced dataset, 50% pre samples and 50% post samples_
### CNN1D

Graphs for the Loss and Accuracy Values over training

<img src="Images/CNN1D_ Train and Validation set during Training_latest.png" alt="Logo" width="90%" height="100%">

Confusion Matrices for Train Validation and Test sets

<img src="Images/CNN1D_ Confusion Matrices_latest.png" alt="Logo" width="90%" height="100%">

The first model ended its training with a 82% validation accuracy and 81% test accuracy.
Since the model tends to overfit at the end we took the best fit at the $30^{th}$ epoch.
### CNN2D

Graphs for the Loss and Accuracy Values over training

<img src="Images/CNN2D_ Train and Validation set during Training_latest.png" alt="Logo" width="90%" height="100%">

Confusion Matrices for Train Validation and Test sets

<img src="Images/CNN2D_ Confusion Matrices_latest.png" alt="Logo" width="90%" height="100%">

The training over 100 epochs ends with a 91% accuracy on the validation set and 90% accuracy on the test set.
The model has a lower learning rate than the CNN1D one (CNN1D had a $10^{-3}$ while CNN2D had $10^
{-4}$ as learning rate)

