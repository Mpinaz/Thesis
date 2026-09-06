# Thesis

Thesis work on the Classification of pre and post eruption seismic events relative to the Klyuchevskoy volcano eruption of April $21^{st}$, 2016.

This project uses the catalog from the [KISS experiment](https://geofon.gfz.de/doi/network/X9/2015) in Kamchatka, recorded by the X9 and D0 networks from July 2015 to August 2016, for this work we consider only the X9 network.

The dataset consists of 11,209 events recorded across 77 different stations at a sampling rate of 50 Hz.

## KISS Dataset Description

The study area is located around Klyuchevskoy Volcano, where the 77 stations are distributed as shown in [Figure 1](https://github.com/Mpinaz/Thesis/blob/main/Images/Figure%201%20(Real%20Map%20of%20Stations%20and%20Volcano).png).

<img src="Images/Figure 1 (Real Map of Stations and Volcano).png" alt="Logo" width="70%" height="70%">

This figure shows the location of all stations, each marked as a blue triangle labeled with its respective name, while the volcano is marked as a red triangle.

For every event, the catalog provides detailed information including Latitude, Longitude, Depth, and Magnitude.

[Figure 2](https://github.com/Mpinaz/Thesis/blob/main/Images/Figure%202%20(Number%20of%20Events%20Over%20Time).png) shows the distribution of events over time, marking the date of April $21^{st}$ with a vertical line.

<img src="Images/Figure 2 (Number of Events Over Time).png" alt="Logo" width="70%" height="90%">

[Figure 3](https://github.com/Mpinaz/Thesis/blob/main/Images/Figure%203%20(Magnitude%20of%20all%20Events).png) shows the magnitude of every event.

<img src="Images/Figure 3 (Magnitude of all Events).png" alt="Logo" width="70%" height="90%">
We note that the maximum magnitude value is 3.1, while the mean magnitude is 1.

In order to train a model to classify events as Pre or Post eruption, we first need to label them correctly by separating them according to their date.

The catalog also includes a Phase Data section, indicating which station recorded which event.

[Figure 4](https://github.com/Mpinaz/Thesis/blob/main/Images/Figure%204%20(Data%20Availability%20Plot).png) shows the recording coverage for each station.

<img src="Images/Figure 4 (Data Availability Plot).png" alt="Logo" width="300" height="500">

In this figure, events registered by stations that also recorded Post eruption events are marked in blue, while those recorded by stations without Post eruption data are marked in orange. Only 41 of the 77 stations in the X9 network actually recorded post-eruption data, so we take data from those stations.

We apply this filtering because the model might otherwise learn that recordings from a particular station always correspond to events occurring before April $21^{st}$. In that case, rather than capturing meaningful seismic features, the model would simply recognize the station and classify the input as a Pre eruption event, basing its prediction solely on the fact that the station has no Post eruption data.

In [Figure 5](https://github.com/Mpinaz/Thesis/blob/main/Images/Figure%205%20(Filtered%20Stations).png), we show the location of the selected stations.

<img src="Images/Figure 5 (Filtered Stations).png" alt="Logo" width="70%" height="70%">

Figure 5a shows all the stations, as already presented in Figure 1, while Figure 5b highlights in blue the stations of interest (those with post-eruption data) and in orange the stations that were discarded.

After this selection, we can introduce the models and describe the dataset in detail.

## Models

For the classification task, we compare two models, **CNN1D** and **CNN2D**, trained on the same dataset for 100 Epochs. The first model is trained on the raw waveforms, while the second is trained on their corresponding spectrograms.

The criterion used to train the models is a weighted cross-entropy loss with label smoothing, since, as we will see in the next section, the dataset is highly unbalanced. 

- The weighting compensates for this imbalance by assigning greater importance to the underrepresented Post eruption class
- Label Smoothing discourages the model from becoming overconfident in its predictions and improves generalization.

The optimizer is Adam, with an initial learning rate of $10^{-3}$ that decreases to a minimum of $8 \times 10^{-5}$ through a Reduce-on-Plateau scheduler.

The scheduler halves the learning rate whenever the validation accuracy begins to plateau, allowing the model to take smaller steps and refine its search for the best fit.

## Dataset

As shown in [Figure 1](https://github.com/Mpinaz/Thesis/blob/main/Images/Figure%201%20(Real%20Map%20of%20Stations%20and%20Volcano).png), the dataset is highly unbalanced: it contains 10,729 Pre eruption events and 480 Post eruption events. 

In [Figure 6](https://github.com/Mpinaz/Thesis/blob/main/Images/Figure%206%20(Filtered%20Stations%20Pre%20and%20Post%20Proportion).png), we show the number of events registered for each of the selected stations.

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

[Figure 7](https://github.com/Mpinaz/Thesis/blob/main/Images/Figure%207%20(Train-Validation-Test%20set%20Splits).png) shows how the datas were split and how many datas are in each set

<img src="Images/Figure 7 (Train-Validation-Test set Splits).png" alt="Logo" width="50%" height="50%">
The total of train samples is 11118: 66.2% Pre eruption samples and 33.8% post samples; the total of validation samples is 1446: 69% Pre eruption samples and 31% post samples; The total of train samples is 1373: 68% Pre eruption samples and 32% post samples.

Since we are training two different models, we also need to specify how the waveforms are provided to each one.

### 1D CNN Dataset

The CNN1D model is trained on the raw waveforms. As defined earlier, each sample consists of a 20-second recording sampled at 50 Hz, where the recording window for each event starts 2 seconds before the P-wave arrival and ends 18 seconds after it, so we have 1000 data for each sample.

[Figure 8](https://github.com/Mpinaz/Thesis/blob/main/Images/Figure%208%20(Waveform%20Example).png) shows an example of a waveform given to the model.

<img src="Images/Figure 8 (Waveform Example).png" alt="Logo" width="60%" height="70%">

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

[Figure 9](https://github.com/Mpinaz/Thesis/blob/main/Images/Figure%209%20(Spectrogram%20Example).png) shows the spectrogram of the same waveform shown in Figure 8, computed as described above and given to the model.

<img src="Images/Figure 9 (Spectrogram Example).png" alt="Logo" width="60%" height="70%">

## Results

In order to understand how well the model is fitting our task, we will also use the Confusion Matrix, a 2×2 matrix (since our model can predict only two classes, pre and post eruption) where each cell contains the number of samples corresponding to a specific combination of true and predicted label.

The columns represent the actual class of the samples, while the rows represent the class predicted by the model. 

As a result, the two cells along the main diagonal count the correctly classified samples while false positives (Pre Eruption Events classified as Post ones) and False Negatives (Post Eruption Events classified as Pre ones) are located on the off-diagonal cells.

From this Matrix we can also extrapolate some useful metrics about the models' performances:

- Recall (Also known as sensitivity): It measures the proportion of Post eruption events that the model correctly identifies over the total of all Post Eruption samples:

 $$\frac{TP}{TP + FN}\\ \text{TP stands for True positives and FN for False Negatives}$$
 
---
- Precision: The proportion of Post eruption events that the model correctly identifies over the sum of them plus the wrongly calssified post eruption samples:

$$\frac{TP}{TP + FP}\\ \text{FP stands for False positives}$$

---
- F1 Score: The harmonic mean of precision and recall. The F1 score penalizes models that achieve a high score on one at the expense of the other. It is particularly informative for the minority class (the post eruption one), where both false positives and false negatives matter:

$$F_1 = 2 \times \frac{Precision \times Recall}{Precision \times Recall}$$

---
- Balanced Accuracy: The average of the recall computed independently for each class. Unlike standard accuracy, it treats both classes as equally important regardless of how many samples each one contains. This makes it a much more reliable indicator of performance on our imbalanced dataset rather than a simple accuracy matric, since it rewards a model only when it classifies both pre and post-eruption events well:

$$B.acc = \frac 12 (\frac{TP}{TP + FN} + \frac{TN}{TN + FP})$$


### 1DCNN

After completing the full training, we compute the evaluation metrics on the test set, a collection of samples the model has never seen (samples belonging to events that are present in neither the training nor the validation set, as previously discussed in the dataset splitting section).

The training concludes with a **balanced accuracy** of $85.8%$

To provide further insight into the model's fit, we report the previously discussed metrics:


|              |   precision |   recall |   f1-score |   support |
|:-------------|------------:|---------:|-----------:|----------:|
| pre          |       0.928 |    0.856 |      0.891 |       931 |
| post         |       0.739 |    0.86  |      0.795 |       442 |
| weighted avg |       0.867 |    0.857 |      0.86  |      1373 |


where support stands for the number of samples in each category.

[Figure 10](https://github.com/Mpinaz/Thesis/blob/main/Images/Figure%2010%20(1DCNN%20Confusion%20Matrices).png) shows the confusion matrix normalized over the total number of samples in each category.

<img src="Images/Figure 10 (1DCNN Confusion Matrices).png" alt="Logo" width="80%" height="70%">

he model overfits considerably on the training set, the high percentages of true positives and true negatives observed are achieved mainly because the model memorizes the training samples and their labels rather than learning generalizable features.

This overfitting can be observed more clearly in. [Figure 11](https://github.com/Mpinaz/Thesis/blob/main/Images/Figure%2011%20(1DCNN%20Loss%20and%20Accuracy%20over%20epochs).png).

<img src="Images/Figure 11 (1DCNN Loss and Accuracy over epochs).png" alt="Logo" width="60%" height="50%">
The first graph shows the values of the cross entropy loss over the epochs: after the $20^{th}$ epoch the training loss becomes significantly lower than the validation loss indicating that this is the point at which the model stops learning meaningful features and begins memorizing the training set.
The second graph shows the accuracy over the epochs for the training and validation sets, here we can see that, despite the validation loss no longer decreasing much after the $20^{th}$ epoch, the validation accuracy still increases by small margins.


Lastly [Figure 12](https://github.com/Mpinaz/Thesis/blob/main/Images/Figure%2012%20(1DCNN%20Confidence%20for%20wrong%20classifications).png) shows every misclassified test event plotted on a time axis.
<img src="Images/Figure 12 (1DCNN Confidence for wrong classifications).png" alt="Logo" width="60%" height="50%">
False positives are shown in red and false negatives in blue. Since the model does not necessarily misclassify every sample of a given event, the y axis reports the percentage of misclassified samples out of the total number of samples for that event.
An event appears in this graph if at least one of its samples was misclassified.

For more detailed information about which events were misclassified, how many samples each event contains, how many of them were misclassified, and the corresponding percentage, please refer to this [Table](https://github.com/Mpinaz/Thesis/blob/main/Tables/Table%201%20(1DCNN%20wrong%20classifications).csv)

### 2DCNN

The same work for the 1DCNN model is computed for the 2DCNN model, with the same samples and events used as test set.

We obtain a signficicantly higher **Balanced Accuracy** of $93.5%$, also the evaluation metrics show the better fit

|              |   precision |   recall |   f1-score |   support |
|:-------------|------------:|---------:|-----------:|----------:|
| pre          |       0.964 |    0.944 |      0.954 |       931 |
| post         |       0.887 |    0.925 |      0.906 |       442 |
| weighted avg |       0.939 |    0.938 |      0.938 |      1373 |

[Figure 13](https://github.com/Mpinaz/Thesis/blob/main/Images/Figure%2013%20(2DCNN%20Confusion%20Matrices).png) shows the normalized confusion matrix for each se.

<img src="Images/Figure 13 (2DCNN Confusion Matrices).png" alt="Logo" width="80%" height="70%">

Unfortunately the models memorizes all the training set, giving the illusion of a perfect fit while it still makes mistakes on validation and test sets, more informations
in [Figure 14](https://github.com/Mpinaz/Thesis/blob/main/Images/Figure%2014%20(2DCNN%20Loss%20and%20Accuracy%20over%20epochs).png) 

<img src="Images/Figure 14 (2DCNN Loss and Accuracy over epochs).png" alt="Logo" width="60%" height="50%">
We notice that the pattern as the 1D model but this time we reach better results.

Finally we show the missclassifications for this model in [Figure 15](https://github.com/Mpinaz/Thesis/blob/main/Images/Figure%2015%20(2DCNN%20Confidence%20for%20wrong%20classifications).png) 

<img src="Images/Figure 15 (2DCNN Confidence for wrong classifications).png" alt="Logo" width="60%" height="50%">

More informations on this graph in this [Table](https://github.com/Mpinaz/Thesis/blob/main/Tables/Table%202%20(2DCNN%20wrong%20classifications).csv)
