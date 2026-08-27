# Thesis

Thesis work on the classification of pre- and post-eruption seismic events.

This project uses the catalog from the [KISS experiment](https://geofon.gfz.de/doi/network/X9/2015) in Kamchatka recorded by the X9 network from July 2015 to August 2016. 
The dataset consists of 11,209 events recorded across 77 different stations recorded at a sampling rate of 50 Hz.


## Event Distribution

The study area is located around Klyuchevskoy Volcano, where the 77 stations are distributed as shown in the following Figure 

![Area](Images/Stazioni_in_relazione_al_vulcano.png)


The following plot shows the distribution of events over the full time span:

![Distribution of events over the observed time span](Images/Number_of_Events_Over_Time.png)

And for each station we show in the next figure which Events it recorded:

<img src="Images/Stations_per_Event.png" alt="Logo" width="300" height="500">



## Classification

In order to compute a proper classification we have to distinguish events occured Before and After the Eruption, as we saw in the Events Distribution Figure we notice there
aren't as many Post-Eruption events as the Pre-Eruption ones.

The main eruption of Klyuchevskoy volcano occurred on the $21^{st}$ of April of 2016, since then 480 events occurred in the experiment's Area and were registred by 41 differents stazions

In the next Figure we show all the stations that recorded Post-Eruption Events in blue plus the ones we are using to train the model in red

![Distribution of events over the observed time span](Images/Stazioni_Evidenziate.png)


## Models

For the classification task, we compare two models, **CNN1D** and **CNN2D**, trained on the same dataset. 
The first model is trained on the raw waveforms, while the second is trained on their corresponding spectrograms.

## Dataset

The dataset we use is highly imbalanced: it contains roughly **90,000 Pre-Eruption** samples and **4,600 Post-Eruption** samples. A sample refers to a specific event recorded by a specific station.ù
in total, we have **41 stations** (all stations that recorded post-eruption as shown in the **Event Distribution** section), which recorded a combined **8,610 distinct Pre-Eruption events** and **480 Post-Eruption events**.

To train the models, we take all Post-Eruption samples together with a random subset of Pre-Eruption samples, forming a balanced dataset. This dataset then is split into training, validation, and test sets following an 80/10/10 ratio.

It is important to note that the splits are built by the event to avoid data leakage: all recordings of a given event are assigned to only one of the three sets (train, validation, or test).

### Waveforms

The CNN1D model is trained on the raw waveforms. Each sample consists of a 20 second recording sampled at 50 Hz.
The recording window for each event starts 2 seconds before the P-wave arrival and ends 18 seconds after it. 

The figure below shows an example waveform with the P and S wave arrivals marked.

<img src="Images/event_waveform.png" alt="Logo" width="90%" height="100%">

## Spectrogram

_WIP_

### Hyper Parameters

We are using for both models the same characteristics, a training over 100 Epochs with a Binary Cross Entropy Loss.
The Optimizer we are using is Adam and we also implemented a Scheduler to stabilize the model's performance in the ending of the training.

### CNN1D

Graphs for the Loss and Accuracy Values over training

<img src="Images/CNN1D_ Train and Validation set during Training.png" alt="Logo" width="90%" height="100%">

Confusion Matrices for Train Validation and Test sets

<img src="Images/CNN1D_ Confusion Matrices.png" alt="Logo" width="90%" height="100%">

### CNN2D

Graphs for the Loss and Accuracy Values over training

<img src="Images/CNN2D_ Train and Validation set during Training.png" alt="Logo" width="90%" height="100%">

Confusion Matrices for Train Validation and Test sets

<img src="Images/CNN2D_ Confusion Matrices.png" alt="Logo" width="90%" height="100%">

The training over 100 epochs ends with a 70% accuracy on the test set.

