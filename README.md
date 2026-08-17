# Thesis

Thesis work on the classification of pre- and post-eruption seismic events.

This project uses the catalog from the KISS experiment in Kamchatka, covering the period from July 2015 to August 2016 (https://geofon.gfz.de/doi/network/X9/2015). The dataset consists of 11,209 events recorded across 77 different stations.




## Event Distribution

The study area is located around Klyuchevskoy Volcano, where the 77 stations are distributed as shown in the following Figure 

![Area](Images/Stazioni_in_relazione_al_vulcano.png)


The following plot shows the distribution of events over the full time span:

![Distribution of events over the observed time span](Images/Number_of_Events_Over_Time.png)

And for each station we show in the next figure which Events it recorded:

<img src="Images/Stations_per_Event.png" alt="Logo" width="300" height="500">




## Waveforms

Each event is recorded by a number of stations. For each station, the catalog provides a waveform along with the arrival times of the P and S waves:

<img src="Images/event_waveform.png" alt="Logo" width="90%" height="100%">




## Classification

In order to compute a proper classification we have to distinguish events occured Before and After the Eruption, as we saw in the Events Distribution Figure we notice there
aren't as many Post-Eruption events as the Pre-Eruption ones.

The main eruption of Klyuchevskoy volcano occurred on the $21^{st}$ of April of 2016, since then 480 events occurred in the experiment's Area and were registred by 41 differents stazions

In the next Figure we show all the stations that recorded Post-Eruption Events in blue plus the ones we are using to train the model in red

![Distribution of events over the observed time span](Images/Stazioni_Evidenziate.png)


## Models

Tried two approaches to see how both models would work with the same dataset, firstly a 1D CNN analizing the waveforms, secondly a 2D CNN analizing the sepctrogram of those waves

### Dataset

The dataset we are using is formed by events gathered from March $20^{th}$ the end of the dataset which is on the end of July of the same year.
We have almost 3000 Elements, which include The N/E/Z components, gathered from 7 different Stations (SV13, SV6, SV7, IR2, IR3, IR4, IR6).

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
