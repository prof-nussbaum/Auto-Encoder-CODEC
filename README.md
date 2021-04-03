# Auto-Encoder-CODEC
<h1>This is an Auto-Encoder or Auto-Associative How To</h1>

This repository includes "how to" examples of auto-associative neural networks. These are sometimes called Auto-Encoders.
These networks have the ability to encode and decode (CODEC) sample data in the most efficient and low-loss way.
The minimization of the loss function is accomplished through the use of gradient descent training algorithms.

<h2>Example 1 - Auto-Associative How to (using TGS Salt Columns)</h2>

The first "how-to" example comes from a competition held on the kaggle.com website. 

<h3>WHAT IS THE OVERARCHING GOAL?</h3>
The competition, as well as details about the data set used for training, can be found here: https://www.kaggle.com/c/tgs-salt-identification-challenge.
The purpose of the competition is to see who best can create an AI system that classifies seismic signals to identify underground salt deposits.
There are human experts who can do this classification from the original data, and indeed have done so to create the training and testing data.

<h3>WHY USE AUTO-ASSOCIATIVE NETWORKS? </h3>
In many cases, AI seeks to extract features from the original data set, and then use those features to train a classifier system that can perform as well as a human can. The extraction of features is somewhat of an artform, that often uses application specific knowledge, also from human experts. For many applications, human experts have NOT found good features to extract - and indeed must use the original data for classification. In these cases, machine learning may be used to try and discern the salient features that still allow correct classification (by the machine).

<h3>LOSSY FEATURE EXTRACTION</h3>
NOTE: feature extraction may be lossy - in that the original data cannot be recreated exactly from the extracted features. For this reason, an auto-associative network can be useful because it recreates a best-estimate of the original data from the extracted features. In this way, a human expert can use the recreated data to tell if classification is still possible. I call this a part of the exercise that I've chosen to give the, perhaps silly, name of "Reading the Robot Mind."

<h3>EXAMPLE CODE</h3>
An example execution of the auto-associative Jupyter Notebook can be found here: https://www.kaggle.com/pnussbaum/auto-associative-how-to-using-tgs-salt-columns/execution. In this example execution, seismic data is divided into columns of 101 points of sound reflection depth data. The data is encoded down to 20 values (down from 101). The decoding process is also shown, recreating the original data, but in a lossy way. The purpose of this exercise is to see if "too much" data has been removed. If "too much" data has been removed, then an expert in seismic analysis will no longer be able to easily discern the classification from the decoded information.

<h2>Example 2 - VSB Power using Auto-encoding</h2>

This second "how-to" example comes from a competition held on the kaggle.com website. 

<h3>WHAT IS THE OVERARCHING GOAL?</h3>
The competition, as well as details about the data set used for training, can be found here: https://www.kaggle.com/c/vsb-power-line-fault-detection.
The purpose of the competition is to see who best can create an AI system that classifies signals on medium voltage overhead power lines indicating a fault (fallen tree, etc.).
There are human experts who can do this classification from the original data, and indeed have done so to create the training and testing data.

<h3>WHY USE MULTIPLE AUTO-ENCODING NETWORKS?</h3>
This example uses multiple auto-encoding networks. 
Similar to wavelet analysis, an auto-encoding network is used to find a small set of features that can be used o most accurately recreate the original signal.
Once this auto-encoding network is trained, the recreated signals are subtracted from the original signal; creating a residual signal.
A second auto-encoding network is trained to find a small set of features that can be used to recreate the residual signal.
This can be repeated with a thrd, fourth, etc. auto-encoding network; with each network being trained on the Nth residual.

<h3>LOSSY COMPRESSION</h3>
Because the compression scheme is naturally lossy, we can add the residuals back up to discern if information needed by a human expert has been maintained.
This is given the, perhaps silly, name of "Reading the Robot Mind."
It allows a human expert, who is not a programmer, to see if sufficuent classification data has been retained by the trained machine learning algorithm.

<h3>DISCUSSION OF EXAMPLE 2 CODE</h3>
An example execution of the auto-associative Jupyter Notebook can be found here: https://www.kaggle.com/pnussbaum/vsb-power-using-autoencoding-v09.
In this example execution, voltage data is divided into three "phases" of 800,000 data points each. 
The three phases correspond to the three power lines comprising a three-phase power delivery system. 
The data has been hand-labeled (for training and testing) as having experienced a fault, or not.
During execution, the data is encoded to a much smaller feature set that can be used to recreate the original signal, as well as the residual (original minus recreated).
The following auto-encoder feature sizes are created:
First auto-encoder - creates a 5 data point feature vector from original data
Second auto-encoder - creates a 20 data point feature vector from prior residual data
Third auto-encoder - creates a 560 data point feature vector from prior residual data
Original data is passed through the three trained auto-encoders to create a 585 data point feature vector from the original 800,000 data points.
These 585 data points per example are used to train a classifier to detect if a fault has occurred on the power line

The decoding process is also shown, recreating the original data, but in a lossy way.
The purpose of this exercise is to see if "too much" data has been removed. 
If "too much" data has been removed, then an expert in power line analysis will no longer be able to easily discern the classification from the decoded information.
In theory, if a human expert cannot discern the classification, it may be diffficult for machine learning to do so.

<h3>UNINTENDED BENEFIT OF READING THE ROBOT MIND IN EXAMPLE 2 CODE</h3>
An unintended benefit of this technique is that residual data pointed out a potential flaw either in data collection or sensor measurements.
The first auto-encoder naturally found the main frequency of the powerline voltage, which was clearly shown in the recreated (lossy) signal.
The second auto-encoder surprisingly found a seventh harmonic of this signal as being pronounced in most of the data samples.
This potential flaw was reported to the manufacturer who promised to investigate further.

<h2>Example 3 - MNIST Digit ConvAutoencoder</h2>

This "how-to" example comes from a competition held on the kaggle.com website. 

<h3>WHAT IS THE OVERARCHING GOAL?</h3>
Recognize handwritten digits from the MNIST dataset. Do it with CNN generated "features" or "encoded" version of the input - the center of an autoencoder. 

The subject matter expert (SME) can recognize digits. They qualitatively rate the amount of data lost due to encoded compression. If the SME can no longer make a classification from a decoded input, they qualitatively assess that too much data has been removed.

Is this true? No. It is possible to train a CNN to classify with internal represntations that are distorted and simplified to the extent that they no longer can be used to recreate the input. I believe this is due to the fact that the SME not only classifies the examples, but so much more else.

