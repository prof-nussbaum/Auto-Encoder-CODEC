# Auto-Associative-CODEC
Auto-Associative How To

This repository includes "how to" examples of auto-associative neural networks. 
These networks have the ability to encode and decode (CODEC) sample data in the most efficient and low-loss way.
The minimization of the loss function is accomplished through the use of gradient descent training algorithms.

Example 1 - Auto-Associative How to (using TGS Salt Columns)

The first "how-to" example comes from a competition held on the kaggle.com website. 

WHAT IS THE OVERARCHING GOAL?
The competition, as well as details about the data set used for training, can be found here: https://www.kaggle.com/c/tgs-salt-identification-challenge.
The purpose of the competition is to see who best can create an AI system that classifies seismic data.
There are human experts who can do this classification from the original data, and indeed have done so to create the training and testing data.

WHY USE AUTO-ASSOCIATIVE NETWORKS?
In many cases, AI seeks to extract features from the original data set, and then use those features to train a classifier system that can perform as well as a human can.
The extraction of features is somewhat of an artform, that often uses application specific knowledge, also from human experts.
For many applications, human experts have NOT found good features to extract - and indeed must use the original data for classification.
In these cases, machine learning may be used to try and discern the salient features that still allow correct classification (by the machine).


NOTE: feature extraction may be lossy - in that the original data cannot be recreated exactly from the extracted features.
For this reason, an auto-associative network can be useful because it recreates a best-estimate of the original data from the extracted features.
In this way, a human expert can use the recreated data to tell if classification is still possible.
I call this a part of the exercise that I've chosen to give the, perhaps silly, name of "Reading the Robot Mind."

An example execution of the auto-associative Jupyter Notebook can be found here: https://www.kaggle.com/pnussbaum/auto-associative-how-to-using-tgs-salt-columns/execution.
In this example execution, seismic data is divided into columns of 101 points of sound reflection depth data.
The data is encoded down to 20 values (down from 101).
The decoding process is also shown, recreating the original data, but in a lossy way.
The purpose of this exercise is to see if "too much" data has been removed. 
If "too much" data has been removed, then an expert in seismic analysis will no longer be able to easily discern the classification from the decoded information.

ORIGINAL DATA IMAGE:
<img src="https://www.kaggleusercontent.com/kf/5296495/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..0t7XA6bCdyiksXNe5uymhw.O9UJCYEqUuKNeShHjawM9uwOwU2EbNgD_hofDeTVEtXwT3kLXw1lmVN3P94E2oS0kF0hfuO5szuGgFs2nyv3hIE4nZt8aBDWqsfW3tpfoyLeqOh90QXmy30KBNYNNoiy2DhriZeWuJFgVF_vdSybNEWHTqZF2wwcqbVMw10J6tUq3w3FJtEDgWag7IZjEmK_zWENo-NYjx6n9EGp0ZrK7GpsBjVlAukuB5KjZ77pTRVSP_ihEH33sVffrrsX_22OLjBf0fW27XvLkZUpgj3oxbUu4BewWm34gk7Cn4RiZI_2pzL3L0OjERRTZvaWtccE26ckob6uE4TFfY2xo3SkTL91xDucKMZT6AU22usSGlz_1_DkF7D8I739UBFhLYXZqWXuxhh58rVNfdxmA8SrrK0qhGx3rtF1qLyajOX_mFeFKqqi5nXbsNcNm13e_h4UewjKQpDhJ6axjbbMJ2DLVbJ1uon4dO4DpYtrut75SMJy0VZTAtzqHpjbAd02ccL4X7w2RuQH2c33GrimhAKYNi9hVh7YYzQ_HE-IylPnfbHMRHtFeLAxg2LluCc7BqcL1ooiT--TySaTfqEQVzxwYzzoUk5wMOoP8etXac0K9Eof_0_edjcav58de9RcSFp8km0BwzwTndRPz9TCe5kklh2qhftObF4_RVBDSoK0qfrchSqIk6iApxEfPc55ZV_L.kBqhesPBzl3gJfrcDMTINQ/__results___files/__results___13_1.png" alt="ORIGINAL DATA IMAGE">

Auto-Encoded (Coded and then Decoded) DATA IMAGE:
<img src="https://www.kaggleusercontent.com/kf/5296495/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..0t7XA6bCdyiksXNe5uymhw.O9UJCYEqUuKNeShHjawM9uwOwU2EbNgD_hofDeTVEtXwT3kLXw1lmVN3P94E2oS0kF0hfuO5szuGgFs2nyv3hIE4nZt8aBDWqsfW3tpfoyLeqOh90QXmy30KBNYNNoiy2DhriZeWuJFgVF_vdSybNEWHTqZF2wwcqbVMw10J6tUq3w3FJtEDgWag7IZjEmK_zWENo-NYjx6n9EGp0ZrK7GpsBjVlAukuB5KjZ77pTRVSP_ihEH33sVffrrsX_22OLjBf0fW27XvLkZUpgj3oxbUu4BewWm34gk7Cn4RiZI_2pzL3L0OjERRTZvaWtccE26ckob6uE4TFfY2xo3SkTL91xDucKMZT6AU22usSGlz_1_DkF7D8I739UBFhLYXZqWXuxhh58rVNfdxmA8SrrK0qhGx3rtF1qLyajOX_mFeFKqqi5nXbsNcNm13e_h4UewjKQpDhJ6axjbbMJ2DLVbJ1uon4dO4DpYtrut75SMJy0VZTAtzqHpjbAd02ccL4X7w2RuQH2c33GrimhAKYNi9hVh7YYzQ_HE-IylPnfbHMRHtFeLAxg2LluCc7BqcL1ooiT--TySaTfqEQVzxwYzzoUk5wMOoP8etXac0K9Eof_0_edjcav58de9RcSFp8km0BwzwTndRPz9TCe5kklh2qhftObF4_RVBDSoK0qfrchSqIk6iApxEfPc55ZV_L.kBqhesPBzl3gJfrcDMTINQ/__results___files/__results___13_3.png" alt="Auto-Encoded DATA IMAGE">
