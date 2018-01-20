# HiggsBosonML
Final project for CS 7140 / EECE 7397

The purpose of this project is to explore the effectiveness of different multi-layer perceptron architectures in a subatomic particle classifcation task.  A report of the results is available [here](https://github.com/jgartman/HiggsBosonML/blob/master/final_paper/final_paper.pdf), a summary is given below.

## Summary

### Introduction
Recently CERN, the organization that runs the Large Hadron Collider, released a dataset of approximately 800,000 simulated particle collisions as part of a competition on the popular data science website Kaggle.  The purpose of the competition was to search for decays of the Higgs boson into two Tau leptons.  The dataset contains a mix of low level features such as particle velocity and trajectory and higher level features derived through kinematic or other physical relations from the low level data. In related work, [Baldi et. al.](https://arxiv.org/pdf/1410.3469.pdf) compare the results of deep and shallow neural networks for an identical particle classification task but using a dataset approximately 50x larger than that released by CERN.  Their results show that a deep neural network trained with low level features can outperform a shallow network trained with a combination of high and low level features and match the performance of a deep network trained with both high and low level features.  This is a suprising result since the high level features would seem to have greater discriminating power between the classes than the low level features as the following plots show:

![plot](https://github.com/jgartman/HiggsBosonML/blob/master/plots/DER_mass_transverse_met_lep.png)

![plot](https://github.com/jgartman/HiggsBosonML/blob/master/plots/PRI_met.png)

The plot on the left shows a typical histogram for a high level feature (in this case the transverse mass between the missing transverse energy and the lepton) and the plot on the right shows a typical histogram for a low level feature (in this case the missing transverse energy).  In both plots the blue is signal decay and green is background.  A further explanation of the meaning of these features can be found [here](http://proceedings.mlr.press/v42/cowa14.pdf)

### Project Details
This project attempts to replicate results of Baldi et. al. but using the smaller competition training data set and a more regularized network architecture.  One of the most computationally expensive aspects of neural network training is the hyperparameter optimization stage.  The network must be trained and evaluated for different settings of the network hyperparameters such as depth and neurons per layer to find the most effective configuration.  One method of limiting this computational cost is to use a Bayesian hyperparameter optimization.  In a Bayesian hyperparameter optimiziation the hyperparameter search space is focused on values most likely to produce the most improvement in the models performance.  This project uses the [Spearmint software package](https://github.com/JasperSnoek/spearmint) to perform a hyperparameter optimization of the search space.

### Conclusion
The most important conclusions of this project are two-fold.  Firstly, the results of Baldi et. al. are not replicated with the smaller training set.  A deep network trained with low level features could not match the performance of the deep network trained with both high and low level features.  Also, again in contrast to Baldi et. al., a shallow network trained with the full feature set was able to match the performance of a deep network trained on the full feature set. Secondly, the network chosen by the hyperparameter optimization was much smaller than that of Baldi et. al.  Their network was 8 layers of 500 neurons each.  The Bayesian hyperparameter optimiztion described previously chose a much smaller network with 150 neurons in only 2 hidden layers.

![plot](https://github.com/jgartman/HiggsBosonML/blob/master/plots/accuracy.png)

Link to CERN data : http://opendata.cern.ch/record/328?ln=en

