**Exercise 1 – Deep Learning (Practical part)**



This project implements the LeNet-5 neural network on the Fashion-MNIST dataset and compares the effects of:



1\. Baseline (no regularization)

2\. Dropout (at the hidden fully-connected layer)

3\. Weight Decay (L2 regularization)

4\. Batch Normalization



**Instructions**



To run all models in a single run, run the script:



Deep\_Learning\_Ex1.py



No changes need to be done in the code.

To run a single model only, change the configs variable in the function main() to include just one of the elements currently listed.

For example, to run only Dropout, change the configs variable to:



configs = \[{"name": "Dropout",     "use\_dropout": True,  "use\_batchnorm": False, "weight\_decay": 0.0}]





**Outputs**



The code generates three folders in the directory from which it runs:

1. Plots - contains the 8 convergence graphs requested in the assignment, two convergence graphs per model, one for train and one for test.
2. Plots combined - contains the 4 recommended combined plots, one per model, showing train and test accuracies on a single figure.
3. Checkpoints - contains the final trained model weights:

* Baseline.pt
* Dropout.pt
* WeightDecay.pt
* BatchNorm.pt

4\. runs - TensorBoard logging directory. To use TensorBoard type:



tensorboard --logdir runs



in terminal from the directory.



Additionally the code outputs a results.csv file, containing a table of the accuracies for train and test in each configuration.





**Load and test weights**

To load and test the saved weights, run the provided script:



Load\_weights.py



Test accuracies for each model will be printed.

