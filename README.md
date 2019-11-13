# bayes_by_backprop

## Abstract
Implement Bayes by Backdrop using numpy only.

## Description
Normal neural network often makes overly confident about their prediction, which may lead to overfitting. Bayes by Backprop add some uncertainties to the this network in order to reduce the strong confidence in it.

In the code, we show how Bayes by Backprop can be applied to Logistic Regression to learn the AND relationship between two bits. Since this code is implemented from scratch, all backpropagation steps are shown and clarified, which is easier to read and learn.

File `bbb.py` contains the code for Bayes by Backprop Logistic Regression, and file `nn.py` contains the code for normal neural networks. File `evaluate.py` provides some illustration about the comparison between two models. These includes the distribution of weights from which they are sampled, and the status of loss function after each iteration. Bash file `run.sh` is used to execute these three files to provide evaluation.

## Intruction
### Dependencies
* numpy
* matplotlib
* seaborn

### Usage
* Step 1: Install python 3.7 and given dependencies.
* Step 2: Open terminal and type `sh run.sh` (Linux)/ `zsh run.sh` (MacOS Catalina, if you use zsh).
* Step 3: See the results.