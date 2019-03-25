# Exercises

## 1. How would you define Machine Learning?

Machine Learning - the science of programming computer to make them able to answer on questions with help from data.

Machine Learning is the field of study that gives computers the ability to learn without being explicitly programmed

A computer program is said to learn from experience E with respect to some task T and some performance measure P, if its performance on T, as measured by P, improves with experience E.

Machine Learning is about making machines get better at some task by learning from data, instead of having to explicitly code rules.
## Can you name four types of problems where it shines?

1. Problems for which existing solutions require a lot of hand-tuning or long lists of rules
2. Complex problem for which there is no good solution using a traditional approach
3. Fluctuation environment, where ML system can be adopted to new data
4. Getting an insight about complex problem and large amount of data

## What is a labeled training set?

Labeled training set is used in supervised learning where the algorithm is fed by data together with the desired solution, called labels. So, labeled training set is data and corresponding right answers.

## What are the two most common supervised tasks?
Classification - is to find to what class the data belongs to.   
Regression - is to predict a target numeric value.

## Can you name four common unsupervised tasks?

1. Clustering
2. Visualization and dimensionality reduction
3. Association rule learning
4. Feature extraction
5. Anomaly detection

## What type of Machine Learning algorithm would you use to allow a robot to walk in various unknown terrains?

Reinforcement learning

## What ype of algorithm would you use t segment your customers into multiple groups?

Clustering:
- K-means
- Hierarchical Cluster Analysis (HCA)
- Expectation Maximization

 
 ## Would you frame the problem of spam detection as a supervised learning problem or an unsupervised learning problem?
 
 It's an unsupervised learning problem, as you don't know exactly how to label spam neither you have enough human resources to label all emails manually
 
 ## What is an online learning system?
 
 An Machine Learning system which is updated incrementally by feeding a new data instances sequentially, either individually or by small groups called mini-batches.
 
 ## What type of learning algorithm relies on a similarity measure to make predictions?
 
 Instance-based learning algorithms
 
## What is the difference between a model parameter and a learning algorithm's hyperparameter?

Hyperparameters are constant during training, they affect how the model would be changed in order to generalize the supplied data.

## What do model-based learning algorithms search for?

A model which can describe the data in the best way.
 Then the found model can be used for prediction.
 
 ## Can you name four of the main challenges in Machine learning
 
 - Insufficient Quantity of Training Data
 - Nonrepresentative Training Data
 - Poor quality data
 - Irrelevant features
 
 ## If your model performs great on the training data but generalizes poorly to new instances, what is happening? Can you name three possible solutions?
 
 There might be happened an overfitting problem.
 The possible solutions are:
 - To simplify the model by selecting one with fewer parameters (e.g. a linear model VS a polynomial model), by reducing the number of attributes in the training data or by constraining the model.
 - To gather more training data
 - To reduct the noise in the training data (e.g. fix data errors and remove outliers) 
 
 ## What is a test set and why would you want to use it?
 
 A test set is a set of data used to estimate the generalization error. Usually, the whole data set is divided in 80/20 for the training data set and the testing data set.
 
 
 ## What is the purpose of a validation set?
 Validation set is a final set data the model's accuracy is estimated at.
 While training different models, they would succeed only for training/testing data set. So you feed an absolutely different data from the same domain, which the model doesn't see before.
 
 Validation set is the second holdout
 
 ## What can go wrong if you tune hyperparameters using the test set?
 The model was adapted the best model for the exact test set. It could fail in real production environment
 
 ## What is cross-validation and why would you prefer it to a validation set?
 
 Cross-validation is a technique to avoid generalization problem and to help to find the model's parameters produces both high accuracy and low error.
 
 The training set is split into complementay subsets, and each model is trained against a different combination of these subsets and validated against the remaining parts.
 
 
  