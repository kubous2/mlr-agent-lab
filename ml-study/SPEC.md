Goal: one sentence about what the result answers.

Context: the file, its columns and row count, the source, the class counts and the duplicates you found.

Constraints:

Duplicates: remove them before splitting, or keep them but make identical rows land on the same side of every split (group-aware splitting). Pick one and say why.
Test set: for example 20 %, stratified by class, a fixed random seed, split once at the start and used once at the end.
Models: for example a majority-class baseline, k-NN, logistic regression, decision tree, SVM (RBF) and random forest.
Hyperparameters: for example k-NN with k from 1 to 30, SVM with C and gamma on a log scale, tree depth from 1 to 10.
Tuning: for example 5-fold stratified cross-validation on the training part only, with scaling inside a Pipeline, and a budget such as a grid search or up to 50 trials per model.
Metric: accuracy, or balanced accuracy / macro F1 if the classes are unbalanced. The model is chosen by cross-validated score, never by the test score.