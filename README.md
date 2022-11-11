# Requirements

Required Python packages can be found in `requirements.txt`

# Running the Code

To inspect the dataset (with details and plots saved to `data_inspection_output/`:

	py inspect_data.py

To train and evaluate the MLP and Random Forest models (using 70% of the original `dataset.csv`):

	py train_and_tune_models.py

To perform the (very long) cross validation process on the MLP model (random search and grid search) to tune the hyperparameters:

	py train_and_tune_models.py --cv-mlp

To perform the (very long) cross validation process on the Random Forest model (random search and grid search) to tune the hyperparameters:

	py train_and_tune_models.py --cv-rf

To load the saved models and evaluate their performance on the holdout test dataset (30% of the original `dataset.csv`):

	py load_and_test_models.py

# Model Evaluation Results

Cross validation results are stored in: `cross_validation_results/`

Model evaluations from the training process are stored in: `saved_model_evaluations/` (output of `train_and_tune_models.py`)

Evaluations of the pre-trained models are stored in: `loaded_model_evaluations/` (output of `load_and_test_models.py`).
- **Note:** the scores are slightly lower in `loaded_model_evaluations/`, because only 30% of `dataset.csv` is tested (the "holdout test dataset" described above). This means the test dataset for the loaded model is rather small. However, this was done to ensure *the saved model is robust against overfitting.*

# Additional Implementation Notes

To avoid duplicating code, functions that are used in both `train_and_tune_models.py` and `load_and_test_models.py` are located in `shared_functions.py`

# Resources

## Dataset Description

	https://www.kaggle.com/datasets/lucidlenn/sloan-digital-sky-survey

## MLP Neural Network

	https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
	https://www.educative.io/answers/implement-neural-network-for-classification-using-scikit-learn
	https://panjeh.medium.com/scikit-learn-hyperparameter-optimization-for-mlpclassifier-4d670413042b

## Random Forest

	https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
	https://stackabuse.com/random-forest-algorithm-with-python-and-scikit-learn/
	https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74

##  Feature Scaling

	https://towardsdatascience.com/what-and-why-behind-fit-transform-vs-transform-in-scikit-learn-78f915cf96fe

## Overfitting

	"We can identify if a machine learning model has overfit
	by first evaluating the model on the training dataset
	and then evaluating the same model on a holdout test dataset."

	https://machinelearningmastery.com/overfitting-machine-learning-models/