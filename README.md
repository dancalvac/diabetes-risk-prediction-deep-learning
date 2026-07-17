
# Diabetes Prediction using a Deep Neural Network (MLP)

This project trains a Multi-Layer Perceptron (MLP) to predict whether a person likely has diabetes, based on health and lifestyle survey responses from the CDC's Behavioral Risk Factor Surveillance System (BRFSS) 2015 dataset.

## Overview

- **Task:** Binary classification — Diabetes (1) vs. No Diabetes (0)
- **Model:** Feedforward deep neural network (MLP) built with TensorFlow/Keras
- **Dataset:** BRFSS 2015, pre-balanced 50/50 between the two classes
  - https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset?select=diabetes_binary_5050split_health_indicators_BRFSS2015.csv
- **Dataset size:** 70,692 rows × 22 columns (21 features + 1 target)

## Dataset

The dataset (`diabetes_dataset.csv`) contains 21 features derived from BRFSS survey responses, covering:

- **Medical history:** `HighBP`, `HighChol`, `CholCheck`, `Stroke`, `HeartDiseaseorAttack`
- **Lifestyle:** `Smoker`, `PhysActivity`, `Fruits`, `Veggies`, `HvyAlcoholConsump`
- **Body/general health:** `BMI`, `GenHlth`, `MentHlth`, `PhysHlth`, `DiffWalk`
- **Healthcare access:** `AnyHealthcare`, `NoDocbcCost`
- **Demographics:** `Sex`, `Age`, `Education`, `Income`

The target column is `Diabetes_binary` (0 = No Diabetes, 1 = Diabetes). The dataset is already balanced 50/50 between classes, so no additional resampling was needed.

**Data split:** 70% train / 15% validation / 15% test, stratified to preserve the class balance in each split (49,512 / 10,576 / 10,604 samples respectively).

## Pipeline

1. **Exploratory Data Analysis (EDA)** — class distribution, correlation heatmap, and boxplots of the top correlated features against the target.
2. **Preprocessing** — features and target separated, then stratified train/val/test split, followed by `StandardScaler` feature scaling (fit on train, applied to val/test) since neural networks are sensitive to feature magnitude.
3. **Model architecture** — a Keras `Sequential` MLP:
   - Hidden layers: **128 → 64 → 32** neurons
   - Each hidden layer: Dense (ReLU + L2 regularization) → BatchNormalization → Dropout
   - Dropout rate: 0.3, L2 regularization: 0.01
   - Output layer: single neuron with sigmoid activation (binary probability)
   - Optimizer: Adam, learning rate 0.001
   - Loss: binary cross-entropy
   - Tracked metrics: accuracy, AUC, precision, recall
4. **Training** — up to 100 epochs, batch size 256, with:
   - `EarlyStopping` (patience 15, restores best weights on validation loss)
   - `ReduceLROnPlateau` (halves the learning rate if validation loss plateaus)
   - `ModelCheckpoint` (saves the best model by validation AUC to `best_diabetes_model.h5`)
   - Training stopped early at **83 epochs**.
5. **Evaluation** — accuracy, precision, recall, F1, AUC-ROC, confusion matrix, and ROC curve on the held-out test set.
6. **Interpretability** — feature importance computed two ways:
   - A simple gradient-based method (mean absolute gradient of the output w.r.t. each input feature)
   - SHAP (`GradientExplainer`) for a more rigorous, game-theoretic attribution, including a SHAP summary plot showing both magnitude and direction of each feature's effect.

## Results

Evaluated on the held-out test set (10,604 samples):

| Metric | Score |
|---|---|
| Accuracy | 0.754 |
| Precision | 0.730 |
| Recall (Sensitivity) | 0.806 |
| F1-Score | 0.766 |
| AUC-ROC | 0.832 |
| Specificity | 0.703 |

Confusion matrix breakdown: 3,725 true negatives, 1,577 false positives, 1,028 false negatives, 4,274 true positives.

The model favors **recall over precision** — it catches ~81% of true diabetes cases, at the cost of somewhat more false positives. For a screening-style health tool, that tradeoff is generally preferable to missing at-risk individuals.

### Most influential features (via SHAP)

1. `GenHlth` (self-reported general health)
2. `BMI`
3. `Age`
4. `HighBP`
5. `HighChol`
6. `Income`
7. `Sex`
8. `HeartDiseaseorAttack`
9. `Education`
10. `DiffWalk`

Self-reported general health, BMI, and age carried the most weight in the model's predictions, which lines up with established clinical risk factors for type 2 diabetes.

## Requirements

```
numpy
pandas
matplotlib
seaborn
scikit-learn
tensorflow
shap
```

## How to run

1. Place `diabetes_dataset.csv` in the working directory.
2. Run the notebook top to bottom (originally developed in Google Colab).
3. Outputs produced:
   - `best_diabetes_model.h5` — the trained Keras model (best checkpoint by validation AUC)
   - `model_architecture.png` — visualized network diagram
   - `model_results_summary.csv` — final test-set metrics

## Notes

- Random seeds (`42`) are fixed for NumPy and TensorFlow for reproducibility.
- The dataset is self-reported survey data, so it reflects the accuracy limitations inherent to self-reported health responses; this model is a demonstration of the ML pipeline rather than a validated diagnostic tool.
