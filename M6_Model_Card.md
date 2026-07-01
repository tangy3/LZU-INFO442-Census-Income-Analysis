# Model Card: Adult Income Prediction Model

## Model Overview

| Item | Description |
|---|---|
| Model name | Adult Income Prediction Model |
| Final selected model | Random Forest Classifier |
| Task type | Binary classification |
| Prediction target | Whether an individual's annual income is `>50K` or `<=50K` |
| Positive class | `>50K` |
| Dataset | Cleaned UCI Adult Income dataset |
| Deployment | Streamlit web application |
| Default decision threshold | `0.50` |

This model predicts whether an individual's annual income is greater than `$50K` based on demographic, education, household, and employment-related features. The final deployed model is a Random Forest classifier selected because it provides a strong F1-score balance, handles non-linear relationships, and performs well on the imbalanced Adult Income classification task.

## Intended Use

This model is intended for **educational analysis and machine learning demonstration**. It is designed to support a course project showing how a cleaned dataset can be transformed into a modelling pipeline, evaluated with multiple metrics, interpreted, and deployed through an interactive Streamlit web app.

Appropriate uses include:

- demonstrating binary classification with tabular census-style data;
- exploring how education, occupation, working hours, and demographic variables relate to income categories;
- comparing model outputs and probability thresholds;
- teaching model evaluation, local explanation, what-if analysis, and responsible interpretation.

## Out-of-Scope and Prohibited Uses

This model should **not** be used for real-world decisions about individuals. It should not be used for:

- hiring, promotion, or employment screening;
- salary, compensation, or workplace evaluation decisions;
- credit, loan, insurance, or financial eligibility decisions;
- admissions, benefit allocation, or legal decisions;
- any automated decision-making process that affects a person's opportunities or rights.

The model output should be interpreted as an educational probability estimate, not as a judgment of personal ability, value, or future potential.

## Data

The model uses the cleaned Adult Income dataset stored as `adult_cleaned.xlsx`.

| Data property | Value |
|---|---:|
| Number of records | 32,534 |
| Number of predictor features | 12 |
| Target classes | `<=50K`, `>50K` |
| `<=50K` records | 24,696 |
| `>50K` records | 7,838 |
| `>50K` class rate | 24.09% |

The dataset is moderately imbalanced because the `>50K` class is the minority class.

## Input Features

| Feature group | Features |
|---|---|
| Numerical | `age`, `hours_per_week` |
| Binary flags | `capital_gain_flag`, `capital_loss_flag` |
| Categorical | `workclass`, `education`, `marital_status`, `occupation`, `relationship`, `race`, `sex`, `native_country` |

The preprocessing pipeline standardizes numerical features, passes binary flag features directly, and applies one-hot encoding to categorical variables. The same preprocessing logic is used during model training and prediction in the Streamlit app.

## Output

The model produces:

- predicted class: `<=50K` or `>50K`;
- predicted probability: estimated probability that income is `>50K`;
- threshold-based decision: default threshold is `0.50`;
- confidence label in the app based on the distance between predicted probability and threshold.

## Model Details

The final model is a Random Forest classifier inside a scikit-learn pipeline.

| Hyperparameter | Value |
|---|---|
| `n_estimators` | `300` |
| `max_depth` | `None` |
| `min_samples_split` | `10` |
| `min_samples_leaf` | `1` |
| `max_features` | `sqrt` |
| `class_weight` | `balanced` |
| `random_state` | `42` |
| `n_jobs` | `-1` |

The model was evaluated using an 80/20 train-test split with stratification on the target variable. This keeps the class distribution similar in the training and test sets.

## Performance

Performance below is measured on the independent test set at threshold `0.50`.

| Metric | Score |
|---|---:|
| Accuracy | 0.833 |
| Precision | 0.628 |
| Recall | 0.751 |
| F1-score | 0.684 |
| ROC-AUC | 0.898 |
| PR-AUC | 0.735 |

Confusion matrix at threshold `0.50`:

| Actual / Predicted | Predicted `<=50K` | Predicted `>50K` |
|---|---:|---:|
| Actual `<=50K` | 4,241 | 698 |
| Actual `>50K` | 391 | 1,177 |

The model has strong ranking ability, as shown by ROC-AUC `0.898`, and a reasonable balance between Precision and Recall. Recall is higher than Precision, meaning the model is relatively effective at identifying actual `>50K` cases, but it also creates some false positives.

## Ethical Considerations

This model uses sensitive or socially meaningful attributes, including `sex`, `race`, and `native_country`. These features may reflect historical inequalities in employment, education, and income. Because the model learns from historical data, it may reproduce existing social patterns rather than provide a fair individual assessment.

Important ethical risks include:

- subgroup performance may differ by sex, race, education, occupation, or native country;
- predictions may be misinterpreted as causal explanations;
- model outputs could be misused in high-impact decisions;
- false positives and false negatives may affect groups differently;
- historical census data may not represent current labour market conditions.

For these reasons, the model should be used only as an academic demonstration and should always be accompanied by clear limitations.

## Limitations

The model has several important limitations:

- The dataset is based on historical United States census data and may not reflect modern income patterns.
- The task is imbalanced because only about 24.09% of records belong to the `>50K` class.
- The model predicts statistical association, not causation.
- The model may perform differently across demographic, education, and occupation subgroups.
- The Random Forest model is less directly interpretable than simpler linear models.
- Predictions depend on the selected classification threshold.
- The Streamlit app is a demonstration deployment, not a production-ready decision system.

## Responsible Use Recommendations

Users should:

- treat predictions as uncertain estimates, not facts;
- report probability and threshold together with the predicted class;
- review Precision, Recall, F1-score, ROC-AUC, and PR-AUC rather than relying only on Accuracy;
- monitor performance across subgroups before any applied use;
- avoid using the model for real individual-level decisions;
- retrain and revalidate the model with modern, representative data if adapting it beyond the course project.

## Deployment Notes

The model is deployed in a Streamlit application that supports:

- dataset exploration;
- individual prediction;
- batch CSV prediction;
- model comparison;
- threshold control;
- local prediction explanation;
- what-if analysis;
- model performance diagnostics.

The deployment is intended to support the final project demo and recorded walkthrough. It should be presented as an educational analytics tool.
