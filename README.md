# Eco-RETINA

Eco-RETINA is an innovative, eco-conscious algorithm designed for out-of-sample prediction. It acts as a flexible, regression-based approximator: linear in parameters but nonlinear in inputs. The algorithm leverages a selective model search to optimize performance while maintaining interpretability. 
Eco-RETINA:
- Automatically generates polynomial and interaction terms
- Selects feature subsets using a grid-based R²-driven search
- Supports linear, logit, and probit regression via statsmodels
- Computes prediction and confidence intervals
- Tracks carbon emissions with CodeCarbon

Required Packages Installation
------------
Run the following:

    pip install -r requirements.txt

Usage Example
-------------
 To see how to use EcoRETINA in practice, check out the interactive Jupyter notebook:

▶️ **[example.ipynb](example.ipynb)**

The notebook walks through:
- Loading and preprocessing data
- Fitting the EcoRETINA model
- Generating predictions and intervals
- Evaluating model performance   


fit() Parameters
----------
| Parameter            | Type               | Default            | Description |
|----------------------|--------------------|--------------------|-------------|
| `y`                  | NDArray            | —                  | Target variable array (1D). |
| `X`                  | NDArray            | —                  | Feature matrix. |
| `con_cols_indices`   | List[int]          | —                  | Indices of continuous  columns in `X`. |
| `dummy_cols_indices` | List[int]          | —                  | Indices of dummy (categorical one-hot) columns in `X`. |
| `col_names`          | List[str] or None  | —                  | Names of original input features (used for labeling). |
| `params`             | List[float]        | `[-1.0, 0.0, 1.0]` | Exponents to apply for generating transformed features. |
| `cross_dummy`        | bool               | `False`            | Allow dummy-dummy feature interactions. |
| `max_r2`             | float              | `0.9`              | Maximum R² allowed for selecting a new feature. |
| `grid`               | float              | `0.005`            | R² step used during feature selection. |
| `reg_type`           | str                | `'linear'`         | Type of regression: `'linear'`, `'logit'`, or `'probit'`. |
| `loss`               | str                | `'mse'`            | Metric to optimize: `'mse'`, `'mae'`, `'mape'`, `'aic'`, or `'bic'`. |
| `max_instances`      | int                | `100000`           | Maximum number of rows to use from `X` and `y`. |
| `max_reg`            | int                | `100`              | Maximum number of features allowed in the model. |
| `model_step`         | int                | `1`                | Interval used to evaluate nested models. |
| `chunk_size`         | int                | `500`              | Feature generation batch size (used for memory efficiency). |
| `seed`               | int                | `8`                | Random seed for reproducibility. |
| `cov_type`           | str                | `'nonrobust'`      | Covariance type for statsmodels fit (e.g., `'HC0'`, `'HC3'`). |



predict() Parameters
----------
| Parameter    | Type   | Default | Description |
|--------------|--------|---------|-------------|
| `X`          | NDArray | —       | Feature matrix for prediction (should match the training structure). |
| `confidence` | float  | `0.95`  | Confidence level for prediction and confidence intervals. |

Methods
-------
- fit(): Train model on input data
- predict(): Predict values and intervals
- load_emissions_report(): Return CodeCarbon emissions report

Attributes
----------
| Attribute         | Type               | Description |
|-------------------|--------------------|-------------|
| `sm_model`        | statsmodels model  | Final fitted model (linear, logit, or probit). |
| `best_score`      | float              | Best score achieved during model selection. |
| `X_total`         | NDArray            | Full engineered feature matrix (before feature selection). |
| `X`               | NDArray            | Final selected features used for modeling. |
| `y`               | NDArray            | Target vector used during training. |
| `pi_lower`        | NDArray            | Lower bounds of the prediction intervals (set after `predict`). |
| `pi_upper`        | NDArray            | Upper bounds of the prediction intervals (set after `predict`). |
| `ci_lower`        | NDArray            | Lower bounds of the confidence intervals (set after `predict`). |
| `ci_upper`        | NDArray            | Upper bounds of the confidence intervals (set after `predict`). |

