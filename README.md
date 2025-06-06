# INN Booking Cancellation Prediction using Decision Tree

This project explores the use of Decision Tree algorithms to predict hotel booking cancellations.Exploratory data analysis (EDA), feature engineering, and model training and tuning (including pre-pruning, post-pruning, and cost-complexity pruning) are performed to build interpretable models for classification.

**Problem Statement**

The goal is to predict whether a hotel booking will be canceled or not, based on input features like lead time, average room price, market segment, special requests, and more.

**Dataset**

- The dataset contains booking information from a hotel.
- Target variable: booking_status (0 = Not Canceled, 1 = Canceled).
- Features include customer behavior, pricing, booking method, and stay details.

**Steps Performed**

1.Data Preprocessing
- Dropped unique ID (Booking_ID) and irrelevant column (arrival_year).
- Converted categorical features using get_dummies() and boolean values to binary (0,1).
- Target column encoded: 'Not_Canceled' → 0, 'Canceled' → 1.
- Features normalized using StandardScaler for lead_time and avg_price_per_room.

2. EDA (Exploratory Data Analysis)
- Verified class balance (~67% Not Canceled, ~33% Canceled).
- Identified feature ranges requiring normalization.
- Plotted feature importance and distributions.

3. Decision Tree Models
   
**Model 1 – Default Decision Tree (Gini)**

- High training accuracy: 99.4%
- Overfitting observed (Test accuracy ~86.5%)

**Model 2 – Pre-Pruned Tree (via GridSearchCV)**

- Used max_depth, min_samples_leaf, max_leaf_nodes, min_impurity_decrease as hyperparameters.
- Improved generalization.
- Test Accuracy: 82.8%

**Model 3 – Cost Complexity Post-Pruning**

- Computed ccp_alpha via cost_complexity_pruning_path.
- Best ccp_alpha chosen by evaluating test accuracy and F1-score.
- Final accuracy: 88.2%

**Visualizations**

- Confusion Matrix
- Feature Importance Bar Charts
- Decision Tree Plots (Full and Pruned)
- Accuracy and F1 Score vs. Alpha Plots (for pruning)

**Conclusion**

- Post-pruning significantly improves generalization compared to default trees.
- Hyperparameter tuning is crucial to prevent overfitting.
- The cost complexity pruning approach with alpha tuning offers a well-balanced model.
