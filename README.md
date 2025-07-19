# AZ Watch – Subscriber Churn Prediction & Segmentation

Business Use Case (Fictional) :
Helped a streaming platform (AZ Watch) predict subscriber churn and identify behavioral segments to reduce retention loss and launch personalized marketing strategies.

Tools & Techniques Used:

Python, pandas, scikit-learn, matplotlib
- Logistic Regression, Random Forest, Decision Tree
- K-Means Clustering for unsupervised segmentation
- Label Encoding, Feature Scaling, Confusion Matrix

AZ Watch wanted to:
- Predict which subscribers were likely to churn
- Understand what behaviors drove churn
- Group subscribers into meaningful segments for campaign targeting



Action Steps:

- Cleaned & Preprocessed Data: Encoded categorical variables and scaled features

- Trained 3 Classification Models:
Logistic Regression outperformed others with 91% accuracy

-I nterpreted Churn Drivers: Identified low engagement time and frequency as key indicators

- Segmented Users via KMeans:
Cluster 0: Low engagement — highest churn risk
Cluster 1: Moderate use — opportunity to upsell
Cluster 2: Frequent visits — personalize content

- Visualized Results: Confusion matrix and cluster analysis


Results
- Achieved 91% accuracy using Logistic Regression
- Identified 3 subscriber segments with unique behaviors
- Proposed targeted retention strategies per segment



