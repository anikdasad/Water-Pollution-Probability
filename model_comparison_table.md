# Machine Learning Models Comparison for Water Potability Prediction

| Model                  |   Accuracy |   Precision |   Recall |   F1-Score |   CV Mean |   CV Std |
|:-----------------------|-----------:|------------:|---------:|-----------:|----------:|---------:|
| Logistic Regression    |     0.628  |      0.3944 |   0.628  |     0.4846 |    0.6057 |   0.0015 |
| Random Forest          |     0.6784 |      0.6656 |   0.6784 |     0.6568 |    0.6687 |   0.0103 |
| Gradient Boosting      |     0.6585 |      0.6424 |   0.6585 |     0.6157 |    0.6355 |   0.012  |
| Support Vector Machine |     0.6951 |      0.6954 |   0.6951 |     0.6596 |    0.6687 |   0.0089 |
| K-Nearest Neighbors    |     0.628  |      0.6167 |   0.628  |     0.6201 |    0.6275 |   0.0089 |
| Naive Bayes            |     0.6311 |      0.6003 |   0.6311 |     0.5837 |    0.6202 |   0.0128 |
| Decision Tree          |     0.5777 |      0.5912 |   0.5777 |     0.5827 |    0.5729 |   0.0064 |

## Key Findings:
- **Best Overall Model**: Based on F1-Score and CV performance
- **Most Stable Model**: Based on CV Standard Deviation
- **Fastest Model**: Based on training time
