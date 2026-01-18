#  Charts generated

### Line Chart
<img width="830" height="442" alt="image" src="https://github.com/user-attachments/assets/761fbb4d-707d-4cb1-9cd2-e09d92526870" />

This data presents the total ETFs accumulation over time. There is a clear upward trend in ETFs accumulation.

### Bar Chart
<img width="830" height="492" alt="image" src="https://github.com/user-attachments/assets/1b1eb8ea-7202-44eb-9379-a7de9612aa12" />

The Bar chart represents the Mean value of each ETF fund flows in and out.

### TreeMap
<img width="830" height="570" alt="image" src="https://github.com/user-attachments/assets/c22ba5c3-2447-48d8-8562-2b22ec4d5dbd" />

This treemap clearly shows the market share of each ETF. 

### Scatter plot
<img width="830" height="450" alt="image" src="https://github.com/user-attachments/assets/230473e9-e004-47d7-ad4a-be9705e178ae" />

This scatter plot represents the exact values and date that ETFs flow in and flow out.

### Chart
<img width="830" height="482" alt="image" src="https://github.com/user-attachments/assets/b5cc5cb1-2a84-4ef5-9a3f-aee59cd4d3f7" />

This chart represents the correlation between ETFs accumulation and Bitcoin price. When the ETFs accumulation is between 0 to 10 billion dollars, there is clear positive linear relation between ETFs accumulation and Bitcoin price. 

### OLS Regression Model
<img width="666" height="450" alt="image" src="https://github.com/user-attachments/assets/396f4bfd-5c7a-4745-9049-21fac16a3a19" />

Regression model is applied to predict the Bitcoin price by using the predictor, ETFs accumulation. The result of R-squared is 0.433, which is moderate. The result of Intercept and coef is 4.828e+04, which means when the ETFs is 0, the Bitcoin price would be 48280. However, the day when ETFs launched, the Bitcoin price was around 42,000, therefore, This number is not very accurate.
The value of TOTAL_accumulated is 9.7e-07, which represents that every 100 million dollars flow into ETFs, the Bitcoin price would increase 97 dollars. This result is very close to the real value, because there is around 20 billion dollars flow into ETFs, and Bitcoin price has increase around 23,000 dollars.

### Support Vector Machine Model
<img width="818" height="320" alt="image" src="https://github.com/user-attachments/assets/af3eef00-d9a2-4646-9e1c-9e88cadfe990" />

The use of a Support Vector Machine (SVM) model is to predict the direction of Bitcoin price change. The data is split into training and testing sets using train_test_split with a 30% test size and a random_state for reproducibility. An SVM classifier is created with SVC(kernel='poly'). This specifies a polynomial kernel for the SVM.
33.33% accuracy means the model correctly predicted the direction of the price change for only about one-third of the test samples. This is barely better than random guessing for a binary classification problem.
