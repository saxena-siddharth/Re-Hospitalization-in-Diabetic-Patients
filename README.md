# Re-Hospitalization-in-Diabetic-Patients
The objective of this project is to build models which will predict the chances of readmissions
in patients whether a patient will be readmitted to the hospital or not. For
the sake of our predictive analyses, we have built KNN model and also a logistic
regression model.
To identify the best and the most significant features, we make use of Principle
Component Analysis (PCA) technique and correlation matrix. Furthermore, we
convert and pre-process the data by removing null values and changing categorical
variables into dummy variables which can be fed into our logistic model as an input.
Some of the categorical variables like “diag_1”, “diag_2”, “diag_3” although
contain numerical values, however in reality represent the diagnosis code and
converting them accurately would require some domain specific knowledge.
With our EDA we get insights into the age distribution, gender distribution as well
as the readmission rate. In conclusion, we provide an overall comparison of methods
based on their accuracy, sensitivity and specificity score.
Based on our predictive analysis, we obtain a higher accuracy from our KNN model
which comes out to be around ~80%, higher than logistic regression accuracy which
is around ~61%. Thus, we recommend the KNN model to the business users for
consideration.
