Overview
This repository contains the work for developing a machine learning-based screening tool for detecting potential MPOX (formerly monkeypox) virus infections. The objective is to create an inexpensive and quick method for predicting MPOX infections without the need for extensive PCR tests, which can be costly and time-consuming. This project was completed as part of the Data Mining & Machine Learning module at the University of Westminster.
Domain Context
MPOX, caused by the monkeypox virus (MPXV), emerged as a significant global health concern, leading to widespread outbreaks. The traditional diagnostic method, PCR testing, although accurate, is expensive and not conducive to rapid mass screening, particularly during a pandemic. This project aims to mitigate this issue by leveraging machine learning models to predict MPOX infections based on patient symptoms and other relevant attributes.
Tools and Technologies Used
•	Python: The primary programming language for implementing the machine learning models.
•	Scikit-Learn: Utilized for implementing various machine learning algorithms.
•	Pandas and NumPy: Used for data manipulation and analysis.
•	Matplotlib and Seaborn: For data visualization.
•	GridSearchCV: For hyperparameter tuning and model optimization.
•	Google Colab: Used for running the code and experiments.
Project Workflow
The project followed the CRISP-DM methodology, comprising the following phases:
1.	Domain Understanding:
o	Analyzed the problem and identified key attributes to be used for model training.
2.	Data Understanding:
o	Explored and described the dataset statistically.
o	Plotted the distribution of the target variable (MPOX).
3.	Data Preparation:
o	Identified and addressed data quality issues.
o	Transformed and cleaned the data for better model performance.
4.	Modeling:
o	Implemented and trained multiple classification models, including:
	Logistic Regression
	Decision Trees
	Support Vector Machine (SVM) with RBF kernel
	Naive Bayes
o	Evaluated each model's performance using confusion matrices and various metrics (accuracy, recall, precision, F1-score, AUC-ROC).
5.	Evaluation:
o	Selected the best model based on evaluation metrics.
o	Conducted hyperparameter tuning using GridSearchCV to enhance model performance.
o	Created an ensemble model to further improve prediction accuracy.
6.	Deployment:
o	Although not covered in this coursework, the final model is designed for potential deployment in real-world scenarios for rapid MPOX screening.
Results
•	Best Performing Model: The Support Vector Machine (SVM) with RBF kernel was identified as the best-performing model based on its precision and recall scores.
•	Model Accuracy: The SVM model achieved an accuracy of XX%, with a recall of YY% and a precision of ZZ%.
•	Hyperparameter Tuning: The performance of the SVM model improved slightly after hyperparameter tuning using GridSearchCV, indicating a robust model.
•	Ensemble Model: Combining the SVM and Decision Tree models into an ensemble resulted in a slight improvement in overall prediction accuracy.
Achievements
•	Successfully implemented and evaluated multiple machine learning models to predict MPOX infections.
•	Identified the most effective model based on performance metrics, optimizing it for better accuracy and reliability.
•	Developed a cost-effective screening tool that can minimize the dependency on expensive PCR tests.
Limitations and Future Work
•	The model's performance is dependent on the quality and completeness of the input data. Missing or inaccurate data can affect predictions.
•	Further research is needed to enhance the model's generalizability across different populations and regions.
•	Ethical considerations, such as data privacy and potential biases in the model, need to be addressed before deployment.
This repository includes the complete code and documentation for replicating the analysis and results. For detailed code and step-by-step implementation, please refer to the provided Google Colab Notebooks.

