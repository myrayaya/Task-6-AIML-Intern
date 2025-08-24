# Task-6 : KNN Classification

## Objective
Implement the **K-Nearest Neighbors (KNN)** algorithm on a classification dataset to understand **instance-based learning**, distance metrics and the effect of different values of K.

## Dataset
- We used **Iris Dataset** from kaggle.com
- It contained various samples and data such as:-
  - Sepal Length
  - Sepal Width
  - Petal Length
  - Petal Width
  - Species

## Steps:-
1. Imported the Iris.csv dataset
2. Normalized features using **StandardScaler**
3. Split data into **train/test sets**
4. Traine KNN classifier with different 'k' values (1-15)
5. Evaluated Model using:
   - Acurracy Score
   - Confusion Matrix
6. Visualized:-
   - Accuracy vs K.
   - Confusion Matrix
   - Decision boundaries (using first two features)
  
## Results
<img width="717" height="557" alt="image" src="https://github.com/user-attachments/assets/b2c8fb5e-ec6d-43a6-ae9e-8f1be9f1a2a5" />

- Best Accuracy achieved for 'k = x'
<img width="907" height="592" alt="image" src="https://github.com/user-attachments/assets/f5d7a838-35f6-4461-a2e9-c4d4b9dca816" />

- Confusion Matrix and decision boundary plots included.
<img width="686" height="557" alt="image" src="https://github.com/user-attachments/assets/88053433-de5d-4970-ab87-c4b208c0db07" />
<img width="867" height="676" alt="image" src="https://github.com/user-attachments/assets/9088bcec-590a-4a85-be35-781186b3c549" />

## Author
- Myra Chauhan
