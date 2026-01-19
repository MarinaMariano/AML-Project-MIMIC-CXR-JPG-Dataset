# CNNs FOR MEDICAL IMAGES DIAGNOSIS: TWO CASE STUDIES


## AIM 
The aim of this project is to develop, test and study the application of CNNs for medical diagnosis porpuses. Two datasets have been used : a Labeled Optical Coherence Tomography (OCT) dataset and a Chest X-Ray dataset (Available at: https://data.mendeley.com/datasets/rscbjbr9sj/3). CNNs are specialised neural Networks for image classification and their application in the medical field has in recent years, obtained great results improving  phisicians diagnosis. 

## Project Pipeline

### **Dataset organization**

The two datasets are organized into class-specific folders: The Labeled Optical Coherence Tomography (OCT)  is divided intwo four classes (`NORMAL`, `CNV`,`DRUSEN`, `DME`), suggesting the need of a multi class classification model. The Chest X-Ray dataset is organized into two folders (`NORMAL`, `PNEUMONIA`)


### **Train / Validation / Test split**
Both datasets are divided into training and test sets. A validation set is obtained by the initial training set, divided and randomised with a specific manner to avoid data leakage.The test set is used exclusively for final model evaluation.

### **Image preprocessing**

To allow  confrontations, all images are resized to `224x224` and converted to RGB format. and Pixel values are normalized to the `[0, 1]` range.

### **Data augmentation (training only)**

  Random transformations such as horizontal flipping, rotation, and zoom are applied: Data augmentation improves generalization and reduces overfitting.

### **Model architecture**

  - A Convolutional Neural Network based on **Transfer Learning** is employed (MobileNetV2) oppure ne creiamo una da zero noi?
  - The pre-trained backbone is initially frozen.
  - A custom fully connected classification head is added.

### **Evaluation metrics**

Accuracy, MCC and the standard statistical measures are used for general performance evaluation. AUC is included to provide a more robust metric for imbalanced datasets.

### **Training strategy**

Supervised training is performed using the training set and monitored on the validation set. Early Stopping is applied to prevent overfitting.

### **Fine-tuning**

Selected layers of the pre-trained backbone are unfrozen for fine-tuning. A lower learning rate is used during this phase.

### **Final evaluation**

The trained model is evaluated on the unseen test set. Final performance metrics are reported.

### **Reproducibility**

Modular and version-controlled codebase suitable for GitHub. The dataset is excluded from the repository using `.gitignore`.

