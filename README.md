# CNNs FOR MEDICAL IMAGES DIAGNOSIS: TWO CASE STUDIES


## INTRODUCTION 
In this project we develop, test and study the application of CNNs for medical diagnosis porpuses. Specifically the core idea was to create a trainable network that didn't rely on transfer of knowledge from pretrained arhitectures, but could be entirely developed using the resources of edge devices. Two datasets have been used : a Labeled Optical Coherence Tomography (OCT) dataset and a Chest X-Ray dataset (Available at: https://data.mendeley.com/datasets/rscbjbr9sj/3). The first one is a large multi-class dataset with 115203 images, while the latter is a smaller dataset dividend into two classes and made of 5856 images. The use of CNNs in the medical field has increased in recent years, and obtained great results improving  phisicians diagnosis; for this reason it is important to develope scalable networks. 

----

## Project Pipeline


## **Dataset organization**

The two datasets are organized into class-specific folders: The Labeled Optical Coherence Tomography (OCT)  is divided intwo four classes (`NORMAL`, `CNV`,`DRUSEN`, `DME`), suggesting the need of a multi class classification model. The Chest X-Ray dataset is organized into two folders (`NORMAL`, `PNEUMONIA`)


### **Train / Validation / Test split**
Both datasets are divided into training, validation and test sets. The validation set is obtained by the initial training set, divided and randomised with a specific manner to avoid data leakage. The test set is used exclusively for final model evaluation.

### X-Ray Dataset Split

| Dataset      | Number of Images | Classes |
|--------------|------------------|---------|
| Training     | 4,186            | 2       |
| Validation   | 1,046            | 2       |
| Test         | 624              | 2       |
| **Total**    | **5,856**        | 2       |


----

## **Image preprocessing**

To allow  confrontations, all images are resized to `224x224` and converted to RGB format (3 channels). Pixel values are normalized to the `[0, 1]` range. qui bisogna dire che abbiamo deciso d non fare più 224 x224 perché runnava troppo lentamente, pù inserire la lista di cambiamenti apportati ad altri parametri per lo stesso motivo. dire che io l'ho fatto non subito dopo aver iportato i dataset ma durante il training

### **Data augmentation for chest X-ray dataset**

Random transformations such as horizontal flipping, rotation, and zoom are applied to the chest-x ray dataset, to improve generalization and reduce overfitting. io non ho usato horizontale flipping ma solo rotatio e un'altra perché nelle immagini mediche invertire la simmetria anatomica ha poco senso, ma verificare questa cosa su articoli. dire che io l'ho fatto non subito dopo aver iportato i dataset ma durante il training

----

## **Model architecture**
Our model architecture was based on a Lightweight convolutional neural 
network specifcillay designed for chest X-ray classification. The proposed CNN architecture is composed of two main feature extraction stages, followed by a classification head.
The design focuses on efficient feature extraction, multi-scale context aggregation, and low computational cost. prendi qualche riga da articolo e citalo dicendo che abbiamo usato quel modello.

### 1. Feature Extraction (FE Module)

The Feature Extraction (FE) module is designed to efficiently extract local spatial features while reducing redundancy. Structure:

- **1×1 Convolution**
- **Channel Split** (A portion of channels is kept unchanged,The remaining channels undergo further processing)
- **Depthwise 3×3 Convolution** (Captures local spatial patterns, Lightweight compared to standard convolutions)
- **1×1 Convolution** (Recombines channel information)
- **Concatenation** (Merges processed and unprocessed channel branches)
- **Residual Connection**

### 2. Multi-scale Feature Module (MF Module)

The Multi-scale Feature (MF) module captures contextual information at different spatial scales. Structure:

- Parallel Max Pooling branches with different receptive fields, pool sizes: 2×2, 4×4, 8×8
- Depthwise Dilated Convolutions, Different dilation rates per branch Capture both local and global context
- 1×1 Convolutions
- Align channel dimensions
-Concatenation Combines multi-scale representations
-Final 1×1 Convolution

### **3. Classification Head**

After feature extraction, the network uses a lightweight classification head:

- Global Average Pooling Reduces spatial dimensions Prevents overfitting
- Fully Connected Layer (128 units, ReLU)
- Dropout (0.5)
- Regularization
- Output Layer: 1 neuron with Sigmoid activation
- Binary classification

----

## **Evaluation metrics**

Accuracy, MCC and the standard statistical measures are used for general performance evaluation. AUC is included to provide a more robust metric for imbalanced datasets.


## **Training strategy**

Supervised training is performed using the training set and monitored on the validation set. Early Stopping is applied to prevent overfitting.

Training Configuration
- Optimizer: Adam
- Loss: Binary Crossentropy
- Batch size: 16
- Epochs : da definire
- Early stopping (patience: 5)

### **Final evaluation**

The trained model is evaluated on the unseen test set. Final performance metrics are reported.

### **Reproducibility**

Modular and version-controlled codebase suitable for GitHub. The dataset is excluded from the repository using `.gitignore`.

