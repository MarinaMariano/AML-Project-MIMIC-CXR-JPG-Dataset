# A COMPARATIVE STUDY OF COSTUM AND PRETRAINED CNNs FOR CHEST X-RAY ClASSIFICATION


## INTRODUCTION 

This project was initially designed to evaluate the same CNN architecture on two different medical imaging datasets, namely Optical Coherence Tomography (OCT) and Chest X-Ray dataset, following the lightweight model proposed by Yen & Tsao (2024). The first one is a large multi-class dataset with 109,309 images, while the latter is a smaller dataset dividend into two classes and made of 5856 images.

The original idea was to keep the model architecture fixed and evaluate it on two different datasets: the Yen & Tsao CNN would have been applied to the Chest X-Ray dataset and, in parallel, to the much larger OCT dataset.
However, the OCT dataset scale and I/O overhead made an end-to-end training computationally impractical: despite multiple optimizations, model depth reduction, lower image resolution, reduced batch size and epochs, removal of expensive feature modules, and local dataset staging on Colab to avoid remote disk access training, the training time remained prohibitive (exceeding one hour per epoch).

To overcome these constraints while preserving a meaningful comparison, the project design shifted to a **comparison on the Chest X-ray dataset** only, **evaluating**:
- the lightweight **custom architecture** derived from Yen & Tsao (2024) (implemented by Luca), **and**
- the **pretrained** DenseNet-121 from TorchXRayVision with weights="densenet121-res224-all" (implemented by Marina).

A pretrained model is a neural network whose weights have already been optimized on large-scale datasets, providing a strong initialization for downstream tasks. Specifically, we used a pretrained DenseNet backbone and re-purpose it for **binary classification**, producing a discrete label **0/1** for the two target classes (**PNEUMONIA vs NORMAL**).
The transfer-learning pipeline consists of loading the pretrained feature extractor and replacing the original classifier with a task-specific head (binary output).
The backbone is kept fixed (non-trainable) and only the new classification head is optimized.

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

### OCT Dataset Split

| Dataset      | Number of Images | Classes |
|--------------|------------------|---------|
| Training     | 86,648           | 4       |
| Validation   | 21,661           | 4       |
| Test         | 1,000            | 4       |
| **Total**    | **109,309**      | 4       |

----

## **Image preprocessing**

To allow  confrontations, all images are resized to `224x224` and converted to RGB format (3 channels). Pixel values are normalized to the `[0, 1]` range. qui bisogna dire che abbiamo deciso d non fare più 224 x224 perché runnava troppo lentamente, pù inserire la lista di cambiamenti apportati ad altri parametri per lo stesso motivo. dire che io l'ho fatto non subito dopo aver iportato i dataset ma durante il training

### **Data augmentation for chest X-ray dataset**

Random transformations such as horizontal flipping, rotation, and zoom are applied to the chest-x ray dataset, to improve generalization and reduce overfitting. io non ho usato horizontale flipping ma solo rotatio e un'altra perché nelle immagini mediche invertire la simmetria anatomica ha poco senso, ma verificare questa cosa su articoli. dire che io l'ho fatto non subito dopo aver iportato i dataset ma durante il training

----

## **Model architecture**
We adopted the same model architecture based on a lightweight convolutional neural 
network inspired by the one proposed by Yen and Tsao (2024), specifcillay designed for chest X-ray classification (which consisted of a redesigned feature extraction (FE) module and multiscale feature (MF) module and validated using publicly available COVID-19 datasets).
The proposed CNN architecture is composed, indeed, of two main feature extraction stages, followed by a classification head. The binary head (single sigmoid output) was replaced with a 4-class softmax head to classify OCT images into CNV, DME, DRUSEN, and NORMAL. Minor hyperparameter adjustments (e.g., pooling sizes, dense units, dropout) were introduced to match the OCT task and computational constraints.

Reference: Yen, C.-T., & Tsao, C.-Y. (2024). Lightweight convolutional neural network for chest X-ray images classification. Scientific Reports, 14, 29759. https://doi.org/10.1038/s41598-024-80826-z

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

