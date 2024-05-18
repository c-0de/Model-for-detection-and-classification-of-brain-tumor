BRAIN TUMOR CLASSIFICATION USING CNN AND ML TECHNIQUES

DATASET:-

Source: https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri?

The dataset contains 826 images of glioma, 822 images of meningioma, 827 images of pituitary tumors, and 395 images without any tumors.

OVERVIEW:-

Brain tumors are abnormal growth of cells in the brain. They are a critical health concern, with diverse manifestations and potential life-threatenin consequences. They can be benign (non-cancerous) or malignant 
(cancerous) and can arise from diverse types of brain cells. The location, size, and type of brain tumor can impact how it affects brain function and health. As brain tumors are classified according to several
criteria, it becomes complex and hard to differentiate between the types of tumors. It is important to effectively classify the type and grade of tumors for suitable treatment.
The process of identifying brain tumors is intricate, given the diverse shapes, sizes, and locations they can take within the structure of the brain. Moreover, addressing issues like noise and artifacts in MRI 
scans adds another layer of complexity. Additionally, the task involves classifying tumors into multiple categories. Effectively addressing these challenges is crucial to ensure the system produces dependable and 
accurate results, facilitating early and precise diagnoses. To deal with these problems, the proposed model includes various stages, including preprocessing, feature extraction, and classification.
During preprocessing, techniques such as resizing, Gaussian blurring, normalization, and shuffling are applied to the MRI images. Following this, the Gray Level Co-occurrence Matrix (GLCM) is used to extract 
features including energy, contrast, entropy, homogeneity, correlation, and dissimilarity. These extracted features are then used for classification through K-Nearest Neighbor (K-NN), Support Vector Machine (SVM),
and Convolutional Neural Network (CNN) algorithms. The achieved accuracies stand at 82% for SVM, 84% for KNN, and 90% for CNN.

Architecture of the model:-

![Model Design](https://github.com/c-0de/Model-for-detection-and-classification-of-brain-tumor/assets/141239361/066a12fc-6c94-4f71-965c-c34654781534)

RESULTS:-

After Preprocessing:-

![Screenshot (109)](https://github.com/c-0de/Model-for-detection-and-classification-of-brain-tumor/assets/141239361/a8340bf2-6758-4ccd-a465-1119826794c2)

![Screenshot (110)](https://github.com/c-0de/Model-for-detection-and-classification-of-brain-tumor/assets/141239361/107e30ce-f5bc-4525-ae2a-6c7c70cdacf3)

After Feature Extraction:-

![Screenshot (111)](https://github.com/c-0de/Model-for-detection-and-classification-of-brain-tumor/assets/141239361/5d53f2cb-023d-4c9a-86a1-36ccaa5cfcc3)

After Classification:-

Sensitivity:

![Screenshot (112)](https://github.com/c-0de/Model-for-detection-and-classification-of-brain-tumor/assets/141239361/cd596dcd-eb3b-42d5-aa3a-8a1fd9a3c431)

Specificity:

![Screenshot (113)](https://github.com/c-0de/Model-for-detection-and-classification-of-brain-tumor/assets/141239361/c6ff2dea-58e7-476e-a1b2-7e04e50942ed)

Accuracy:

![Screenshot (114)](https://github.com/c-0de/Model-for-detection-and-classification-of-brain-tumor/assets/141239361/9c840e86-efd5-47c6-b5b8-f5aedf31d169)
