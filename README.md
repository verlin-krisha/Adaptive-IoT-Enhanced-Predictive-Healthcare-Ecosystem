# Adaptive-IoT-Enhanced-Predictive-Healthcare-Ecosystem
Performed data preprocessing and developed an RNN + LSTM model achieving 97% accuracy in predicting patient  health conditions. Integrated real-time sensor data visualization and alert notifications using ThingsBoard dashboard for  efficient monitoring.

 # Methodology 
# Step 1 - Dataset Processing
# Data Collection
Real-time health data from IoT-based sensors monitoring parameters like heart rate, blood pressure, SpO₂, temperature, humidity and ECG.
# Data Preprocessing
Handling missing values using K-Nearest Neighbors (KNN) imputation.
Encoding categorical data via one-hot encoding.
Normalizing numeric features using StandardScaler or MinMaxScaler.
Managing class imbalance using Synthetic Minority Over-sampling Technique (SMOTE).
Feature selection via Principal Component Analysis (PCA).

# Step 2 -  Model Architecture: RNN + LSTM
Input Layer  --  Accepts time-series patient health data in 3D format: [samples, timesteps, features].
 # RNN Layer
A Simple RNN layer extracts basic sequential features.
L2 Regularization, Batch Normalization, and Dropout (0.4) to prevent overfitting.
# LSTM Layer
LSTM layer for long-term dependency learning.
Further dropout (0.4) to improve generalization.
Output Layer
A Dense layer with Softmax activation for multi-class classification (disease detection).

# Step 3 -  Training & Optimization
Optimizer -  RMSprop (learning rate = 0.0005) for stable learning.
Hyperparameter Tuning
Adjusting batch size, learning rate, and number of units.
EarlyStopping to stop training if validation loss plateaus.
ReduceLROnPlateau to lower the learning rate dynamically when improvement stagnates.

# Step 4 - Model Evaluation
Accuracy, Precision, Recall, F1-score.
Receiver Operating Characteristic (ROC) curves & AUC scores.
Confusion Matrix to visualize misclassification.

# Step 5 -  Deployment & Application
Integrated into an IoT-based healthcare system for real-time patient monitoring.
Provides anomaly detection & predictive analytics for proactive medical interventions.
This methodology ensures efficient sequential data processing, real-time patient monitoring, and early disease detection using hybrid deep learning models.

# RNN + LSTM Algorithm
1. Import libraries (pandas, numpy, tensorflow, sklearn, imblearn). 
2. Read dataset from ”healthcare iot advanced preprocessed.csv”. 
3. Define features (X) and target (y) by binning target into 3 categories. 
4. Standardize features with StandardScaler(). 
5. One-hot encode target variable with to categorical(). 
6. Use SMOTE to balance class distribution. 
7. Split dataset into 80% training and 20% testing using stratified train test split(). 
8. Reshape data to RNN input format (samples, timesteps, features). 
9. Define Self-Attention Layer to improve feature learning. 
10. Create Transformer Encoder Layer with Multi-Head Attention and Feedforward Network. 
11. Implement Squeeze-and-Excitation Block for adaptive feature scaling. 
12. Define Learning Rate Scheduler using Cosine Annealing. 
13. Construct Hybrid Model with CNN, BiLSTM, Transformer, and Attention layers. 
14. Use AdamW optimizer with weight decay for better training stability. 
15. Compile model with categorical crossentropy loss and accuracy metric. 
16. Use callbacks (EarlyStopping, ReduceLROnPlateau, LearningRateScheduler). 
17. Train model for 200 epochs with batch size=64. 
18. Test performance by computing Training and Validation Accuracy. 
19. Make predictions and calculate Confusion Matrix.
20. Display Classification Report with precision, recall, and F1-score. 
21. Plot ROC Curves and AUC Scores for all classes. 
22. Save trained model as ”final optimized model.h5”. 
23. Print success message that model was trained and saved.

# Conclusion 
The Adaptive IoT-Enhanced Predictive Healthcare Ecosystem successfully addresses critical challenges in current IoT-based healthcare systems, such as latency, inaccurate anomaly detection, and improves them. By integrating IoT technologies, advanced machine learning models (RNN + LSTM), and explainable AI techniques, the system achieves:
 Real-time monitoring of vital health parameters using IoT-based wearable sensors.
 Improved predictive accuracy for anomaly detection through sequential deep learning models.
Reduced hospital workload through automated early disease detection and remote monitoring capabilities.
Enhanced robustness and generalization of predictive models through advanced data preprocessing and hyperparameter tuning
The results demonstrate a high accuracy rate (97%) for disease detection, validating the effectiveness of the RNN + LSTM architecture in processing time-series health data. Furthermore, the integration with the ThingsBoard dashboard ensures seamless real-time data visualization and anomaly alerts.














