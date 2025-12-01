import os
import scipy.io as sio
from self_py_fun.HW10Fun import *
from sklearn.linear_model import LogisticRegression as LR
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


# In HW7, you have the chance to visualize a truncated EEG dataset stratified by
# target and non-target stimulus type.
#
# The fundamental problem of P300 ERP-BCI speller system is to perform a binary classification.
#
# In HW10, you are asked to implement the binary classification using various methods,
# and evaluate the model performance with a testing dataset.
#
# You will use K114_001_BCI_TRN_Truncated_Data_0.5_6.mat as a training set, and
# K114_001_BCI_FRT_Truncated_Data_0.5_6.mat as a testing set.
#
# Notice that here, we do not split training/testing within K114_001_BCI_TRN_Truncated_Data_0.5_6.mat
# because each row is not entirely independent of each other due to the special structure of the dataset.

# Global constants:
np.random.seed(100)
bp_low = 0.5
bp_upp = 6
electrode_num = 16
parent_dir = "E:/Git Hub/BIOS-584"
parent_data_dir = f"{parent_dir}/data"
time_index = np.linspace(0, 800, 25)
electrode_name_ls = ['F3', 'Fz', 'F4', 'T7', 'C3', 'Cz', 'C4', 'T8',
                     'CP3', 'CP4', 'P3', 'Pz', 'P4', 'PO7', 'PO8', 'Oz']
subject_name = 'K114'
subject_dir = f"{parent_dir}/{subject_name}"
if not os.path.exists(subject_dir):
    os.mkdir(subject_dir)

char_trn = "THE0QUICK0BROWN0FOX"
char_trn_size = len(char_trn)

# Step 1: Import dataset
# Step 1.1: TRN dataset
trn_data_name = f"{subject_name}_001_BCI_TRN_Truncated_Data_{bp_low}_{bp_upp}"
trn_data_dir = f"{parent_data_dir}/{trn_data_name}.mat"
eeg_trn_obj = sio.loadmat(trn_data_dir)

# eeg_trn_obj is a dictionary!
print(eeg_trn_obj.keys())
eeg_trn_signal = eeg_trn_obj["Signal"]
print(eeg_trn_signal.shape)  # (3420, 400)
eeg_trn_type = eeg_trn_obj["Type"]
print(eeg_trn_type.shape)    # (3420, 1)
eeg_trn_type = eeg_trn_type.ravel()

# Step 1.2: FRT dataset
# The following code should be completed by students themselves.
# you should be able to obtain relevant data files named
# eeg_frt_signal and eeg_frt_type
# Write your own code below:
frt_data_name = f"{subject_name}_001_BCI_FRT_Truncated_Data_{bp_low}_{bp_upp}"
frt_data_dir = f"{parent_data_dir}/{frt_data_name}.mat"
eeg_frt_obj = sio.loadmat(frt_data_dir)

eeg_frt_signal = eeg_frt_obj["Signal"]
print(eeg_frt_signal.shape)

eeg_frt_type = eeg_frt_obj["Type"]
print(eeg_frt_type.shape)
eeg_frt_type = eeg_frt_type.ravel()

# Calculate trn_seq_size (MISSING IN YOUR CODE)
trn_seq_size = int(eeg_trn_signal.shape[0] / char_trn_size / 12)

# You have completed the exploratory data analysis in HW7 and HW8.
# The dataset has been carefully reviewed by Dr. Jane E. Huggins,
# so we do not need to worry about missing, outliers, errors of the dataset.

# Step 2: Fit classification models
# You will try the following methods:
# Logistic Regression,
# Linear Discriminant Analysis,
# Support Vector Machine (sometimes called support vector classification)
# You do not need to modify the parameters of each classifier
# except for LogisticRegression: set max_iter=1000
# Write your own code below:
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
eeg_trn_signal_scaled = scaler.fit_transform(eeg_trn_signal)
eeg_frt_signal_scaled = scaler.transform(eeg_frt_signal)

# Logistic Regression (uses scaled data for better convergence)
logistic_model = LR(max_iter=1000)
logistic_model.fit(eeg_trn_signal_scaled, eeg_trn_type)

# LDA (uses original data - LDA has built-in scaling)
lda_model = LDA()
lda_model.fit(eeg_trn_signal, eeg_trn_type)

# SVM (uses original data)
svm_model = SVC(probability=True)
svm_model.fit(eeg_trn_signal, eeg_trn_type)

# Step 3: Evaluate model performance on both TRN and FRT files
# Step 3.1: Prediction accuracy on TRN files
# We will compute the probability of each stimulus with .predict_proba()
# before we convert the stimulus-level probability to character-level probability.
# You are asked to generate stimulus-level probability for each method on TRN files,
# denoted as logistic_y_trn, lda_y_trn, and svm_y_trn.
# Write your own code below:
logistic_y_trn = logistic_model.predict_proba(eeg_trn_signal_scaled)
lda_y_trn = lda_model.predict_proba(eeg_trn_signal)
svm_y_trn = svm_model.predict_proba(eeg_trn_signal)

# Step 3.2: Prediction accuracy on FRT files
# Similarly, you are asked to generate stimulus-level probability for each method on FRT files,
# denoted as logistic_y_frt, lda_y_frt, and svm_y_frt.
# Write your own code below:
logistic_y_frt = logistic_model.predict_proba(eeg_frt_signal_scaled)
lda_y_frt = lda_model.predict_proba(eeg_frt_signal)
svm_y_frt = svm_model.predict_proba(eeg_frt_signal)

# Step 4: Convert binary classification probability to character-level accuracy
# This involves advanced data manipulation, so you do not need to write any new code.
# Please run the following code to view the final results
eeg_trn_code = eeg_trn_obj['Code']
eeg_frt_code = eeg_frt_obj['Code']
char_frt = convert_raw_char_to_alphanumeric_stype(eeg_frt_obj['Text'])
char_frt_size = len(char_frt)
frt_seq_size = int(eeg_frt_signal.shape[0] / char_frt_size / 12)

# Logistic regression
print('Logistic Regression on TRN:')
logistic_letter_mat_trn, logistic_letter_prob_mat_trn = streamline_predict(
    logistic_y_trn, eeg_trn_type, eeg_trn_code, char_trn_size, trn_seq_size,
    stimulus_group_set, eeg_rcp_array
)
print(logistic_letter_mat_trn)
print(list(char_trn))
logistic_trn_accuracy = np.mean(logistic_letter_mat_trn == np.array(list(char_trn))[:, np.newaxis], axis=0)

print('Logistic Regression on FRT:')
logistic_letter_mat_frt, logistic_letter_prob_mat_frt = streamline_predict(
    logistic_y_frt, eeg_frt_type, eeg_frt_code, char_frt_size, frt_seq_size,
    stimulus_group_set, eeg_rcp_array
)
print(logistic_letter_mat_frt)
print(list(char_frt))
logistic_frt_accuracy = np.mean(logistic_letter_mat_frt == np.array(list(char_frt))[:, np.newaxis], axis=0)

# LDA:
print('LDA on TRN:')
lda_letter_mat_trn, lda_letter_prob_mat_trn = streamline_predict(
    lda_y_trn, eeg_trn_type, eeg_trn_code, char_trn_size, trn_seq_size,
    stimulus_group_set, eeg_rcp_array
)
print(lda_letter_mat_trn)
print(list(char_trn))
lda_trn_accuracy = np.mean(lda_letter_mat_trn == np.array(list(char_trn))[:, np.newaxis], axis=0)

print('LDA on FRT:')
lda_letter_mat_frt, lda_letter_prob_mat_frt = streamline_predict(
    lda_y_frt, eeg_frt_type, eeg_frt_code, char_frt_size, frt_seq_size,
    stimulus_group_set, eeg_rcp_array
)
print(lda_letter_mat_frt)
print(list(char_frt))
lda_frt_accuracy = np.mean(lda_letter_mat_frt == np.array(list(char_frt))[:, np.newaxis], axis=0)

# SVM:
print('Support Vector Machine on TRN:')
svm_letter_mat_trn, svm_letter_prob_mat_trn = streamline_predict(
    svm_y_trn, eeg_trn_type, eeg_trn_code, char_trn_size, trn_seq_size,
    stimulus_group_set, eeg_rcp_array
)
print(svm_letter_mat_trn)
print(list(char_trn))
svm_trn_accuracy = np.mean(svm_letter_mat_trn == np.array(list(char_trn))[:, np.newaxis], axis=0)

print('Support Vector Machine on FRT:')
svm_letter_mat_frt, svm_letter_prob_mat_frt = streamline_predict(
    svm_y_frt, eeg_frt_type, eeg_frt_code, char_frt_size, frt_seq_size,
    stimulus_group_set, eeg_rcp_array
)
print(svm_letter_mat_frt)
print(list(char_frt))
svm_frt_accuracy = np.mean(svm_letter_mat_frt == np.array(list(char_frt))[:, np.newaxis], axis=0)

print(logistic_trn_accuracy)
print(lda_trn_accuracy)
print(svm_trn_accuracy)

print(logistic_frt_accuracy)
print(lda_frt_accuracy)
print(svm_frt_accuracy)

print("\n==========================")
print("FINAL SUMMARY: FRT Accuracy")
print("==========================")
print(f"Logistic Regression FRT Accuracy: {logistic_frt_accuracy[-1]:.4f}")
print(f"LDA FRT Accuracy: {lda_frt_accuracy[-1]:.4f}")
print(f"SVM FRT Accuracy: {svm_frt_accuracy[-1]:.4f}")

print("\n==========================")
print("FINAL SUMMARY: TRN Accuracy")
print("==========================")
print(f"Logistic Regression TRN Accuracy: {logistic_trn_accuracy[-1]:.4f}")
print(f"LDA TRN Accuracy: {lda_trn_accuracy[-1]:.4f}")
print(f"SVM TRN Accuracy: {svm_trn_accuracy[-1]:.4f}")



# Remember to answer two questions below:

# What do rows 122, 131, 141, 150, 160, and 169 do? Briefly answer the question below:
# In case that your row IDs are messed up when you start to fill in the blank,
# I attach the lines of code for your reference.
# logistic_trn_accuracy = np.mean(logistic_letter_mat_trn == np.array(list(char_trn))[:, np.newaxis], axis=0)
# logistic_frt_accuracy = np.mean(logistic_letter_mat_frt == np.array(list(char_frt))[:, np.newaxis], axis=0)
# lda_trn_accuracy = np.mean(lda_letter_mat_trn == np.array(list(char_trn))[:, np.newaxis], axis=0)
# lda_frt_accuracy = np.mean(lda_letter_mat_frt == np.array(list(char_frt))[:, np.newaxis], axis=0)
# svm_trn_accuracy = np.mean(svm_letter_mat_trn == np.array(list(char_trn))[:, np.newaxis], axis=0)
# svm_frt_accuracy = np.mean(svm_letter_mat_frt == np.array(list(char_frt))[:, np.newaxis], axis=0)

"""
These lines calculate the character-level accuracy for each classification method on both 
training and testing datasets. Specifically:
1. They compare the predicted letter matrix (from streamline_predict) with the true characters
2. np.array(list(char_trn/char_frt))[:, np.newaxis] converts the true character string to a 
  column array for broadcasting
3. The == operator creates a boolean array indicating correct predictions
4. np.mean(..., axis=0) computes the accuracy across all characters for each prediction sequence
5. This gives accuracy as a function of the number of sequences used for classification
"""

# Step 5: Summary
# Which method performs the best? Why?
"""
RESULTS:
FRT Accuracy
==========================
Logistic Regression FRT Accuracy: 1.0000
LDA FRT Accuracy: 0.9630
SVM FRT Accuracy: 1.0000
==========================
TRN Accuracy
==========================
Logistic Regression TRN Accuracy: 1.0000
LDA TRN Accuracy: 1.0000
SVM TRN Accuracy: 1.0000
==========================

Both Logistic Regression and SVM achieve perfect performance with 100% accuracy on the FRT (testing) dataset, while LDA achieves 96.30% accuracy. All three methods show excellent performance on the training set (100% accuracy).
Why Logistic Regression and SVM perform best:

Logistic Regression:
Feature standardization allows the optimizer to converge properly, creating an effective linear decision boundary
The scaled features help the gradient descent algorithm navigate the high-dimensional space (400 features) efficiently
Despite being a simple linear classifier, it captures the essential discriminative patterns in P300 responses
Perfect generalization (100% TRN and FRT) indicates the model found a robust linear separator without overfitting


SVM:
The RBF kernel can model non-linear relationships in the P300 ERP signal across electrodes and time points
SVM's margin maximization provides good generalization even with high-dimensional data
The probability estimates from the decision function are well-calibrated for the character-level prediction task

While LDA achieves strong performance (96.30%), it falls slightly short of perfect accuracy because:
LDA assumes multivariate Gaussian distributions with equal covariance matrices, which may not perfectly hold for EEG data
The method is sensitive to the correlation structure among the 400 features

Conclusion:
Both Logistic Regression and SVM are optimal choices for this P300 BCI speller system, achieving perfect character recognition on unseen data. The key to Logistic Regression's success was feature standardization, which enabled proper convergence and created an effective decision boundary. For practical deployment, Logistic Regression might be preferred due to its simplicity and computational efficiency, while SVM offers the flexibility of non-linear decision boundaries if needed for other subjects or conditions
"""