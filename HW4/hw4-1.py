import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot  as plt
import cv2


def load_train_data(train_csv,image_size=(30,30)):  
    
    #Load CSV files
    train_labels = pd.read_csv(train_csv) 
    x_train=[]
    y_train=[]
    
    #Read the train image
    for _,row in train_labels.iterrows():
        img_path="/Users/ivanlin328/Desktop/CSE 276C/HW4/archive/" + row['Path']
        class_id=row['ClassId']
        
        #Get the ROI(Region of interest)coordinate and crop the image
        img=cv2.imread(img_path)
        x1,y1,x2,y2 = int(row['Roi.X1']),int(row['Roi.Y1']),int(row['Roi.X2']),int(row['Roi.Y2'])
        roi_img=img[y1:y2,x1:x2]
        
        #resize the cropped image and convert to grayscale
        roi_img= cv2.resize(roi_img,image_size)
        roi_img= cv2.cvtColor(roi_img,cv2.COLOR_BGR2GRAY)
        
        #Append the cropped image data and label(class_id)to the test lists
        x_train.append(roi_img)
        y_train.append(class_id)
        
    x_train=np.array(x_train).reshape(len(x_train),-1)
    y_train=np.array(y_train)
        
    return x_train,y_train
    
    
        
def load_test_data(test_csv,image_size=(30,30)): 
    
    #Load CSV files
    test_labels = pd.read_csv(test_csv) 
    x_test, y_test = [], []
        
        #Read the test image
    for _,row in test_labels.iterrows():
        img_path="/Users/ivanlin328/Desktop/CSE 276C/HW4/archive/" + row['Path']
        class_id=row['ClassId']
        
        #Get the ROI(Region of interest)coordinate and crop the image
        img=cv2.imread(img_path)
        x1,y1,x2,y2 = int(row['Roi.X1']),int(row['Roi.Y1']),int(row['Roi.X2']),int(row['Roi.Y2'])
        roi_img=img[y1:y2,x1:x2]
        
        #resize the cropped image and convert to grayscale
        roi_img= cv2.resize(roi_img,image_size)
        roi_img= cv2.cvtColor(roi_img,cv2.COLOR_BGR2GRAY)
        
        #Append the cropped image data and label(class_id)to the test lists
        x_test.append(roi_img)
        y_test.append(class_id)
        
    x_test=np.array(x_test).reshape(len(x_test),-1)
    y_test=np.array(y_test)
        
    return x_test, y_test

def apply_pca(x_train,x_test):
    pca=PCA(n_components=0.95)
    x_train_pca= pca.fit_transform(x_train)
    x_test_pca=pca.transform(x_test)
    components=pca.components_
    
    return x_train_pca, x_test_pca,components

def plot_eigenvectors_pca(components_):
     plt.figure(figsize=(10, 5))
     plt.subplot(121)
     plt.imshow(components_[0].reshape(30, 30))
     plt.title('1st Eigenvector')
     plt.subplot(122)
     plt.imshow(components_[1].reshape(30, 30))
     plt.title('2nd Eigenvector')
     plt.show()

def apply_lda(x_train,x_test,y_train):
    lda= LDA(n_components=len(np.unique(y_train))-1)
    x_train_lda=lda.fit_transform(x_train,y_train)
    x_test_lda=lda.transform(x_test)
    eigen=lda.scalings_[:, :2]
    
    return x_test_lda,x_train_lda,eigen

def plot_eigenvectors_lda(eigen):
     plt.figure(figsize=(10, 5))
     plt.subplot(121)
     plt.imshow(eigen[:,0].reshape(30, 30))
     plt.title('1st Eigenvector')
     plt.subplot(122)
     plt.imshow(eigen[:,1].reshape(30, 30))
     plt.title('2nd Eigenvector')
     plt.show()

def compute_accuracy(x_train, x_test, y_train, y_test):
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(x_train, y_train)
    y_pred = rf.predict(x_test)
    correct = accuracy_score(y_test, y_pred)
    incorrect = 1 - correct
    print(f"Accuracy: {correct:.4f}")
    print(f"Error Rate: {incorrect:.4f}")
    return correct, incorrect

# Load and preprocess data
x_train, y_train = load_train_data("/Users/ivanlin328/Desktop/CSE 276C/HW4/archive/Train.csv", image_size=(30,30))
x_test, y_test = load_test_data("/Users/ivanlin328/Desktop/CSE 276C/HW4/archive/Test.csv", image_size=(30,30))

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test= scaler.transform(x_test)

# Apply PCA and LDA
x_train_pca, x_test_pca, components = apply_pca(x_train, x_test)
x_test_lda, x_train_lda,eigen = apply_lda(x_train, x_test, y_train)

plot_eigen_pca= plot_eigenvectors_pca(components)
plot_eigen_lda=plot_eigenvectors_lda(eigen)

# Calculate recognition rates
print("PCA Recognition Rate:")
PCA_Recognition_Rate = compute_accuracy(x_train_pca, x_test_pca, y_train, y_test)

print("\nLDA Recognition Rate:")
LDA_Recognition_Rate = compute_accuracy(x_train_lda, x_test_lda, y_train, y_test)






    
    

        
        

        
        
        
        
    
        
        
        
        
        
        
        
   
                    
                    
                    
    
    
    
    
    
    
    
    
    
    

    
    

    
    
    
    
    
    
    
    
    







    
    