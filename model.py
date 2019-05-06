#%matplotlib inline
#Data manipulation 
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

#Data Visulization 
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.externals import joblib #saving the model
from sklearn.model_selection import train_test_split, GridSearchCV

from imblearn.over_sampling import SMOTE # balancing data
#Data Modeling 
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from yellowbrick.classifier import ClassificationReport
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.pipeline import Pipeline

###### Classifiers #####
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

## DL ##
from keras import Sequential
from keras.layers import Dense


def get_data(csv_file_name, header_file_name):
    
    """
    Gets the data file and adds the header to data
    """

    header = []
    lines = [line.rstrip() for line in open(header_file_name)] # get the header file
    for line in lines:
        header.append(line.replace(',',''))
        
    missing_values = ["n/a", "na", "--", ""] # Making a list of missing value types
       
    df = pd.read_csv(csv_file_name, names=header, skipinitialspace=True, na_values=missing_values) # read the csv file 
    
    df['BC']  = label_encoding(df,'BC') # categorical to numeric variable
    df['client_state'] = label_encoding(df,'client_state') #categorical to numeric variable
    
    #for key in df.keys():
    #df = detect_outlier(df, 'CPL_wrt_BC')
 
    
    
    df.to_csv('data_cleaned.csv')
    
    return df

def detect_outlier(data):
    """
    @detects outlier using Z-Score (y - mean)/std
    """
    
    outliers=[]
    threshold = 3
    mean = np.mean(data)
    std = np.std(data)
    
    for y in data:
        z_score = (y - mean)/std 
        if np.abs(z_score) > threshold:
            outliers.append(y)
    
    return outliers


def column_to_binary(df, namecol):  
    """
    converts categorical a feature to binary  
    I have not used it since we will have my features!!!
    """
    df = df.copy()
    df = pd.get_dummies(df, columns=[namecol], prefix = [namecol])

    return df


def correlation_test(df):
    """
    displays the corelations among all the features of input data frame
    """
    plt.figure(figsize=(12, 8))
    vg_corr = df.corr()
    sns.heatmap(vg_corr, 
            xticklabels = vg_corr.columns.values,
            yticklabels = vg_corr.columns.values,
            annot = True)

    
def label_encoding(df, columnName):

    """
    Converts categorical to numeric variable
    """
    le = preprocessing.LabelEncoder()
    df[columnName] = le.fit_transform(df[columnName])
    
    return df[columnName]


def model_ML(data):
    
    data = data.fillna(data.mean())
    
    """
    creates a machine learning model using RandomForestClassifier 
    uses pipeline to impute the missing values and replace with most frequent strategy
    standardizes the features to get standard normally distributed data  
    """
    X = data.drop('churn', axis=1).values #features 
    y = data['churn'].values #target

    #imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent') #setup the imputation
    imp = SimpleImputer(missing_values=np.nan, strategy='mean') #setup the imputation

    
    #n_estimators (The number of trees in the forest)
    clf = RandomForestClassifier(n_estimators=100, max_depth=50, random_state=2)
     
     ### oversampling 
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_sample(X, y)

    steps = [('imputation', imp),
             ('scaler', StandardScaler()),
             ('clf', clf )]

    pipeline = Pipeline(steps) # creat piplien 
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size = 0.2, random_state = 2)
    model = pipeline.fit(X_train, y_train) #fit the model
    y_pred = pipeline.predict(X_test) # predict the response
    y_prob=pipeline.predict_proba(X_test) # prob for each record

    
    # evaluate accuracy
    print('ML model accuracy score for the test:', pipeline.score(X_test,y_test)) #whether or not the model has over fited

    print ("ML model accuracy score:", accuracy_score(y_test, y_pred))
    
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    
    print('Model Performance:')
    target_names = ['retention', 'churn']
    
    # making prediction for out of sample data
    sample = [[-0.104875203,12,18,1,9,-0.084501058,768.3529346,23,98], 
              [-0.082850078,11,42,1,4,-0.319290368,661.7538154,5,150],
              [0.180727928,5,12,2,4,0.377594573,3639.980843,10,144],
              [-0.062326729,10,8,1,5,-0.833590931,625,7,85],
              [0.334965528,14,15,3,8,0.262987328,3266.800831,7,63]]

    preds = model.predict(sample) 
    pred_churn = [target_names[p] for p in preds] 
    print("Predictions:", pred_churn)

    return model


def model_DL(data):
    
    """
    creates a deep learning model with 2 hidden layers and 1 output -- this is inactive now
    """
    data= data.fillna(data.mean())
    
    X = data.drop('churn', axis=1).values #features 
    y = data['churn'].values #target
    
    sc = StandardScaler()
    X = sc.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    model = Sequential()
    model.add(Dense(4, activation='relu', kernel_initializer='random_normal', input_dim=9)) #1st HL
    model.add(Dense(4, activation='relu', kernel_initializer='random_normal'))  #2nd  HL
    model.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal')) #Output
    model.compile(optimizer ='adam',loss='binary_crossentropy', metrics =['accuracy']) # Compiling the NN
    model.fit(X_train, y_train, batch_size=10, epochs=10) #Fitting the data to the training dataset

    eval_model = model.evaluate(X_train, y_train)
    print(eval_model)
    
    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.5)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # making prediction for out of sample data #1st and last must be churn
    Xnew = [[-0.104875203,12,18,1,9,-0.084501058,768.3529346,23,98], 
              [-0.082850078,11,42,1,4,-0.319290368,661.7538154,5,150],
              [0.180727928,5,12,2,4,0.377594573,3639.980843,10,144],
              [-0.062326729,10,8,1,5,-0.833590931,625,7,85],
              [0.334965528,14,15,3,8,0.262987328,3266.800831,7,63]]
    
    for i in range(0,len(Xnew)):
        # make a prediction
        ynew = model.predict(np.array(Xnew))
        # show the inputs and predicted outputs
        print("X=%s, Predicted=%s" % (Xnew[i], ynew[i]))

def main():
    """
    Main entry of the program
    """
    csv_file_name = 'data.csv'
    header_file_name = 'header.txt'
    df = get_data(csv_file_name, header_file_name)
    correlation_test(df)
    model = model_ML(df)
    #model_DL(df)
    joblib.dump(model, 'advertising_ML.pkl') #save the model
    
    return
    
if __name__ == "__main__":
    main()