import data_preparation as t2
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import confusion_matrix, accuracy_score, log_loss, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle




def split2():
  X_train1, X_test, y_train1, y_test = t2.split()
  
  # Make a second split to create the final training set and the validation set and use random_state=2 (will use validation at later stage of project)
  X_train2, X_val, y_train2, y_val = train_test_split(X_train1, y_train1, train_size=0.8, test_size=0.2, random_state=2)
  
  return X_train2, X_val, y_train2, y_val


def train_mlp_model():
    X_train, X_test, y_train, y_test = t2.split()

    # Initialize a sequential model
    model = Sequential()

    # Import the Dense layer from Keras
    from keras.layers import Dense

    # add layers through iteration
    n = 7 #number of layers
    for i in range(1, n+1):
        if i != n:
            model.add(Dense(8, activation = "relu", name=f'layer{i}'))
        else:
            model.add(Dense(1, activation="sigmoid", name=f'layer{i}'))
    model.compile(loss="binary_crossentropy", optimizer="sgd")
    print(f'number of layers: {len(model.layers)}')
    
    # Train the model on the training data for 200 epochs
    model.fit(X_train, y_train, epochs=200) #batch_size default setting is max allowable
    # print(model.summary())

    toInt = lambda x: int(round(x))

    # Evaluate the model on the training data and compute loss and accuracy (and f1 score)
    y_train_pred = model.predict(X_train)
    train_length = len(y_train)
    # print(train_length)

    for i in range(len(y_train_pred)):
       pred = y_train_pred[i]
       y_train_pred[i] = toInt(pred[0])
    # print(y_train_pred)
    
    y_train_pred = y_train_pred.reshape(train_length,)
    # print(y_train_pred)
    
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_f1score = f1_score(y_train, y_train_pred)
    train_loss = log_loss(y_train, y_train_pred)
    print(f'training data model accuracy: {train_accuracy}; training data model f1 score: {train_f1score}; training data model log loss: {train_loss}')
    
    # Evaluate the model on the test data and compute loss and accuracy (and f1 score)
    y_test_pred = model.predict(X_test)
    test_length = len(y_test_pred)
    # print(test_length)

    for i in range(len(y_test_pred)):
       pred = y_test_pred[i]
       y_test_pred[i] = toInt(pred[0])
    # print(y_test_pred)
    
    y_test_pred = y_test_pred.reshape(test_length,)
    # print(y_test_pred)
    
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_f1score = f1_score(y_test, y_test_pred)
    test_loss = log_loss(y_test, y_test_pred)
    print(f'test data model accuracy: {test_accuracy}; test data model f1 score: {test_f1score}; test data model log loss: {test_loss}') 

    # return model
    print("returning...model")
    return model

def save_mlp_model(model):
    #save model
    with open("mlp_model.pkl", "wb") as f:
        pickle.dump(model, f)


def mlp_prediction_results():
    # load model
    with open("mlp_model.pkl", "rb") as f:
        mlp_model = pickle.load(f)

    X_train, X_test, y_train, y_test = t2.split()
    
    toInt = lambda x: int(round(x))
    
    # Perform prediction on the test data which whether each predicted probability is greater than 0.5 and convert to int
    y_test_pred = mlp_model.predict(X_test)
    test_length = len(y_test_pred)
    # print(test_length)

    for i in range(len(y_test_pred)):
       pred = y_test_pred[i]
       y_test_pred[i] = toInt(pred[0])
    # print(y_test_pred)
    
    y_test_pred = y_test_pred.reshape(test_length,)
    print(y_test_pred)
    # Compute the confusion matrix
    c_matrix = confusion_matrix(y_test, y_test_pred)
    
    ax = sns.heatmap(c_matrix, annot=True, xticklabels=['No Diabetes', 'Diabetes'],
                     yticklabels=['No Diabetes', 'Diabetes'], cbar=False, cmap='Blues')
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Actual")
    plt.show()
    plt.savefig('confusion_matrix.png')
    return c_matrix