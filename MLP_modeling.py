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


# Define a global variable to hold the model object
model = None

def train_model():
    global model
    
    X_train, X_test, y_train, y_test = t2.split()
    X_train2, X_val, y_train2, y_val = split2()

    # Initialize a sequential model
    model = Sequential()

    # Import the Dense layer from Keras
    from keras.layers import Dense
    model.add(Dense(8, activation="relu", name="layer1"))
    model.add(Dense(8, activation="relu", name="layer2"))
    model.add(Dense(8, activation="relu", name="layer3"))
    model.add(Dense(8, activation="relu", name="layer4"))
    model.add(Dense(8, activation="relu", name="layer5"))
    model.add(Dense(8, activation="relu", name="layer6"))
    model.add(Dense(1, activation="sigmoid", name="layer7"))
    model.compile(loss="binary_crossentropy", optimizer="sgd")
    print(f'number of layers: {len(model.layers)}')
    
    # Train the model on the training data for 200 epochs
    model.fit(X_train2, y_train2, epochs=200) #batch_size default setting is max allowable
    print(model.summary())

    toInt = lambda x: int(round(x))

    # Evaluate the model on the training data and compute loss and accuracy
    y_train2_pred = model.predict(X_train2)
    train_length = len(y_train2)
    print(train_length)

    for i in range(len(y_train2_pred)):
       pred = y_train2_pred[i]
       y_train2_pred[i] = toInt(pred[0])
    print(y_train2_pred)
    
    y_train2_pred = y_train2_pred.reshape(train_length,)
    print(y_train2_pred)
    
    training2_accuracy = accuracy_score(y_train2, y_train2_pred)
    train2_f1score = f1_score(y_train2, y_train2_pred)
    training2_loss = log_loss(y_train2, y_train2_pred)
    print(f'training data model accuracy: {training2_accuracy}; training data model log loss: {training2_loss}')
    
    # Evaluate the model on the test data and compute loss and accuracy
    y_test_pred = model.predict(X_test)
    test_length = len(y_test_pred)
    print(test_length)

    for i in range(len(y_test_pred)):
       pred = y_test_pred[i]
       y_test_pred[i] = toInt(pred[0])
    print(y_test_pred)
    
    y_test_pred = y_test_pred.reshape(test_length,)
    print(y_test_pred)
    
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_f1score = f1_score(y_test, y_test_pred)
    test_loss = log_loss(y_test, y_test_pred)
    print(f'test data model accuracy: {test_accuracy}; test data model log loss: {test_loss}') 

    # Return the training and test accuracies rounded to two decimal places for both trained and tested data
    print("returning: training accuracy, training f1 score, test accuracy, test f1 score, model")
    return round(training2_accuracy, 3), round(train2_f1score, 3), round(test_accuracy, 3), round(test_f1score, 3), model

def save_model(model):
    #save model
    with open("mlp_model.pkl", "wb") as f:
        pickle.dump(model, f)


def prediction():
    # load model
    with open("mlp_model.pkl", "rb") as f:
        mlp_model = pickle.load(f)

    X_train, X_test, y_train, y_test = t2.split()
    
    toInt = lambda x: int(round(x))
    
    # Perform prediction on the test data which whether each predicted probability is greater than 0.5 and convert to int
    y_test_pred = mlp_model.predict(X_test)
    test_length = len(y_test_pred)
    print(test_length)

    for i in range(len(y_test_pred)):
       pred = y_test_pred[i]
       y_test_pred[i] = toInt(pred[0])
    print(y_test_pred)
    
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