from modelGenerate import load_train_data, load_test_data, KerasBatchGenerator, get_hyperparameters
from keras.models import load_model
import numpy as np

"""
Predictions using test set
"""
def predict(data_path, hyperparameters):
    train_data, num_features = load_train_data('all_training_processed.csv')
    test_data = load_test_data('all_test_processed.csv')
    full_path = data_path + "/model-" + str(hyperparameters['num_epochs']) + ".hdf5"
    model = load_model(full_path)
    generator_parameters = {'batch_size': 1, 'num_classes':3, 'num_steps':25}
    test_data_generator = KerasBatchGenerator(test_data, generator_parameters, num_features)
    #idx_to_label = {1: "Normal", 2: "Web", 0: "IoT"}
    num_steps = hyperparameters['num_steps']
    correct_predictions = 0

    for i in range(0,test_data.shape[0], num_steps):
        data = next(test_data_generator.generate())
        predictions = model.predict(data[0])
        predict_traffics = np.argmax(predictions, axis=-1)
        for j in range(num_steps):
             if(i+j == test_data.shape[0]):
                 break
             elif(test_data[i+j,-1] == predict_traffics[0][j]):
                 correct_predictions += 1
                 
    return correct_predictions / test_data.shape[0]

hyperparameters = get_hyperparameters()
path_lst = ['lstm_model_history','gru_model_history','bilstm_model_history']

for path in path_lst:
    accuracy = predict(path, hyperparameters)
    print(path + " accuracy: " + str(accuracy))
