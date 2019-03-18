import modelGenerate

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
    idx_to_label = {1: "Normal", 2: "Web", 0: "IoT"}
    num_steps = hyperparameters['num_steps']
    for i in range(0,test_data.shape[0], num_steps):
        true_label = "True Label: "
        predict_label = "Predicted: "
        data = next(test_data_generator.generate())
        predictions = model.predict(data[0])
        predict_traffics = np.argmax(predictions, axis=-1)
        for j in range(num_steps):
            true_label += test_data[i+j,-1]
            predict_label += idx_to_label[predict_traffics[j]]
            print(true_label)
            print(predict_label)
