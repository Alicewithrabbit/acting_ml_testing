import data_getter
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

def main():
    
    data_loc = 'no_name_counts_actresses.txt'
    samples_per_career = 5
    min_length = 10
    min_am = 5
    training_number = 10000
    
    sample_careers, sample_labels = data_getter.get_sample_paths(data_loc, samples_per_career, min_length, min_am)
    
    training_careers = sample_careers[0:training_number]
    training_labels = sample_labels[0:training_number]
    testing_careers = sample_careers[training_number:]
    testing_labels = sample_labels[training_number:]
    
    #Null model first.
    null_score = null_model_classifier(testing_careers, testing_labels)
    print('null score: ', null_score)

    input_size = len(training_careers[0])
    #Resize the input data if needed.
    training_careers = training_careers.reshape(training_careers.shape + (1,))
    testing_careers = testing_careers.reshape(testing_careers.shape + (1,))
    print(training_careers.shape)

    #Now ML model.
    model = keras.models.Sequential()
   
    #model.add(keras.layers.Dense(32, activation='relu', input_shape=(input_size,)))
    #model.add(keras.layers.Dense(16, activation='relu'))
    #model.add(keras.layers.Dense(16, activation='relu'))
    #model.add(keras.layers.Dense(1, activation='sigmoid'))

    model.add(keras.layers.Conv1D(16, 5, activation='relu', input_shape=(None,training_careers.shape[-1])))
    model.add(keras.layers.MaxPooling1D(5))
    model.add(keras.layers.Conv1D(16, 5, activation='relu'))
    model.add(keras.layers.GlobalMaxPooling1D())
    model.add(keras.layers.Dense(16, activation='relu'))
    model.add(keras.layers.Dense(16, activation='relu'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
    history = model.fit(training_careers, training_labels, epochs=20, batch_size=256, validation_data=(testing_careers,testing_labels))

    #results_1 = model.evaluate(testing_careers, testing_labels)

    history_dict = history.history
    #print(history_dict.keys())


    #Print section. 
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    acc_values = history_dict['acc']
    vall_acc_values = history_dict['val_acc']

    epochs = range(1, len(acc_values) + 1)

    #Plot the training values and validation loss.
    plt.plot(epochs, loss_values, 'bo', label='Training loss')
    plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
    plt.title('Training value and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend

    plt.savefig('acting_class_training_and_validation_loss.png')
    plt.close()

    #Plot the training and validation accuracy.
    plt.plot(epochs, acc_values, 'bo', label='Training acc')
    plt.plot(epochs, vall_acc_values, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.savefig('acting_class_traing_and validation_acc.png')
    plt.close()

    #print('model score: ', score)


def null_model_classifier(series, labels):
    
    score = 0.0
    for x in range(len(series)):
        am_index = data_getter.get_am(list(series[x]), True)
        if am_index == len(series[x])-1:
            if labels[x] == 1:
                score += 1.0
        else:
            if labels[x] == 0:
                score += 1.0
    score /= len(series)
    
    return score
            

if __name__ == "__main__":
    main()