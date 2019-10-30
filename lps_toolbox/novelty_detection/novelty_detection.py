import numpy as np
import pandas as pd

def create_threshold(quantity, 
                     interval):
        """
        creates a threshold array for the model

        quantity: iterable
            each position has the number of values for each interval in the same position
        interval: iterable
            each position has a len 2 iterable with the interval where to extract the corresponding quantity
        """
        threshold = np.array(list())
        for q, i in zip(quantity, interval):
            threshold = np.concatenate((threshold, np.linspace(i[0], i[1], q)))
        return threshold

def get_results(predictions,
                labels,
                classes_names, 
                novelty_index,
                threshold, 
                novelty_matrix = False,
                save_csv = False, 
                filepath = './'):
        """
        
        predictions: numpy array
            prediction set with events on lines ans neurons on columns
        labels: numpy array
            correct labels of the prediction set
        classes_names: list
            list which the position i has the name of the class i
        novelty_index: int
            index of the class that will be treated as novelty
        threshold: numpy array
            array with one or multiples values of threshold to be tested    
        novelty_matrix: boolean
            if true outputs a data frame with the values of the boolean matrix for novelty
        save _csv: boolean
            if true saves the data frame as a .csv file
        filepath: string
            where to save the .csv file
        """

        classf = _get_classf_matrix(predictions, classes_names, novelty_index, threshold)
        target = _get_target_matrix(labels, classes_names, novelty_index)
        if novelty_matrix: 
            novelty = _get_novelty_matrix(predictions, threshold)
            data = np.concatenate((predictions, novelty, classf, target), axis = 1)
            outer_level = ['Neurons', 'Neurons', 'Neurons', 'Target']
            inner_level = ['out0', 'out1', 'out2', 'T']
            index = ['Novelty', 'Classification']
            for i in range(2):
                current_index = index[i]
                for t in threshold:
                    outer_level.insert(-2, current_index)
                    inner_level.insert(-1, t)
        else:
            data = np.concatenate((predictions, classf, target), axis = 1)
            outer_level = ['Neurons', 'Neurons', 'Neurons', 'Target']
            inner_level = ['out0', 'out1', 'out2', 'T']
            for t in threshold:
                outer_level.insert(-1, 'Classification')
                inner_level.insert(-1, t)
        print(data.shape)
        print(len(inner_level))
        print(len(outer_level))
        novelty_frame = pd.DataFrame(data, columns = pd.MultiIndex.from_arrays([outer_level, inner_level]))
        if save_csv:
            filepath = filepath +'/threshold_from_' + str(threshold[0]) + 'to' + str(threshold[-1]) + '.csv'
            novelty_frame.to_csv(filepath, index = False)
        return novelty_frame

def read_results_csv(filepath):
    raise NotImplementedError

def _get_novelty_matrix(predictions, threshold):
        """
        returns a boolean array with the value of novelty detection for the event in line i in predictions[i]
        with a threshold
        predictions: numpy array
            prediction set
        threshold: numpy array
            array with the threshold values
        """

        novelty_matrix = np.empty(shape = (predictions.shape[0], threshold.shape[0]), dtype = bool)
        for i,t in enumerate(threshold):
            novelty_matrix[:, i] = (predictions < t).all(axis=1)
        
        return novelty_matrix
            
def _get_classf_matrix(predictions, 
                       classes_names,
                       novelty_index, 
                       threshold):
        """
        """

        novelty = classes_names[novelty_index]
        classes_names.pop(novelty_index)
        novelty_matrix = _get_novelty_matrix(predictions, threshold)
        stack = list()
        results = list()
        for i in np.argmax(predictions, axis = 1):
            results.append(classes_names[int(i)])
        classes_names.insert(novelty_index, novelty)
        for i in range(threshold.shape[0]):
            stack.append(results)
        classf_matrix = np.where(novelty_matrix, 'Nov', np.column_stack(tuple(stack)))
        return classf_matrix

        
    
def _get_target_matrix(labels, 
                       classes_names, 
                       novelty_index):
        novelty = classes_names[novelty_index]
        classes_names[novelty_index] = 'Nov'
        target = list()
        for l in labels:
            target.append(classes_names[l])
        classes_names[novelty_index] = novelty
        return np.array(target).reshape((-1, 1))
