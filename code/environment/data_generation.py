import numpy as np
import os

def generateRequestExponential(_lambda, _arrival_time, avg_data_size, num_day, bound_data_size, complexity,
                               version):
    folder_name = f"../../data/task_data/data_{_lambda}_{avg_data_size}_{num_day}_{bound_data_size}_{complexity}_ver_{version}"
    isExist = os.path.exists(folder_name)
    if not isExist:
        os.makedirs(folder_name) 
        print("The new directory is created!")
        print(folder_name)
    train_file_name = f"{folder_name}/train.csv"
    test_file_name = f"{folder_name}/test.csv"
    def generateData(file_name):
        timepoint = _arrival_time
        with open(file_name, "w") as f:
            f.write(f'"lambda": {_lambda}; "num_day": {num_day}; "avg_data_size": {avg_data_size}; "bound_data_size": {bound_data_size}; "complexity": {complexity}\n')       
        with open(file_name, "a") as f:
            f.write("type,arrival_time,data_size,complexity\n")
    
        while timepoint < num_day * 24:
            # Plug it into the inverse of the CDF of Exponential(_lamnbda)
            _inter_arrival_time = np.random.exponential(1 / _lambda, size=1)[0]
            # Add the inter-arrival time to the running sum
            timepoint = timepoint + _inter_arrival_time
            datasize = np.random.uniform(avg_data_size - bound_data_size, avg_data_size + bound_data_size)
            # self.event_queue.push(Event(0, _arrival_time, datasize))
            with open(file_name, "a") as f:
                f.write(f"{0},{timepoint},{datasize},{complexity}\n")
                # f.write([0, _arrival_time, datasize])
    generateData(train_file_name)
    generateData(test_file_name)
        
generateRequestExponential(_lambda = 30, _arrival_time = 0, avg_data_size = 160000000, 
                           num_day = 366, bound_data_size = int(10 * 8 * 1e6), complexity = 20000,
                           version = 1)