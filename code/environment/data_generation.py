import numpy as np
import csv

def generateRequestExponential():
    _lambda = 30
    _arrival_time = 0
    avg_data_size = 160000000
    num_day = 360
    bound_data_size = int(10 * 8 * 1e6)
    complexity = 20000
    file_name = f"../../data/task_data/data_{_lambda}_{avg_data_size}_{num_day}_{bound_data_size}_{complexity}_ver_1.csv"
    with open(file_name, "w") as f:
        f.write(f'"lambda": {_lambda}; "num_day": {num_day}; "avg_data_size": {avg_data_size}; "bound_data_size": {bound_data_size}; "complexity": {complexity}\n')       
    with open(file_name, "a") as f:
        f.write("type,arrival_time,data_size,complexity\n")

    while _arrival_time < num_day * 24:
        # Plug it into the inverse of the CDF of Exponential(_lamnbda)
        _inter_arrival_time = np.random.exponential(1 / _lambda, size=1)[0]
        # Add the inter-arrival time to the running sum
        _arrival_time = _arrival_time + _inter_arrival_time
        datasize = np.random.uniform(avg_data_size - bound_data_size, avg_data_size + bound_data_size)
        # self.event_queue.push(Event(0, _arrival_time, datasize))
        with open(file_name, "a") as f:
            f.write(f"{0},{_arrival_time},{datasize},{complexity}\n")
            # f.write([0, _arrival_time, datasize])
        
generateRequestExponential()