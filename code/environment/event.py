import heapq

# class of system events
class Event:
    # 0:request arrival 1:energy arrival 2-14 task finish
    def __init__(self, type, arrival_time, extra_msg=0):
        self.name = type
        self.arrival_time = arrival_time
        self.extra_msg = extra_msg
        self.core_number = 12

    def __repr__(self):
        name = 'problem!!'
        if self.name == 0:
            name = "request arrival"
        if self.name == 1:
            name = "energy consumption pattern change"
        if 2 <= self.name < self.core_number + 2:
            name = "task finish of CPU core {}".format(self.name - 2)
        return "event: '{}', arrival time: {}".format(name, self.arrival_time)
    


# event queues record the event happening order
class EventQueue():
    def __init__(self):
        self._queue = []
        self._index = 0

    def push(self, event):
        #print(event)
        heapq.heappush(self._queue, (event.arrival_time, self._index, event))
        self._index += 1

    def pop(self):
        # return event
        return heapq.heappop(self._queue)[-1]
    
    # def print_queue(self):
    #     print(self._queue)
    #     for event in self._queue:
    #         print(event)