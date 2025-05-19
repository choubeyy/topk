import numpy as np
from dfscode import DFSCode, DFSEdge
import heapq

class ProbabilisticMaxQueue:
    """
    A priority queue that orders items according to a custom priority function.
    Uses Python's built-in heapq for a min-heap.
    """
    def __init__(self):
        """
        :param priority_func: A function that, given an item, returns its priority.
        """
        self._heap = []

    def push(self, item):
        """
        Push an item onto the priority queue.
        The priority is computed via the provided priority function.
        """
        priority = self.priority_function(item)
        heapq.heappush(self._heap, (priority, item))

    def pop(self):
        """
        Pop the highest-priority (lowest priority value) item off the queue.
        If the queue is empty, raises IndexError.
        """
        if not self._heap:
            raise IndexError("Pop from an empty priority queue")
        
        priority, item = heapq.heappop(self._heap)
        return item

    def peek(self):
        """
        Return the highest-priority item without removing it from the queue.
        If the queue is empty, returns None.
        """
        if not self._heap:
            return None
        
        # Look at the first element (lowest priority value).
        priority, item = self._heap[0]
        return item
    
    def get_mu_sigma_gaussian(self, a, b):
        mu = (a + b) / 2
        sigma = (mu - a) / 3
        return mu, sigma


    def kl_gaussian(self, mu_p, sigma_p, mu_q, sigma_q):
        return np.log(sigma_q / sigma_p) + (sigma_p**2 + (mu_p - mu_q)**2) / (2 * sigma_q**2) - 0.5

    def priority_function(self, item):
        if item[0] == item[1]:
            return 1 - item[0]
        mu_o, sigma_o = self.get_mu_sigma_gaussian(0.99, 1.01)
        mu_p, sigma_p = self.get_mu_sigma_gaussian(item[0], item[1])
        return self.kl_gaussian(mu_p, sigma_p, mu_o, sigma_o)

    def is_empty(self):
        """Check whether the priority queue is empty."""
        return len(self._heap) == 0

    def __len__(self):
        """Return the number of items in the priority queue."""
        return len(self._heap)
    
    def __repr__(self):
        return str(self._heap)


class ProbabilisticMinQueue:
    """
    A priority queue that orders items according to a custom priority function.
    Uses Python's built-in heapq for a min-heap.
    """
    def __init__(self):
        """
        :param priority_func: A function that, given an item, returns its priority.
        """
        self._heap = []

    def push(self, item):
        """
        Push an item onto the priority queue.
        The priority is computed via the provided priority function.
        """
        priority = self.priority_function(item)
        heapq.heappush(self._heap, (priority, item))

    def pop(self):
        """
        Pop the highest-priority (lowest priority value) item off the queue.
        If the queue is empty, raises IndexError.
        """
        if not self._heap:
            raise IndexError("Pop from an empty priority queue")
        
        priority, item = heapq.heappop(self._heap)
        return item

    def peek(self):
        """
        Return the highest-priority item without removing it from the queue.
        If the queue is empty, returns None.
        """
        if not self._heap:
            return None
        
        # Look at the first element (lowest priority value).
        priority, item = self._heap[0]
        return item
    
    def get_mu_sigma_gaussian(self, a, b):
        mu = (a + b) / 2
        sigma = (mu - a) / 3
        return mu, sigma

    def kl_gaussian(self, mu_p, sigma_p, mu_q, sigma_q):
        return np.log(sigma_q / sigma_p) + (sigma_p**2 + (mu_p - mu_q)**2) / (2 * sigma_q**2) - 0.5

    def priority_function(self, item):
        if item[0] == item[1]:
            return -(1 - item[0])
        mu_o, sigma_o = self.get_mu_sigma_gaussian(0.99, 1.01)
        mu_p, sigma_p = self.get_mu_sigma_gaussian(item[0], item[1])
        return -self.kl_gaussian(mu_p, sigma_p, mu_o, sigma_o)

    def is_empty(self):
        """Check whether the priority queue is empty."""
        return len(self._heap) == 0

    def __len__(self):
        """Return the number of items in the priority queue."""
        return len(self._heap)
    
    def __repr__(self):
        return str(self._heap)


if __name__=='__main__':

    codeA = DFSCode()
    codeA.add(DFSEdge(0, 1, 1, 1, 1))
    codeA.add(DFSEdge(1, 2, 2, 1, 1))
    codeB = DFSCode()
    codeB.add(DFSEdge(1, 2, 1, 2, 2))

    print("ProbabilisticMaxQueue")
    pq = ProbabilisticMaxQueue()
    pq.push((0.6, 0.7, codeA))
    pq.push((0.45, 0.55, codeB))
    
    print(pq.pop())
    print(pq.pop())
    print(pq.is_empty())

    print("ProbabilisticMinQueue")
    pq = ProbabilisticMinQueue()
    pq.push((0.6, 0.7, codeA))
    pq.push((0.45, 0.55, codeA))
    pq.push((0.5, 0.6, codeB))
    pq.push((0.9, 0.95, codeB))
    
    print(pq.pop())
    print(pq.pop())
    print(pq.is_empty())
    
