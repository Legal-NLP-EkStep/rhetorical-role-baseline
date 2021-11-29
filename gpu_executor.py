from queue import Queue
from threading import Thread
import time

class GPUWorker(Thread):
    def __init__(self, gpu, queue):
        Thread.__init__(self)
        self.gpu = gpu
        self.queue = queue
        self.setName(f'GPU {self.gpu} worker')

    def run(self):
        print(f'Started thread: {self.getName()}')
        while True:
            func = self.queue.get()
            if func is None:
                break
            func(self.gpu)
            self.queue.task_done()

        print(f'Shutdown thread: {self.getName()}')


class GPUExecutor:

    def __init__(self, gpus):
        self.gpus = gpus
        self.queue = Queue()
        self.workers = []
        for gpu in self.gpus:
            worker = GPUWorker(gpu=gpu, queue=self.queue)
            self.workers.append(worker)
            worker.start()


    def submit(self, func):
        """func: lambda getting gpu-num as argument. """
        self.queue.put(func)


    def shutdown(self):
        print(f'Shutdown...')
        self.queue.join()

        for _ in self.workers:
            self.queue.put(None)

        for w in self.workers:
            print(f'Joining thread {w.getName()}...')
            w.join()
        print(f'Shutdown finished!')


