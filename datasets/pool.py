
from Queue import Queue, Empty
from threading import Thread

####################################################
#MultiThreaded helpers for building databatches
####################################################
class Worker(Thread):
    """Thread executing tasks from a given tasks queue"""
    def __init__(self, tasks):
        Thread.__init__(self)
        self.tasks = tasks
        self.daemon = True
        self.running = True
        self.start()

    def run(self):
        while self.running:
            try:
                func, args, kargs = self.tasks.get(timeout=3)
            except Empty:
                return

            try:
                func(*args, **kargs)
            except Exception, e:
                print e
            finally:
                self.tasks.task_done()

class ThreadPool:
    """Pool of threads consuming tasks from a queue"""
    def __init__(self, num_threads):
        self.tasks = Queue(num_threads)
        self.workers = []
        for _ in range(num_threads): 
            self.workers.append(Worker(self.tasks))

    def add_task(self, func, *args, **kargs):
        """Add a task to the queue"""
        self.tasks.put((func, args, kargs))

    def wait_completion(self):
        """Wait for completion of all the tasks in the queue"""
        self.tasks.join()

####################################################
#end MultiThreaded helpers for building databatches
####################################################
