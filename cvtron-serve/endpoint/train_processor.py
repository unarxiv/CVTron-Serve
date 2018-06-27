from multiprocessing import Process
import os
import json
from cvtron.utils.logger.Logger import logger

class TrainTask(object):
    def __init__(self, trainer, logFile, modelFile, path):
        self.trainer = trainer
        self.logFile = os.path.join(path, logFile)
        self.modelFile = os.path.join(path, modelFile)
        self.pid = None
        self.status = 'Ready'
        logger.info("A new train task is " + self.status)

    def getPid(self):
        return self.pid

    def start(self):
        p = Process(target=self.trainer.train())
        p.start()
        self.pid = p.pid
        self.status = 'Running'

    def pause(self):
        pass

    def toDict(self):
        return {
            'pid': self.pid,
            'logFile': self.logFile,
            'modelFile': self.modelFile,
            'status': self.status
        }

class TrainTasks(object):
    def __init__(self):
        self.tasks = []
    
    def add(self, task):
        self.tasks.append(task.toDict())
    
    def load(self, filename):
        with open(filename, 'r') as f:
            self.tasks = json.load(f)

    def save(self, filename):
        print(self.tasks)
        with open(filename, 'w') as f:
            json.dump(self.tasks, f)


    def get(self, pid):
        existed = False
        for each in self.tasks:
            if each.getPid() == pid:
                existed = True
                return each
        if not existed:
            return None
