import os
import json
import uuid
import multiprocessing as mul
from multiprocessing import Process
from cvtron.utils.logger.Logger import logger

class TrainTask(object):
    def __init__(self, trainer, logFile, modelFile, trainType):
        self.trainer = trainer
        self.type = trainType
        self.taskId = uuid.uuid4().hex
        self.logFile =  logFile
        self.modelFile = modelFile
        self.pid = None
        self.status = 'Ready'

    def getPid(self):
        return self.pid

    def getId(self):
        return self.taskId

    def start(self):
        if self.status == 'Ready':
            pid = os.fork()
            if pid == 0:
                self.trainer.train()
            else:
                self.pid = pid
                self.status = 'Running'
        else:
            logger.warn('Cannot Start for Bad Status')

    def load(self, modelFile, logFile, pid, taskId, taskType, status):
        self.modelFile = modelFile
        self.pid = pid
        self.status = status 
        self.logFile = logFile
        self.id = taskId
        self.type = taskType

    def pause(self):
        pass

    def toDict(self):
        return {
            'id': self.taskId,
            'pid': self.pid,
            'logFile': self.logFile,
            'modelFile': self.modelFile,
            'status': self.status,
            'type': self.type
        }

class TrainTasks(object):
    def __init__(self):
        self.tasks = []
    
    def add(self, task):
        self.tasks.append(task)
    
    def load(self, filename):
        self.tasks = []
        with open(filename, 'r') as f:
            fromTasks = json.load(f)
            for each in fromTasks:
                print(each)
                t = TrainTask(None, each['logFile'], each['modelFile'], each['type'])
                t.load(each['modelFile'], each['logFile'], each['pid'], each['id'], each['type'], each['status'])
                self.tasks.append(t)

    def save(self, filename):
        logger.info('saving')
        res = []
        for each in self.tasks:
            res.append(each.toDict())
        with open(filename, 'w') as f:
            json.dump(res, f)

    def get(self, taskId):
        existed = False
        for each in self.tasks:
            if each.getId() == taskId:
                existed = True
                return each
        if not existed:
            logger.warn('Task not Found')
            return None

    def toDict(self):
        return list(map(lambda x:x.toDict(), self.tasks))