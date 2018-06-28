#coding:utf-8

import json

import cherrypy

from .cors import cors
from .machine_reporter import Machine
from .config import isOccupied
from .train_processor import TrainTasks
cherrypy.tools.cors = cherrypy._cptools.HandlerTool(cors)


class Resource(object):
    def __init__(self):
        pass

    @cherrypy.config(**{'tools.cors.on': True})
    @cherrypy.expose
    def device(self):
        m = Machine()
        out = {'result': m.get_all()}
        return json.dumps(out)
    @cherrypy.config(**{'tools.cors.on': True})
    @cherrypy.expose
    def status(self):
        if isOccupied:
            return 'busy'
        else:
            return 'ok'
            
    @cherrypy.config(**{'tools.cors.on': True})
    @cherrypy.expose
    def tasks(self):
        tts = TrainTasks()
        tts.load('static/tts.json')
        return json.dumps(tts.toDict())
class Static(object):
    def __init__(self):
        # Use as Static Folder Endpoint
        pass
