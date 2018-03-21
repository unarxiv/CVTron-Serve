#coding:utf-8

import json

import cherrypy

from .cors import cors
from .machine_reporter import Machine

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


class Static(object):
    def __init__(self):
        # Use as Static Folder Endpoint
        pass
