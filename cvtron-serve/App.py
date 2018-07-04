# coding:utf-8
import os
import cherrypy
from endpoint.scheduler import start_scheduler
from endpoint.scheduler import start_monitor
from endpoint.Classifier import Classifier
from endpoint.Detector import Detector
from endpoint.Segmentor import Segmentor
from endpoint.Resource import Resource
from endpoint.Resource import Static
from endpoint.cors import cors
import tornado.wsgi
from Monitor import FlowerServer


def main():
    fs = FlowerServer()
    cherrypy.tools.cors = cherrypy._cptools.HandlerTool(cors)
    cherrypy.config.update({
        'server.socket_host': '0.0.0.0',
        'server.socket_port': 9090,
    })
    cherrypy.tree.graft(fs.getWSGIServer(), '/monitor')
    cherrypy.tree.mount(Segmentor(), '/segmentor')
    cherrypy.tree.mount(Classifier(), '/classifier')
    cherrypy.tree.mount(Detector(), '/detector')
    cherrypy.tree.mount(Resource(), '/resource')
    cherrypy.tree.mount(
        Static(),
        '/static',
        config={
            '/': {
                'tools.cors.on':
                True,
                'tools.staticdir.on':
                True,
                'tools.staticdir.dir':
                os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), 'static'),
                'tools.staticdir.index':
                'index.html',
            }
        })
    cherrypy.engine.start()
    cherrypy.engine.block()


if __name__ == "__main__":
    main()
