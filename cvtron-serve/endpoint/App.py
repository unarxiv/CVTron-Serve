#coding:utf-8
import cherrypy
from Classifier import Classifier
from Detector import Detector
from Segmentor import Segmentor
import tensorflow as tf


class App(object):
    def index(self):
        return "It Works!"

    index.exposed = True


cherrypy.tree.mount(Segmentor(), '/segmentor')
cherrypy.tree.mount(Classifier(), '/classifier')
cherrypy.tree.mount(Detector(), '/detector')
cherrypy.engine.start()
