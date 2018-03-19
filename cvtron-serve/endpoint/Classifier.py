#coding:utf-8
import uuid
import cherrypy
from cvtron.modeling.classifier import api
from cors import cors
class Classifier(object):
    def __init__(self,folder_name = None):
        if not folder_name:
            self.folder_name = 'img_'+str(uuid.uuid4()).split('-')[0]
        else:
            self.folder_name = folder_name
        self.classifier = api.get_classifier()
    
    @cherrypy.config(**{'tools.cors.on':True})
    @cherrypy.expose
    