# coding:utf-8
import os
import uuid
import json
import cherrypy

from cvtron.modeling.segmentor import api
from cvtron.utils.image_loader import write_image

from .config import BASE_FILE_PATH
from .cors import cors

cherrypy.tools.cors = cherrypy._cptools.HandlerTool(cors)


class Segmentor(object):
    def __init__(self, folder_name=None):
        self.BASE_FILE_PATH = BASE_FILE_PATH
        if not folder_name:
            self.folder_name = 'img_' + str(uuid.uuid4()).split('-')[0]
        else:
            self.folder_name = folder_name
        self.segmentor = api.get_segmentor()

    @cherrypy.config(**{'tools.cors.on': True})
    @cherrypy.expose
    def segment(self, ufile):
        if not self.segmentor:
            self.segmentor = api.get_segmentor()
        upload_path = os.path.join(self.BASE_FILE_PATH, self.folder_name)
        if not os.path.exists(upload_path):
            os.makedirs(upload_path)
        upload_file = os.path.join(upload_path, ufile.filename)
        size = 0
        with open(upload_file, 'wb') as out:
            while True:
                data = ufile.file.read(8192)
                if not data:
                    break
                out.write(data)
                size += len(data)
        pred_image = self.segmentor.segment(upload_file)
        out_filename = 'img_i_' + str(uuid.uuid4()).split('-')[0] + '.jpg'
        out_path = os.path.join('./static', 'img')
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        write_image(pred_image, os.path.join(out_path, out_filename))
        return out_filename

    @cherrypy.config(**{'tools.cors.on': True})
    @cherrypy.expose
    def get_train_config(self):
        config = api.get_defaultConfig()
        print(config)
        return json.dumps(config)

    @cherrypy.config(**{'tools.cors.on': True})
    @cherrypy.expose
    def upload_train_file(self, ufile):
        upload_path = os.path.join(self.BASE_FILE_PATH, self.folder_name)
        if not os.path.exists(upload_path):
            os.makedirs(upload_path)
        upload_file = os.path.join(upload_path, ufile.filename)
        size = 0
        with open(upload_file, 'wb') as out:
            while True:
                data = ufile.file.read(8192)
                if not data:
                    break
                out.write(data)
                size+=len(data)
        return upload_file
    
    @cherrypy.config(**{'tools.cors.on': True})
    @cherrypy.expose
    def start_train(self):
        cl = cherrypy.request.headers['Content-Length']
        rawbody = cherrypy.request.body.read(int(cl))
        config = json.loads(rawbody)
        print(config)
        '''
        try:
            dlt = api.get_segmentor_trainer(config)
            dlt.train()
            return 'started'
        except Exception:
            return 'failed'
        '''