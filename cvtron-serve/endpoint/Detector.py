import json
import os
import uuid

import cherrypy
from cvtron.data_zoo.compress_util import ArchiveFile
from cvtron.modeling.detector import api

from .config import BASE_FILE_PATH
from .cors import cors

cherrypy.tools.cors = cherrypy._cptools.HandlerTool(cors)


class Detector(object):
    def __init__(self, folder_name=None):
        self.BASE_FILE_PATH = BASE_FILE_PATH
        if not folder_name:
            self.folder_name = 'img_d_' + str(uuid.uuid4()).split('-')[0]
        else:
            self.folder_name = folder_name

    @cherrypy.config(**{'tools.cors.on': True})
    @cherrypy.expose
    def get_infer_config(self):
        config = api.get_infer_config()
        return json.dumps(config)

    @cherrypy.config(**{'tools.cors.on': True})
    @cherrypy.expose
    def detect(self, ufile, model_name):
        model_name = cherrypy.request.params.get('model_name')
        print(model_name)
        # Handler for model: model_name
        detector = api.get_detector()
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
        result = [{
            "x_min": 19.675,
            "class_name": "dog",
            "y_max": 554.65,
            "x_max": 323.3521,
            "y_min": 24.7567
        }]
        return json.dumps(result)

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
                size += len(data)
        # Unzip File and return file Id
        ## Generate file id
        fid = uuid.uuid4().hex
        ## Set Uncompress path
        uncompress_path = os.path.join(self.BASE_FILE_PATH, 'uncompress')
        ## Unzip
        af = ArchiveFile(upload_file)
        ### Delete Origin File to save disk space
        af.unzip(uncompress_path, deleteOrigin=True)
        result = {'result': 'success', 'file_id': fid}
        return json.dumps(result)

    @cherrypy.config(**{'tools.cors.on': True})
    @cherrypy.expose
    def get_train_config(self):
        config = api.get_train_config()
        print(config)
        return json.dumps(config)

    @cherrypy.config(**{'tools.cors.on': True})
    @cherrypy.expose
    def start_train(self):
        cl = cherrypy.request.headers['Content-Length']
        rawbody = cherrypy.request.body.read(int(cl))
        config = json.loads(rawbody.decode('utf-8'))
        config[
            'weblog_dir'] = '/home/wujia/examples/platform/test-platform/CVTron-Serve/cvtron-serve/static/log'
        print(config)
        try:
            detector = api.get_detector(config)
            detector.train()
            result = {'config': config, 'log_file_name': 'log.json'}
            return json.dumps(result)
        except Exception:
            return 'failed'
