import os
import uuid
import json

import cherrypy
from cvtron.modeling.detector import api
from config import BASE_FILE_PATH
from cors import cors

cherrypy.tools.cors = cherrypy._cptools.HandlerTool(cors)

class Detector(object):
    def __init__(self, folder_name = None):
        self.BASE_FILE_PATH = BASE_FILE_PATH
        if not folder_name:
            self.folder_name = 'img_d_'+str(uuid.uuid4()).split('-')[0]
        else:
            self.folder_name = folder_name
        self.detector = api.get_detector()

    @cherrypy.config(**{'tools.cors.on': True})
    @cherrypy.expose
    def detect(self, ufile):
        if not self.detector:
            self.detector = api.get_detector()
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
        result = self.detector.detect(upload_file)
        print(result)
        return json.dumps(result)
