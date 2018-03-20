# coding:utf-8
import json
import os
import uuid

import cherrypy
from cvtron.modeling.classifier import api
from cvtron.utils.reporter import print_prob

from config import BASE_FILE_PATH
from cors import cors

cherrypy.tools.cors = cherrypy._cptools.HandlerTool(cors)


class Classifier(object):
    def __init__(self, folder_name=None):
        self.BASE_FILE_PATH = BASE_FILE_PATH
        if not folder_name:
            self.folder_name = 'img_' + str(uuid.uuid4()).split('-')[0]
        else:
            self.folder_name = folder_name
        self.classifier = api.get_classifier()

    def process_result(self, result):
        json_result = []
        for each in result:
            result_dic = {'type': each[0], 'prob': str(each[1])}
            json_result.append(result_dic)
        return json_result

    @cherrypy.config(**{'tools.cors.on': True})
    @cherrypy.expose
    def classify(self, ufile):
        if not self.classifier:
            self.classifier = api.get_classifier()
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
        topn = print_prob(self.classifier.classify(upload_file), 5)
        out = {'result': self.process_result(topn)}
        return json.dumps(out)
