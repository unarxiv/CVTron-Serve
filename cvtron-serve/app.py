# coding:utf-8

import os
import cherrypy
import uuid
from cvtron.modeling.classifier import api
from cvtron.utils.reporter import print_prob
from utils.machine_reporter import Machine
from config import BASE_FILE_PATH
import json

def cors():
  if cherrypy.request.method == 'OPTIONS':
    # preflign request 
    # see http://www.w3.org/TR/cors/#cross-origin-request-with-preflight-0
    cherrypy.response.headers['Access-Control-Allow-Methods'] = 'POST'
    cherrypy.response.headers['Access-Control-Allow-Headers'] = 'content-type'
    cherrypy.response.headers['Access-Control-Allow-Origin']  = '*'
    # tell CherryPy no avoid normal handler
    return True
  else:
    cherrypy.response.headers['Access-Control-Allow-Origin'] = '*'

CHERRY_CONFIG = {
    'global': {
        'server.socket_host': '0.0.0.0',
        'server.socket_port': 9090,
        'server.thread_pool': 8,
        'server.max_request_body_size': 0,
        'server.socket_timeout': 60,
    }
}

def process_result(result):
    '''
    Result is a tuple looks like [(name,prob),(name,prob),...]
    '''
    json_result = []
    for each in result:
        result_dic = {
            'type': each[0],
            'prob': str(each[1])
        }
        json_result.append(result_dic)
    return json_result

cherrypy.tools.cors = cherrypy._cptools.HandlerTool(cors)

class App(object):
    def __init__(self):
        self.folder_name = 'img_'+str(uuid.uuid4()).split('-')[0]
        self.classifier = api.get_classifier()

    @cherrypy.config(**{'tools.cors.on': True})
    @cherrypy.expose(['classify'])
    def classify(self, ufile):
        upload_path = os.path.join(BASE_FILE_PATH, self.folder_name)
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
        # Now classify the input image
        topn = print_prob(self.classifier.classify(upload_file), 5)
        out = {
            'result': process_result(topn)
        }
        return json.dumps(out)

    @cherrypy.config(**{'tools.cors.on': True})
    @cherrypy.expose(['device'])
    def device(self):
        m = Machine()
        out = {
            'result': m.get_all()
        }
        return json.dumps(out)


if __name__ == '__main__':
    cherrypy.quickstart(App(), '/', CHERRY_CONFIG)
