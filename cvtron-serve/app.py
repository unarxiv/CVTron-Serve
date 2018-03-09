#coding:utf-8

import os
import cherrypy
import uuid
from cvtron.modeling.classifier import api
from cvtron.utils.reporter import print_prob
import json

CHERRY_CONFIG = {
    'global' : {
        'server.socket_host' : '127.0.0.1',
        'server.socket_port' : 9090,
        'server.thread_pool' : 8,
        'server.max_request_body_size' : 0,
        'server.socket_timeout' : 60,
    }
}

BASE_FILE_PATH = './tmp'

def process_result(result):
    '''
    Result is a tuple looks like [(name,prob),(name,prob),...]
    '''
    json_result = []
    for each in result:
        result_dic = {
            'type':each[0],
            'prob':str(each[1])
        }
        json_result.append(result_dic)
    return json_result


class App(object):
    def __init__(self):
        self.folder_name = 'img_'+str(uuid.uuid4()).split('-')[0]
        self.classifier = api.get_classifier()
    @cherrypy.expose
    def upload(self,ufile):
        upload_path = os.path.join(BASE_FILE_PATH,self.folder_name)
        if not os.path.exists(upload_path):
            os.makedirs(upload_path)
        upload_file = os.path.join(upload_path,ufile.filename)
        size = 0
        with open(upload_file,'wb') as out:
            while True:
                data = ufile.file.read(8192)
                if not data:
                    break 
                out.write(data)
                size += len(data)
        # Now classify the input image
        topn = print_prob(self.classifier.classify(upload_file),5)
        out = {
            'result': process_result(topn)
        }
        return json.dumps(out)
    @cherrypy.expose
    def query_device(self):
        hardware = None
        out = {
            'result': hardware
        }
        return json.dumps(out)

if __name__ == '__main__':
    cherrypy.quickstart(App(), '/', CHERRY_CONFIG)