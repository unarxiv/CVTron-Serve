# coding:utf-8
import os
import sys
import json
import uuid
import _thread
import traceback

import cherrypy
from cvtron.modeling.classifier import api
from cvtron.data_zoo.compress_util import ArchiveFile
from cvtron.utils.reporter import print_prob
from cvtron.trainers.classifier.slim_classifier_trainer import SlimClassifierTrainer

from .config import BASE_FILE_PATH
from .config import STATIC_FILE_PATH
from .cors import cors

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

    @cherrypy.config(**{'tools.cors.on': True})
    @cherrypy.expose
    def upload(self, ufile):
        fid = uuid.uuid4().hex
        upload_path = os.path.join(self.BASE_FILE_PATH, fid)
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
        uncompress_path = upload_path
        af = ArchiveFile(upload_file)
        af.unzip(uncompress_path, deleteOrigin=False)
        configs = {
            'fine_tune_ckpt': os.path.join(upload_path, 'pretrained/model.ckpt'),
            'data_dir': upload_path,
            'weblog_dir': os.path.join(STATIC_FILE_PATH, fid),
            'train_dir': upload_path,
        }
        configs['pre-trained_model'] = 'inception_v1'
        self.trainer = SlimClassifierTrainer(configs, upload_path)
        self.trainer.parse_dataset(os.path.join(upload_path, 'annotations.json'))
        result = {
            'result': 'success',
            'file_id': fid
        }
        return json.dumps(result)

    @cherrypy.config(**{'tools.cors.on': True})
    @cherrypy.expose
    def get_train_config(self):
        config = {
          'batch_size': 24,          
          'optimizer':'rmsprop',
          'log_every_n_steps': 1,
          'learning_rate': 0.0001
        }
        return json.dumps(config)
    
    @cherrypy.config(**{'tools.cors.on': True})
    @cherrypy.expose
    def start_train(self):
        cl = cherrypy.request.headers['Content-Length']
        rawbody = cherrypy.request.body.read(int(cl))
        config = json.loads(rawbody.decode('utf-8'))
        try:
            request_id = config['file']
            request_folder_name = 'img_d_' + request_id
            train_path = os.path.join(self.BASE_FILE_PATH, request_folder_name)
            configs = {
                'fine_tune_ckpt': os.path.join(train_path, 'pretrained/model.ckpt'),
                'data_dir': train_path,
                'weblog_dir': os.path.join(STATIC_FILE_PATH, request_id),
                'train_dir': train_path,
            }
            self.trainer = SlimClassifierTrainer(configs, train_path)
            _thread.start_new_thread(self.trainer.start, ())
            result = {'config': config, 'log_file_name': request_id + '/log.json', 'taskId': request_id}
            return json.dumps(result)
        except Exception:
            traceback.print_exc(file=sys.stdout)