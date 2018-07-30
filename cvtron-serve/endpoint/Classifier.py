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
from cvtron.modeling.classifier.slim_classifier import SlimClassifier

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
        self.ready_to_infer = False

    def _init_inference(self, model_id, test_pic_dir):
        self.classifier = SlimClassifier()
        base_path = model_id
        base_path = os.path.join(self.BASE_FILE_PATH, base_path)
        if not os.path.exists(base_path):
            raise cherrypy.HTTPError(404)
        model_name = model_id
        model_path = base_path
        self.classifier.init(model_name, base_path)
        self.ready_to_infer = True

    @cherrypy.tools.json_out()
    @cherrypy.config(**{'tools.cors.on': True})
    @cherrypy.expose
    def classify(self, ufile):
        model_name = cherrypy.request.params.get('model_name')
        # Handler for model: model_name
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
        # if not self.ready_to_infer:
        #     self._init_inference(model_name, upload_path)
        base_path = os.path.join(self.BASE_FILE_PATH, 'img_d_1234567890')
        model_name = 'inception_v1'
        self.classifier = SlimClassifier()
        results =  self.classifier.classify(upload_file, model_name, base_path)
        print(results)
        response = {
            'results': results
        }
        return results        

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
        upload_path = os.path.join(self.BASE_FILE_PATH, 'img_d_1234567890') 
        upload_file = os.path.join(upload_path, 'flowers.zip')
        uncompress_path = upload_path
        af = ArchiveFile(upload_file)
        af.unzip(uncompress_path, deleteOrigin=False)
        modelFile = '/home/ubuntu/.cvtron/model_zoo/inception_v1.zip'
        modelZip = ArchiveFile(modelFile)
        modelZip.unzip(upload_path, deleteOrigin=False)

        # configs = {
        #     'fine_tune_ckpt': os.path.join(upload_path, 'model.ckpt'),
        #     'data_dir': upload_path,
        #     'weblog_dir': os.path.join(STATIC_FILE_PATH, fid),
        #     'train_dir': upload_path,
        # }
        # configs['pre-trained_model'] = 'inception_v1'
        # upload_path = os.path.join(self.BASE_FILE_PATH, 'img_d_1234567890')
        self.trainer = SlimClassifierTrainer({}, upload_path)
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
          'log_every_n_steps': 100,
          'learning_rate': 0.0001
        }
        return json.dumps(config)
    
    @cherrypy.config(**{'tools.cors.on': True})
    @cherrypy.expose
    def start_train(self):
        cl = cherrypy.request.headers['Content-Length']
        rawbody = cherrypy.request.body.read(int(cl))
        config = json.loads(rawbody.decode('utf-8'))
        print(config)
        try:
            request_id = config['file']
            request_folder_name = 'img_d_' + request_id
            # train_path = os.path.join(self.BASE_FILE_PATH, request_folder_name)
            train_path = os.path.join(self.BASE_FILE_PATH, 'img_d_1234567890')
            configs = {
                'fine_tune_ckpt': os.path.join(train_path, 'model.ckpt'),
                'data_dir': train_path,
                'weblog_dir': os.path.join(STATIC_FILE_PATH, request_id),
                'train_dir': train_path,
            }
            configs['pre-trained_model'] = 'inception_v1'
            configs.update(config['config'])

            self.trainer = SlimClassifierTrainer(configs, train_path)
            self.trainer.set_dataset_info(os.path.join(train_path, 'annotations.json'))
            _thread.start_new_thread(self.trainer.start, ())
            result = {'config': config, 'log_file_name': request_id + '/log.json', 'taskId': request_id}
            return json.dumps(result)
        except Exception:
            traceback.print_exc(file=sys.stdout)
