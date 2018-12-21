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

from .config import MODEL_PATH
from .config import BASE_FILE_PATH
from .config import STATIC_FILE_PATH
from .cors import cors
sys.path.append('..')
from utils import inform

cherrypy.tools.cors = cherrypy._cptools.HandlerTool(cors)

_MODEL_MAP = {
    'alexnet': '',
    'inception_v1': 'inception_v1_2016_08_28.zip',
    'inception_v2': 'inception_v2_2016_08_28.zip',
    'inception_v3': 'inception_v3_2016_08_28.zip',
    'inception_v4': 'inception_v4_2016_09_09.zip',
    'inception_resnet_v2': 'inception_resnet_v2_2016_08_30.zip',
    'resnet_v1_50': 'resnet_v1_50_2016_08_28.zip',
    'resnet_v1_101': 'resnet_v1_101_2016_08_28.zip',
    'resnet_v1_152': 'resnet_v1_152_2016_08_28.zip',
    'resnet_v2_50': 'resnet_v2_50_2017_04_14.zip',
    'resnet_v2_101': 'resnet_v2_101_2017_04_14.zip',
    'resnet_v2_152': 'resnet_v2_152_2017_04_14.zip',    
    'vgg_16': 'vgg_16_2016_08_28.zip',
    'vgg_19': 'vgg_19_2016_08_28.zip',  
    'mobilenet_v1': 'mobilenet_v1_1.0_224.zip',
    'mobilenet_v2': 'mobilenet_v2_1.0_224.zip',      
}

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
        base_path = os.path.join(self.BASE_FILE_PATH, model_id)
        if not os.path.exists(base_path):
            raise cherrypy.HTTPError(404)
        model_name = self.modelId
        model_path = base_path
        self.classifier.init(model_name, base_path)
        self.ready_to_infer = True

    @cherrypy.tools.json_out()
    @cherrypy.config(**{'tools.cors.on': True})
    @cherrypy.expose
    def classify(self, ufile, model_name):
        model_id = cherrypy.request.params.get('model_name')
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
        #     self._init_inference(model_id, upload_path)
        model_name = self.modelId
        model_path = os.path.join(self.BASE_FILE_PATH, model_id)
        self.classifier = SlimClassifier()
        results =  self.classifier.classify(upload_file, model_name, base_path)
        print(results)
        response = {
            'results': results
        }
        return results        

    @cherrypy.config(**{'tools.cors.on': True})
    @cherrypy.expose
    def upload(self, ufile, modelId):
        self.modelId = modelId
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

        modelFile = os.path.join(os.path.join(MODEL_PATH, '.cvtron/model_zoo'), _MODEL_MAP[self.modelId])
        modelZip = ArchiveFile(modelFile)
        modelZip.unzip(upload_path, deleteOrigin=False)

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
            request_folder_name = request_id
            train_path = os.path.join(self.BASE_FILE_PATH, request_folder_name)
            configs = {
                'fine_tune_ckpt': os.path.join(train_path, 'pre-trained/model.ckpt'),
                'data_dir': train_path,
                'weblog_dir': os.path.join(STATIC_FILE_PATH, request_id),
                'train_dir': train_path,
            }
            configs['pre-trained_model'] = self.modelId
            configs.update(config['config'])
            self.trainer = SlimClassifierTrainer(configs, train_path)
            self.trainer.set_dataset_info(os.path.join(train_path, 'annotations.json'))
            emailAddr = 'jia.wu@szu.edu.cn'
            _thread.start_new_thread(self.trainer.start, (inform, (request_id, emailAddr)))
            result = {'config': config, 'log_file_name': request_id + '/log.json', 'taskId': request_id}
            return json.dumps(result)
        except Exception:
            traceback.print_exc(file=sys.stdout)

    @cherrypy.config(**{'tools.cors.on': True})
    @cherrypy.expose
    def get_model(self, model_id):
        always_refresh = True
        model_id = cherrypy.request.params.get('model_id')
        request_folder_name = model_id
        train_path = os.path.join(self.BASE_FILE_PATH, request_folder_name)
        if not os.path.exists(train_path):
            raise cherrypy.HTTPError(404)
        taf = ToArchiveFolder(train_path)
        compressedFile = os.path.join(STATIC_FILE_PATH, model_id)
        if not os.path.exists(compressedFile):
            os.makedirs(compressedFile)
        compressedFile = os.path.join(compressedFile, model_id + '.zip')
        if os.path.exists(compressedFile) and not always_refresh:
            logger.info('model exists,skipping')
            result = {
                'code': '200',
                'url': 'http://134.175.1.246:80/static/' + model_id + '/' + model_id + '.zip'
            }
            return json.dumps(result)
        else:
            taf.zip(compressedFile)
            result = {
                'code': '200',
                'url': 'http://134.175.1.246:80/static/' + model_id + '/' + model_id + '.zip'
            }
            return json.dumps(result)            
