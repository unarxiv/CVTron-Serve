import os
import sys
import json
import uuid
import shutil
import cherrypy
import traceback
from cvtron.modeling.detector import api
from cvtron.utils.logger.Logger import logger
from cvtron.data_zoo.compress_util import ArchiveFile
from cvtron.data_zoo.compress_util import ToArchiveFolder
from cvtron.trainers.detector.object_detection_trainer import ObjectDetectionTrainer

from .cors import cors
from .config import BASE_FILE_PATH
from .config import STATIC_FILE_PATH

cherrypy.tools.cors = cherrypy._cptools.HandlerTool(cors)


class Detector(object):
    def __init__(self, folder_name=None):
        self.BASE_FILE_PATH = BASE_FILE_PATH
        self.id = str(uuid.uuid4()).split('-')[0]
        if not folder_name:
            self.folder_name = 'img_d_' + self.id
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
    def upload(self, ufile):
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
        uncompress_path = upload_path
        ## Unzip
        af = ArchiveFile(upload_file)
        ### Delete Origin File to save disk space
        af.unzip(uncompress_path, deleteOrigin=True)
        modelFile = '/home/sfermi/Documents/Programming/model/ssd_inception_v2_coco_11_06_2017.zip'
        modelZip = ArchiveFile(modelFile)
        modelZip.unzip(upload_path, deleteOrigin=False)

        train_config = {
            'pipeline_config_file': os.path.join(upload_path, 'pipeline.config'),
            'weblog_dir': os.path.join(STATIC_FILE_PATH, self.id),
            'log_every_n_steps':1,
            'train_dir': upload_path,
            'fine_tune_ckpt': os.path.join(upload_path, 'model.ckpt'),
            'data_dir': upload_path
        }
        self.trainer = ObjectDetectionTrainer(train_config, upload_path)
        self.trainer.parse_dataset(os.path.join(upload_path, 'annotations.json'))
        result = {'result': 'success', 'file_id': self.id}
        return json.dumps(result)

    @cherrypy.config(**{'tools.cors.on': True})
    @cherrypy.expose
    def get_train_config(self):
        config = {
            'num_steps':200000
        }
        return json.dumps(config)

    @cherrypy.config(**{'tools.cors.on': True})
    @cherrypy.expose
    def start_train(self):
        cl = cherrypy.request.headers['Content-Length']
        rawbody = cherrypy.request.body.read(int(cl))
        config = json.loads(rawbody.decode('utf-8'))
        try:
            override_config = config['config']
            request_id = config['file']
            request_folder_name = 'img_d_' + request_id
            train_path = os.path.join(self.BASE_FILE_PATH, request_folder_name)
            train_config = {
                'pipeline_config_file': os.path.join(train_path, 'pipeline.config'),
                'weblog_dir': os.path.join(STATIC_FILE_PATH, self.id),
                'log_every_n_steps':1,
                'train_dir': train_path,
                'fine_tune_ckpt': os.path.join(train_path, 'model.ckpt'),
                'data_dir': train_path
            }
            self.trainer = ObjectDetectionTrainer(train_config, train_path)
            self.trainer.set_annotation(os.path.join(train_path, 'annotations.json'))
            self.trainer.override_train_configs(override_config)
            self.trainer.start()
            result = {'config': config, 'log_file_name': 'log.json'}
            return json.dumps(result)
        except Exception:
            traceback.print_exc(file=sys.stdout)

    @cherrypy.config(**{'tools.cors.on': True})
    @cherrypy.expose
    def get_model(self, model_id):
        model_id = cherrypy.request.params.get('model_id')
        request_folder_name = 'img_d_' + model_id
        train_path = os.path.join(self.BASE_FILE_PATH, request_folder_name)
        if not os.path.exists(train_path):
            raise cherrypy.HTTPError(404)
        taf = ToArchiveFolder(train_path)
        compressedFile = os.path.join(STATIC_FILE_PATH, model_id)
        if not os.path.exists(compressedFile):
            os.makedirs(compressedFile)
        compressedFile = os.path.join(compressedFile, model_id + '.zip')
        print(compressedFile)
        taf.zip(compressedFile)
        result = {
            'code': '200',
            'url': '/static/' + model_id + '/' + model_id + '.zip'
        }
        return json.dumps(result)