import os
import sys
import json
import uuid
import shutil
import decimal
import _thread
import cherrypy
import traceback
from cvtron.modeling.detector import api
from cvtron.utils.logger.Logger import logger
from cvtron.data_zoo.compress_util import ArchiveFile
from cvtron.data_zoo.compress_util import ToArchiveFolder
from cvtron.modeling.detector.slim_object_detector import SlimObjectDetector
from cvtron.trainers.detector.object_detection_trainer import ObjectDetectionTrainer

from .cors import cors
from .config import BASE_FILE_PATH
from .config import STATIC_FILE_PATH
sys.path.append('..')
from utils import inform

cherrypy.tools.cors = cherrypy._cptools.HandlerTool(cors)

class Detector(object):
    def __init__(self, folder_name=None):
        # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(1) 
        self.BASE_FILE_PATH = BASE_FILE_PATH
        self.id = str(uuid.uuid4()).split('-')[0]
        self.ready_to_infer = False
        if not folder_name:
            self.folder_name = 'img_d_' + self.id
        else:   
            self.folder_name = folder_name

    def _init_inference(self, model_id, test_pic_dir):
        self.sod = SlimObjectDetector()
        base_path = model_id
        base_path = os.path.join(self.BASE_FILE_PATH, base_path)
        if not os.path.exists(base_path):
            raise cherrypy.HTTPError(404)
        label_map_path = os.path.join(base_path, 'label_map.pbtxt')
        # ckpt_path = os.path.join(base_path, 'frozen_inference_graph.pb')
        exported_dir = os.path.join(test_pic_dir, 'exported')
        ckpt_path = os.path.join(exported_dir, 'frozen_inference_graph.pb')
        if os.path.exists(os.path.abspath(exported_dir)):
            shutil.rmtree(os.path.abspath(exported_dir))
        self.sod.export_latest_ckpt(base_path, exported_dir)
        self.sod.set_label_map(label_map_path)
        self.sod.init(ckpt_path)
        self.ready_to_infer = True

    @cherrypy.config(**{'tools.cors.on': True})
    @cherrypy.expose
    def get_infer_config(self):
        config = api.get_infer_config()
        return json.dumps(config)

    @cherrypy.tools.json_out()
    @cherrypy.config(**{'tools.cors.on': True})
    @cherrypy.expose
    def detect(self, ufile, model_name):
        model_name = cherrypy.request.params.get('model_name')
        # model_name = 'da67f84dc38f44d9943a9cdc56cab7f5'
        # Handler for model: model_name
        # detector = api.get_detector()
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
        if not self.ready_to_infer:
            self._init_inference(model_name, upload_path)
            # self.ready_to_infer = False
        results =  self.sod.detect(upload_file)
        response = {
            'results': results
        }
        return results

    @cherrypy.config(**{'tools.cors.on': True})
    @cherrypy.expose
    def upload(self, ufile, modelId):
        print('upload')
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
        # Unzip File and return file Id
        ## Generate file id
        ## Set Uncompress path
        uncompress_path = upload_path
        ## Unzip
        af = ArchiveFile(upload_file)
        ### Delete Origin File to save disk space
        af.unzip(uncompress_path, deleteOrigin=True)
        print(modelId)
        modelFile = '/home/ubuntu/.cvtron/model_zoo/ssd_inception_v2_coco_11_06_2017.zip'
        # modelFile = '/home/wujia/.cvtron/model_zoo/ssd_inception_v2_coco_11_06_2017.zip'
        modelZip = ArchiveFile(modelFile)
        modelZip.unzip(upload_path, deleteOrigin=False)

        # train_config = {
        #     'pipeline_config_file': os.path.join(upload_path, 'pipeline.config'),
        #     'weblog_dir': os.path.join(STATIC_FILE_PATH, fid),
        #     'log_every_n_steps':1,
        #     'train_dir': upload_path,
        #     'fine_tune_ckpt': os.path.join(upload_path, 'pretrained/model.ckpt'),
        #     'data_dir': upload_path
        # }
        # train_config['pre-trained_model'] = 'ssd_inception_v2'
        self.trainer = ObjectDetectionTrainer({}, upload_path)
        self.trainer.parse_dataset(os.path.join(upload_path, 'annotations.json'))
        result = {'result': 'success', 'file_id': fid}
        print(result)
        return json.dumps(result)

    @cherrypy.config(**{'tools.cors.on': True})
    @cherrypy.expose
    def get_train_config(self):
        config = {
            'batch_size':24,
            'learning_rate':0.001,
            'num_steps':200000,
            'log_every_n_steps':100
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
            override_config = config['config']
            request_id = config['file']
            request_folder_name = request_id
            train_path = os.path.join(self.BASE_FILE_PATH, request_folder_name)
            train_config = {
                'pipeline_config_file': os.path.join(train_path, 'pipeline.config'),
                'weblog_dir': os.path.join(STATIC_FILE_PATH, request_id),
                'log_every_n_steps':config['config']['log_every_n_steps'],
                'train_dir': train_path,
                'fine_tune_ckpt': os.path.join(train_path, 'model.ckpt'),
                'data_dir': train_path
            }
            self.trainer = ObjectDetectionTrainer(train_config, train_path)
            self.trainer.set_annotation(os.path.join(train_path, 'annotations.json'))
            self.trainer.override_pipeline_config(override_config, os.path.join(train_path, 'pipeline.config'))
            # pid = os.fork()
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
        print(model_id)
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

