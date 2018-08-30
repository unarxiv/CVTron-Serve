# coding:utf-8
import json
import os
import sys
import uuid
import _thread
import cherrypy
import traceback
from cvtron.data_zoo.compress_util import ArchiveFile
from cvtron.trainers.segmentor.deeplab_trainer2 import DeepLabTrainer
from cvtron.utils.image_loader import write_image

from .cors import cors
from .config import MODEL_PATH
from .config import BASE_FILE_PATH
from .config import STATIC_FILE_PATH
sys.path.append('..')
from utils import inform

cherrypy.tools.cors = cherrypy._cptools.HandlerTool(cors)

DEEP_LAB = ['deeplab_mobilenet_v2', 'deeplab_xception_65']
MASK_R_CNN = ['mask_rcnn_resnet50', 'mask_rcnn_resnet101']
_MODEL_MAP = {
    'deeplab_mobilenet_v2': 'deeplabv3_mnv2_pascal_train_aug_2018_01_29.zip',
    'deeplab_xception_65': 'deeplabv3_pascal_train_aug_2018_01_04.zip',
    'mask_rcnn_resnet50': 'mask_rcnn_resnet50_atrous_coco_2018_01_28.zip',
    'mask_rcnn_resnet101': 'mask_rcnn_resnet101_atrous_coco_2018_01_28.zip'
}

class Segmentor(object):
    def __init__(self, folder_name=None):
        self.BASE_FILE_PATH = BASE_FILE_PATH
        if not folder_name:
            self.folder_name = 'img_' + uuid.uuid4().hex
        else:
            self.folder_name = folder_name

    @cherrypy.config(**{'tools.cors.on': True})
    @cherrypy.expose
    def segment(self, ufile):
        config.isOccupied = True
        if not self.segmentor:
            self.segmentor = api.get_segmentor()
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
        pred_image = self.segmentor.segment(upload_file)
        out_filename = 'img_i_' + str(uuid.uuid4()).split('-')[0] + '.jpg'
        out_path = os.path.join('./static', 'img')
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        write_image(pred_image, os.path.join(out_path, out_filename))
        return out_filename

    def get_trainer(self, modelId, path, config=None):
        if modelId in DEEP_LAB:
            return DeepLabTrainer(config, path)
        if modelId in MASK_R_CNN:
            return None

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
        # Unzip File and return file Id
        ## Generate file id
        ## Set Uncompress path
        uncompress_path = upload_path
        ## Unzip
        af = ArchiveFile(upload_file)          
        af.unzip(uncompress_path, deleteOrigin=True)
        modelFile = os.path.join(os.path.join(MODEL_PATH, '.cvtron/model_zoo'), _MODEL_MAP[self.modelId])
        modelZip = ArchiveFile(modelFile)
        modelZip.unzip(upload_path, deleteOrigin=False)

        self.trainer = self.get_trainer(self.modelId, upload_path)
        if not self.trainer:
            result = {'result': 'fail', 'file_id': fid}
            return json.dumps(result) 
        self.trainer.parse_dataset(os.path.join(upload_path, 'annotations.json'))
        result = {'result': 'success', 'file_id': fid}
        return json.dumps(result)        

    @cherrypy.config(**{'tools.cors.on': True})
    @cherrypy.expose
    def get_train_config(self):
        config = {
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
            train_path = os.path.join(self.BASE_FILE_PATH, request_id)
            train_config = {
                'fine_tune_ckpt': os.path.join(train_path, 'pre-trained/model.ckpt'),
                'data_dir': train_path,
                'weblog_dir': os.path.join(STATIC_FILE_PATH, request_id),
                'train_dir': train_path
            }
            self.trainer = self.get_trainer(self.modelId, train_path, train_config)
            if not self.trainer:
                return json.dumps({'result': 'fail'})
            self.trainer.set_dataset_info(os.path.join(train_path, 'annotations.json'))
            emailAddr = 'jia.wu@szu.edu.cn'                     
            _thread.start_new_thread(self.trainer.start, (inform, (request_id, emailAddr)))
            result = {'config': config, 'log_file_name': request_id + '/log.json', 'taskId': request_id}
            return json.dumps(result)
        except Exception as e:
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
