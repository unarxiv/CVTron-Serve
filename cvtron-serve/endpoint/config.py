#coding:utf-8
MODEL_PATH = '/home/ubuntu'
BASE_FILE_PATH = './tmp'
STATIC_FILE_PATH = './static'
LOADED_ENDPOINT = {
    'is_classifier_open': True,
    'is_detection_open': True,
    'is_segmentation_open': True
}

isOccupied = False

celery_broker_url = 'redis://:cvtrondemo@118.89.28.34:6379/0'
celery_backend_url = 'redis://:cvtrondemo@118.89.28.34:6379/0'