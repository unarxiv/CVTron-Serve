#coding:utf-8
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
# celery_broker_url = 'sqs://AKIAJZVDLOMLODWL3B5Q:7SwEU9g34mCF8WOEw2Z8dIuKYwfpFVBt0QefZ6pj@'
