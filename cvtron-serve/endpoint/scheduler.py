import os

from celery import Celery
from cvtron.modeling.segmentor import api

from .config import celery_backend_url, celery_broker_url

scheduler = Celery(
    'cvtron', broker=celery_broker_url, backend=celery_backend_url)


@scheduler.task
def start_train_task(name, config):
    if name == 'segmentor':
        dlt = api.get_segmentor_trainer(config)
        return dlt.train()
    else:
        pass


def start_scheduler(isDaemon=False):
    pid = os.fork()
    if pid == 0:
        if not isDaemon:
            os.system('celery -A endpoint.scheduler worker --loglevel=debug')
        else:
            print('Daemon mode is not supported yet')
    else:
        pass


def start_monitor(isDaemon=False):
    if not isDaemon:
        os.system(
            'celery flower -A endpoint.scheduler --address=127.0.0.1 --port=5555 --broker='
            + celery_broker_url)
    else:
        print('Daemon mode is not supported yet')
