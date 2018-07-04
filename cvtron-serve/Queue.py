import os

os.system('celery -A endpoint.scheduler worker --loglevel=debug')
