import os
from pprint import pformat

import tornado.web
from tornado.options import options, parse_command_line, parse_config_file
from celery import Celery
from cvtron.utils.logger.Logger import logger
from flower.api import control, events, tasks, workers
from flower.app import Flower
from flower.options import default_options
from flower.urls import settings
from paste.exceptions.errormiddleware import ErrorMiddleware
from paste.translogger import TransLogger

import flowerconfig
from endpoint.config import celery_backend_url, celery_broker_url
from endpoint.scheduler import scheduler

handlers = [
    (r"/monitor/api/workers",
     workers.ListWorkers), (r"/monitor/api/worker/shutdown/(.+)",
                            control.WorkerShutDown),
    (r"/monitor/api/worker/pool/restart/(.+)",
     control.WorkerPoolRestart), (r"/monitor/api/worker/pool/grow/(.+)",
                                  control.WorkerPoolGrow),
    (r"/monitor/api/worker/pool/shrink/(.+)",
     control.WorkerPoolShrink), (r"/monitor/api/worker/pool/autoscale/(.+)",
                                 control.WorkerPoolAutoscale),
    (r"/monitor/api/worker/queue/add-consumer/(.+)",
     control.WorkerQueueAddConsumer),
    (r"/monitor/api/worker/queue/cancel-consumer/(.+)",
     control.WorkerQueueCancelConsumer), (r"/monitor/api/tasks",
                                          tasks.ListTasks),
    (r"/monitor/api/task/types",
     tasks.ListTaskTypes), (r"/monitor/api/queues/length",
                            tasks.GetQueueLengths),
    (r"/monitor/api/task/info/(.*)",
     tasks.TaskInfo), (r"/monitor/api/task/apply/(.+)", tasks.TaskApply),
    (r"/monitor/api/task/async-apply/(.+)",
     tasks.TaskAsyncApply), (r"/monitor/api/task/send-task/(.+)",
                             tasks.TaskSend),
    (r"/monitor/api/task/result/(.+)",
     tasks.TaskResult), (r"/monitor/api/task/abort/(.+)",
                         tasks.TaskAbort), (r"/monitor/api/task/timeout/(.+)",
                                            control.TaskTimout),
    (r"/monitor/api/task/rate-limit/(.+)",
     control.TaskRateLimit), (r"/monitor/api/task/revoke/(.+)",
                              control.TaskRevoke),
    (r"/monitor/api/task/events/task-sent/(.*)",
     events.TaskSent), (r"/monitor/api/task/events/task-received/(.*)",
                        events.TaskReceived),
    (r"/monitor/api/task/events/task-started/(.*)",
     events.TaskStarted), (r"/monitor/api/task/events/task-succeeded/(.*)",
                           events.TaskSucceeded),
    (r"/monitor/api/task/events/task-failed/(.*)",
     events.TaskFailed), (r"/monitor/api/task/events/task-revoked/(.*)",
                          events.TaskRevoked),
    (r"/monitor/api/task/events/task-retried/(.*)",
     events.TaskRetried), (r"/monitor/api/task/events/task-custom/(.*)",
                           events.TaskCustom)
]

debug_flag = True


def get_wsgi_server():
    flower_app = Flower(
        capp=scheduler, options=default_options, handlers=handlers, **settings)
    wsgi_app = tornado.wsgi.WSGIAdapter(flower_app)
    wsgi_app = ErrorMiddleware(wsgi_app, debug=debug_flag)
    wsgi_app = TransLogger(wsgi_app, setup_console_handler=debug_flag)

    return wsgi_app


class FlowerServer():
    ENV_VAR_PREFIX = 'FLOWER_'

    def __init__(self):
        self.prog_name = 'flower'
        self.apply_options()
        self.app = Celery()
        self.app.config_from_object(flowerconfig)

        self.flower = Flower(
            capp=self.app, options=options, handlers=handlers, **settings)
        self.flower.pool = self.flower.pool_executor_cls(max_workers=self.flower.max_workers)
        self.flower.events.start()
        self.flower.io_loop.add_future(
            control.ControlHandler.update_workers(app=self.flower),
            callback=lambda x: print(
                'Successfully updated worker cache'))
        self.flower.started = True
        # self.flower.io_loop.start()
        self.print_banner('ssl_options' in settings)

    def apply_options(self):
        # parse the command line to get --conf option
        try:
            parse_config_file(os.path.abspath('flowerconfig.py'), final=False)
        except IOError:
            if os.path.basename(options.conf) != 'flowrconfig.py':
                raise

    def getWSGIServer(self):
        wsgi_app = tornado.wsgi.WSGIAdapter(self.flower)
        wsgi_app = ErrorMiddleware(wsgi_app, debug=debug_flag)
        wsgi_app = TransLogger(wsgi_app, setup_console_handler=debug_flag)

        return wsgi_app

    @staticmethod
    def is_flower_option(arg):
        name, _, value = arg.lstrip('-').partition("=")
        name = name.replace('-', '_')
        return hasattr(options, name)

    def is_flower_envvar(self, name):
        return name.startswith(self.ENV_VAR_PREFIX) and\
            name[len(self.ENV_VAR_PREFIX):].lower() in default_options

    def print_banner(self, ssl):
        print('Broker: %s', self.app.connection().as_uri())
        print('Registered tasks: \n%s', pformat(sorted(self.app.tasks.keys())))
        print('Settings: %s', pformat(settings))
