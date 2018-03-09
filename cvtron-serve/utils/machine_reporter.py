# coding:utf-8

import re

class Machine(object):
    def __init__(self):
        self.host = 'localhost'
    def get_cpu(self):
        cpus = []
        cpu = None
        with open('/proc/cpuinfo') as f:
            for line in f.readlines():
                core = re.match(r'processor\s+: (\d+)', line)
                if core:
                    if cpu:
                        cpus.append(cpu)
                    cpu = {}
                    cpu['processor'] = int(core.group(1))
                vendor = re.match(r'vendor_id\s+: (.*)', line)
                if vendor:
                    cpu['vendor_id'] = vendor.group(1)

                model_name = re.match(r'model name\s+: (.*)', line)
                if model_name:
                    cpu['model_name'] = model_name.group(1)
        if cpu:
            cpus.append(cpu)
        return cpus

m = Machine()
cpus = m.get_cpu()
print(cpus)