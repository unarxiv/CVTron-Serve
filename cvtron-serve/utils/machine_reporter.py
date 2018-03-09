# coding:utf-8
import os
import re
from subprocess import Popen, PIPE


def safeFloatCast(strNumber):
    try:
        number = float(strNumber)
    except ValueError:
        number = float('nan')
    return number


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

    def get_mem(self):
        memory = None
        with open('/proc/meminfo') as f:
            for line in f.readlines():
                mem = re.match(r'MemTotal+: (.*)', line)
                if mem:
                    memory = mem.group(1)
        if memory:
            return memory.strip()
        else:
            return '0 kB'

    def get_gpu(self):
        # Get ID, processing and memory utilization for all GPUs
        p = Popen(["nvidia-smi", "--query-gpu=index,uuid,utilization.gpu,memory.total,memory.used,memory.free,driver_version,name,gpu_serial,display_active,display_mode",
                   "--format=csv,noheader,nounits"], stdout=PIPE)
        output = p.stdout.read().decode('UTF-8')
        # output = output[2:-1] # Remove b' and ' from string added by python
        # print(output)
        # Parse output
        # Split on line break
        lines = output.split(os.linesep)
        # print(lines)
        numDevices = len(lines)-1
        deviceIds = []
        gpuUtil = []
        memTotal = []
        memUsed = []
        memFree = []
        driver = []
        GPUs = []
        for g in range(numDevices):
            line = lines[g]
            # print(line)
            vals = line.split(', ')
            # print(vals)
            for i in range(11):
                # print(vals[i])
                if (i == 0):
                    deviceIds.append(int(vals[i]))
                elif (i == 1):
                    uuid = vals[i]
                elif (i == 2):
                    gpuUtil.append(safeFloatCast(vals[i])/100)
                elif (i == 3):
                    memTotal.append(safeFloatCast(vals[i]))
                elif (i == 4):
                    memUsed.append(safeFloatCast(vals[i]))
                elif (i == 5):
                    memFree.append(safeFloatCast(vals[i]))
                elif (i == 6):
                    driver = vals[i]
                elif (i == 7):
                    gpu_name = vals[i]
                elif (i == 8):
                    serial = vals[i]
                elif (i == 9):
                    display_active = vals[i]
                elif (i == 10):
                    display_mode = vals[i]
            gpu = {
                'device_id': deviceIds[g],
                'uuid': uuid,
                'gpuUtil': gpuUtil[g],
                'memTotal': memTotal[g],
                'memUsed': memUsed[g],
                'memFree': memFree[g],
                'driver': driver,
                'gpu_name': gpu_name,
                'serial': serial,
                'display_mode': display_mode,
                'display_active': display_active
            }
            GPUs.append(gpu)
        return GPUs  # (deviceIds, gpuUtil, memUtil)

    def get_all(self):
        return {
            'cpu':self.get_cpu(),
            'gpu':self.get_gpu(),
            'mem':self.get_mem()
        }
