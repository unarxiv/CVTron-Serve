# coding:utf-8
import subprocess
import re

def combine_dicts(recs):
    """Combine a list of recs, appending values to matching keys"""
    if not recs:
        return None

    if len(recs) == 1:
        return recs.pop()

    new_rec = {}
    for rec in recs:
        for k, v in rec.iteritems():
            if k in new_rec:
                new_rec[k] = "%s, %s" % (new_rec[k], v)
            else:
                new_rec[k] = v


class CommandParser(object):
    """Object for extending to parse command outputs"""

    ITEM_REGEXS = []
    ITEM_SEPERATOR = False
    DATA = None
    MUST_HAVE_FIELDS = []

    def __init__(self, data, regexs=None, seperator=None):
        self.set_data(data)
        self.set_regexs(regexs)
        self.set_seperator(seperator)

    def set_data(self, data):
        if data:
            self.DATA = data.strip()
        else:
            self.DATA = ""

    def set_regexs(self, regexs):
        if regexs:
            self.ITEM_REGEXS = regexs

    def set_seperator(self, seperator):
        if seperator:
            self.ITEM_SEPERATOR = seperator

    def parse_item(self, item):
        rec = {}
        print(item)
        for regex in self.ITEM_REGEXS:
            matches = [m.groupdict() for m in re.finditer(regex, item)]
            mdicts = combine_dicts(matches)
            if mdicts:
                rec = dict(list(rec.items()) + list(mdicts.items()))
        print(rec)
        return rec

    def parse_items(self):
        if not self.ITEM_SEPERATOR:
            return [self.parse_item(self.DATA)]
        else:
            recs = []
            for data in self.DATA.split(self.ITEM_SEPERATOR):
                rec = self.parse_item(data)
                recs.append(rec)
            return recs

    def parse(self):
        if self.ITEM_SEPERATOR:
            raise Exception("A seperator has been specified: '%s'. " +
                            "Please use 'parse_items' instead")

        return self.parse_item(self.DATA)


class CPUInfoParser(CommandParser):
    REGEX_TEMPLATE = r'%s([\ \t])+\:\ (?P<%s>.*)'

    ITEM_SEPERATOR = "\n\n"

    ITEM_REGEXS = [
        REGEX_TEMPLATE % ('processor', 'processor'),
        REGEX_TEMPLATE % ('vendor_id', 'vendor_id'),
        REGEX_TEMPLATE % (r'cpu\ family', 'cpu_family'),
        REGEX_TEMPLATE % ('model', 'model'),
        REGEX_TEMPLATE % (r'model\ name', 'model_name'),
        REGEX_TEMPLATE % ('stepping', 'stepping'),
        REGEX_TEMPLATE % ('microcode', 'microcode'),
        REGEX_TEMPLATE % (r'cpu\ MHz', 'cpu_mhz'),
        REGEX_TEMPLATE % (r'cache\ size', 'cache_size'),
        REGEX_TEMPLATE % (r'fpu', 'fpu'),
        REGEX_TEMPLATE % (r'fpu_exception', 'fpu_exception'),
        REGEX_TEMPLATE % (r'cpuid\ level', 'cpuid_level'),
        REGEX_TEMPLATE % (r'wp', 'wp'),
        REGEX_TEMPLATE % (r'flags', 'flags'),
        REGEX_TEMPLATE % (r'bogomips', 'bogomips'),
        REGEX_TEMPLATE % (r'clflush\ size', 'clflush_size'),
        REGEX_TEMPLATE % (r'cache_alignment', 'cache_alignment'),
        REGEX_TEMPLATE % (r'address\ sizes', 'address_sizes'),
        REGEX_TEMPLATE % (r'power\ management', 'power_management'),
    ]

class Machine(object):
    client = None

    def __init__(self):
        self.host = 'localhost'

    def __del__(self):
        if self.client:
            self.client.close()

    def _run(self, cmd):
        cmdstr = ' '.join(cmd)
        process = subprocess.Popen(cmdstr, stdout=subprocess.PIPE, shell=True)
        stdout, stderr = process.communicate()
        if process.returncode == 0:
            return str(stdout).strip()
        else:
            raise Exception("stderr: %s" % str(stderr))

    def execute(self, cmd):
        return self._run(cmd)

    def parse_string(self, data):
        rec = {}
        lines = data.split('\n')
        for line in lines:
            if not line:
                continue
            k, v = line.split('=')
            rec[k] = v.strip("'")
        return rec

    def get_cpu_data(self):
        data = self.execute(['cat /proc/cpuinfo'])
        cip = CPUInfoParser(data)
        return cip.parse_items()

m = Machine()
a = m.get_cpu_data()
print(a)