import io
import tarfile
import time
import zlib

import numpy as np

PREFIX = 'data/'
SUFFIX = '.npy.z'

class NpyTarWriter(object):
    def __init__(self, fname):
        self.tfile = tarfile.open(fname, 'w|')

    def add(self, arr, name):

        sio = io.BytesIO()
        np.save(sio, arr)
        zbuf = zlib.compress(sio.getvalue())
        sio.close()

        zsio = io.BytesIO(zbuf)
        tinfo = tarfile.TarInfo('{}{}{}'.format(PREFIX, name, SUFFIX))
        tinfo.size = len(zbuf)
        tinfo.mtime = time.time()
        zsio.seek(0)
        self.tfile.addfile(tinfo, zsio)
        zsio.close()

    def close(self):
        self.tfile.close()

class NpyTarReader(object):
    def __init__(self, fname):
        self.fname = fname
        self.tfile = tarfile.open(self.fname, 'r|')

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        entry = self.tfile.next()
        if entry is None:
            raise StopIteration()
        name = entry.name[len(PREFIX): -len(SUFFIX)]
        fileobj = self.tfile.extractfile(entry)
        buf = zlib.decompress(fileobj.read())
        arr = np.load(io.BytesIO(buf))
        return arr, name

    def length(self):
        return len(self.tfile.getnames())

    def reopen(self):
        self.tfile.close()
        self.tfile = tarfile.open(self.fname, 'r|')

    def close(self):
        self.tfile.close()
