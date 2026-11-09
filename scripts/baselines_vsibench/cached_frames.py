"""Model-free RGB cache; the MLLM process never imports or invokes Decord.

Source IDs retain the frozen 1 FPS sampling rule. Missing caches are produced
by a fresh CPU-only interpreter; existing caches are verified before loading.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def file_sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(8*1024*1024),b''):
            result.update(part)
    return result.hexdigest()


class RGBBatch:
    def __init__(self,frames):
        self.frames=frames

    def asnumpy(self):
        import numpy as np
        return np.array(self.frames,copy=True)


class CachedReader:
    def __init__(self,folder,source_path):
        import numpy as np
        self.folder=Path(folder)
        self.receipt=json.loads((self.folder/'receipt.json').read_text())
        meta=self.receipt
        if meta['status']!='decoded' or Path(meta['source']).resolve()!=Path(source_path).resolve():
            raise ValueError('cache/source provenance mismatch')
        if file_sha(source_path)!=meta['source_sha256']:
            raise ValueError('source video changed since RGB cache creation')
        if file_sha(self.folder/'rgb.npy')!=meta['npy_sha256']:
            raise ValueError('RGB cache file checksum mismatch')
        self.frames=np.load(self.folder/'rgb.npy',mmap_mode='r')
        if self.frames.dtype!=np.uint8 or list(self.frames.shape)!=meta['shape']:
            raise ValueError('RGB cache shape or dtype mismatch')
        if hashlib.sha256(self.frames.tobytes()).hexdigest()!=meta['rgb_sha256']:
            raise ValueError('actual RGB pixel checksum mismatch')
        self.ids=meta['source_frame_ids']
        # Same time-arange sampling as the accepted original implementation.
        fps=meta['source_fps'];count=meta['declared_source_frames']
        expected=sorted(set(min(count-1,int(t*fps)) for t in np.arange(0,count/fps,1)))
        if self.ids!=expected or len(self.ids)!=len(self.frames):
            raise ValueError('cache does not contain every exact frozen source ID')

    def __len__(self):
        return self.receipt['declared_source_frames']

    def get_avg_fps(self):
        return self.receipt['source_fps']

    def get_batch(self,ids):
        if list(ids)!=self.ids:
            raise ValueError('this cache reader accepts only the complete frozen frame list')
        return RGBBatch(self.frames)


def sample_frames(path,fps):
    if fps!=1:
        raise ValueError('this comparison is frozen to 1 FPS')
    path=Path(path).resolve()
    folder=Path(__file__).resolve().parent.parent/'rgb_cache'/(path.parent.name+'_'+path.stem)
    if not (folder/'receipt.json').exists():
        folder.mkdir(parents=True,exist_ok=False)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',
                 OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        started=time.perf_counter()
        with (folder/'worker.log').open('w') as log:
            result=subprocess.run([sys.executable,'-u',str(Path(__file__).with_name('cpu_decode_worker.py')),
                '--source',str(path),'--output',str(folder)],env=env,stdout=log,stderr=subprocess.STDOUT)
        (folder/'worker_process.json').write_text(json.dumps(dict(returncode=result.returncode,
            seconds=time.perf_counter()-started,CUDA_VISIBLE_DEVICES='',learned_model_calls=0),indent=2))
        if result.returncode:
            raise RuntimeError('isolated CPU decode failed; see '+str(folder/'worker.log'))
    reader=CachedReader(folder,path)
    return reader,list(reader.ids),reader.get_avg_fps()


def decode_selected(reader,source_ids,source_path,receipt_path):
    started=time.perf_counter()
    if not isinstance(reader,CachedReader):
        raise ValueError('GPU inference must use a validated CPU RGB cache')
    value=reader.get_batch(source_ids).asnumpy()
    receipt=dict(status='decoded',backend='validated_cpu_rgb_cache_no_decord_in_model_process',
        source=str(Path(source_path).resolve()),source_frame_ids=list(source_ids),
        source_fps=reader.get_avg_fps(),declared_source_frames=len(reader),
        shape=list(value.shape),rgb_sha256=reader.receipt['rgb_sha256'],
        cache_receipt=str((reader.folder/'receipt.json').resolve()),
        learned_model_calls=0,seconds=time.perf_counter()-started)
    Path(receipt_path).write_text(json.dumps(receipt,indent=2))
    return value
