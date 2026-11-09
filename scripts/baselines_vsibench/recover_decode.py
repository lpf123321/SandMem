"""Decode the exact frozen frame IDs; no frame replacement or model retry.

Decord's indexed-seek path may raise EOF on a video which can be decoded in
presentation order. A fresh sequential reader recovers the same indexed RGB
observations. Original errors and both decoder costs remain visible.
"""
import hashlib
import json
from pathlib import Path
import time
import traceback


def decode_selected(reader, source_ids, source_path, receipt_path):
    import decord
    import numpy as np
    source_ids=list(source_ids)
    if source_ids != sorted(set(source_ids)) or not source_ids or source_ids[-1]>=len(reader):
        raise ValueError('exact, ordered, valid frozen source IDs required')
    started=time.perf_counter()
    receipt=dict(source=str(Path(source_path).resolve()),source_frame_ids=source_ids,
                 source_fps=float(reader.get_avg_fps()),declared_source_frames=len(reader),
                 learned_model_calls=0,frame_replacement=False,status='decoding')
    target=Path(receipt_path)
    try:
        if Path(source_path).name=='42446056.mp4' and Path(source_path).parent.name=='arkitscenes':
            # Already reproduced in the failed GPU job and independent CPU
            # probe. Do not repeat the known failing random-seek operation.
            receipt['backend']='decord_sequential_known_eof'
            value=None
        else:
            try:
                value=reader.get_batch(source_ids).asnumpy()
                receipt['backend']='decord_original_get_batch'
            except decord.DECORDError as error:
                if 'Unable to handle EOF' not in str(error):
                    raise
                receipt['indexed_decode_failure']=str(error)
                receipt['backend']='decord_sequential_after_eof'
                value=None
        if value is None:
            fresh=decord.VideoReader(str(source_path),ctx=decord.cpu(0),num_threads=1)
            if len(fresh)!=len(reader) or float(fresh.get_avg_fps())!=receipt['source_fps']:
                raise ValueError('source metadata changed between decoding paths')
            wanted=set(source_ids);frames=[]
            for index in range(source_ids[-1]+1):
                frame=fresh.next()
                if index in wanted:
                    frames.append(frame.asnumpy())
            if len(frames)!=len(source_ids):
                raise ValueError('sequential decoding did not recover every required frame')
            value=np.stack(frames)
        if value.shape[0]!=len(source_ids) or value.dtype!=np.uint8:
            raise ValueError('exact uint8 source RGB observations required')
        receipt.update(status='decoded',shape=list(value.shape),
                       rgb_sha256=hashlib.sha256(value.tobytes()).hexdigest())
        return value
    except Exception:
        receipt.update(status='failed',failure=traceback.format_exc())
        raise
    finally:
        receipt['seconds']=time.perf_counter()-started
        target.write_text(json.dumps(receipt,indent=2))
