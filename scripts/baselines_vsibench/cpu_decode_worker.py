"""Isolated CPU-only decoding; no Torch, Spatial model or answers are loaded."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

from spatial_features import sample_frames
from recover_decode import decode_selected
from cached_frames import file_sha


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    import numpy as np
    started=time.perf_counter()
    reader,ids,fps=sample_frames(args.source,1)
    frames=decode_selected(reader,ids,args.source,args.output/'decoder_receipt.json')
    if 'torch' in sys.modules:
        raise RuntimeError('isolated CPU decoder unexpectedly imported Torch')
    target=args.output/'rgb.npy'
    if target.exists():
        raise ValueError('never overwrite an existing RGB cache')
    np.save(target,frames)
    receipt=dict(status='decoded',source=str(args.source.resolve()),
        source_sha256=file_sha(args.source),source_fps=fps,declared_source_frames=len(reader),
        source_frame_ids=ids,shape=list(frames.shape),
        rgb_sha256=hashlib.sha256(frames.tobytes()).hexdigest(),npy_sha256=file_sha(target),
        learned_model_calls=0,torch_imported=False,seconds=time.perf_counter()-started)
    (args.output/'receipt.json').write_text(json.dumps(receipt,indent=2))
    print('CPU RGB cache saved',args.source,len(ids),flush=True)


if __name__=='__main__':
    main()
