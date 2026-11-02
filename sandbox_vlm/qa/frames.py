"""Video-level frame-ID lookup with source identity and image integrity checks."""
import hashlib
from pathlib import Path

from PIL import Image


class PreparedFrameReader:
    def __init__(self, sample_dir, manifest, *, video_id):
        if manifest['video_id'] != video_id:
            raise ValueError('cross-video source mapping')
        self.sample_dir = Path(sample_dir).resolve()
        self.frames = {}
        for clip in manifest['clips']:
            for frame in clip['frames']:
                old = self.frames.get(frame['frame_id'])
                if old and any(old[k] != frame[k] for k in ('source_index', 'sha256', 'image_size_hw')):
                    raise ValueError('overlap frame identity differs')
                self.frames.setdefault(frame['frame_id'], frame)

    def available_frame_ids(self):
        return sorted(self.frames)

    def read(self, frame_id, *, max_side):
        frame = self.frames[frame_id]
        path = (self.sample_dir / frame['path']).resolve()
        if not path.is_relative_to(self.sample_dir):
            raise ValueError('image path escapes video source mapping')
        if hashlib.sha256(path.read_bytes()).hexdigest() != frame['sha256']:
            raise ValueError('source image changed')
        with Image.open(path) as opened:
            image = opened.convert('RGB')
            if [image.height, image.width] != frame['image_size_hw']:
                raise ValueError('source image dimensions differ')
            image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
            return image.copy(), dict(frame_id=frame_id, source_index=frame['source_index'],
                                     source_sha256=frame['sha256'], source_image_size_hw=frame['image_size_hw'],
                                     input_image_size_hw=[image.height, image.width])
