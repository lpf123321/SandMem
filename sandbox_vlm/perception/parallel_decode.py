"""Bounded exact-source contiguous decode, never segmentation or future reads."""
from concurrent.futures import ThreadPoolExecutor


class CoverageReader:
    def __init__(self, path, mapping, *, workers=0):
        import cv2
        self.path, self.mapping = path, mapping
        self.workers = workers if round(mapping.fps / mapping.sample_fps) >= 8 else 0
        self.sequential = None if self.workers else cv2.VideoCapture(str(path))
        self.position = -1
        if self.sequential is not None and not self.sequential.isOpened():
            raise ValueError('cannot open sequential coverage source')
        self.pool = ThreadPoolExecutor(max_workers=self.workers) if self.workers else None
        self.last_id = -1

    def read_batch(self, ids, *, allowed_through, max_side):
        if not ids or ids != list(range(self.last_id + 1, self.last_id + 1 + len(ids))) or ids[-1] > allowed_through:
            raise ValueError('missing or future parallel source read')
        if self.pool:
            import math
            size = math.ceil(len(ids) / self.workers)
            pending = [self.pool.submit(self._segment, ids[i:i + size], max_side)
                       for i in range(0, len(ids), size)]
            images = [image for future in pending for image in future.result()]
        else:
            import cv2
            from PIL import Image
            images = []
            for fid in ids:
                source = self.mapping.by_id(fid).source_index
                while self.position < source:
                    if not self.sequential.grab():
                        raise ValueError('sequential coverage source decode failed')
                    self.position += 1
                ok, pixels = self.sequential.retrieve()
                if not ok:
                    raise ValueError('sequential coverage source retrieval failed')
                image = Image.fromarray(cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB))
                image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
                images.append(image)
        self.last_id = ids[-1]
        return images

    def _segment(self, ids, max_side):
        import cv2
        from PIL import Image
        capture = cv2.VideoCapture(str(self.path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])
        if not capture.isOpened():
            raise ValueError('cannot open bounded source segment')
        try:
            first = self.mapping.by_id(ids[0]).source_index
            if first:
                capture.set(cv2.CAP_PROP_POS_FRAMES, first)
            position = first - 1
            output = []
            for fid in ids:
                source = self.mapping.by_id(fid).source_index
                while position < source:
                    if not capture.grab():
                        raise ValueError('bounded source decode failed')
                    position += 1
                ok, pixels = capture.retrieve()
                if not ok or int(capture.get(cv2.CAP_PROP_POS_FRAMES)) != source + 1:
                    raise ValueError('bounded source decode index mismatch')
                image = Image.fromarray(cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB))
                image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
                output.append(image)
            return output
        finally:
            capture.release()

    def close(self):
        if self.pool:
            self.pool.shutdown(wait=True)
        if self.sequential:
            self.sequential.release()
