"""Original frozen discovery with a complete-record stopping boundary only."""
from sandbox_vlm.perception.discovery import QwenDiscovery


class ContinuousDiscovery(QwenDiscovery):
    def discover_batch(self, image_paths, frame_ids, *, question=None):
        if len(image_paths) != 1 or len(frame_ids) != 1:
            raise ValueError('continuous perception discovers one chronological source frame per call')
        return [self.discover(image_paths[0], frame_ids[0], question=question)]

    def discover(self, image_path, frame_id, *, question=None):
        from transformers import LogitsProcessorList
        from sandbox_vlm.perception.record_boundary import RecordBoundary
        original = self.model
        owner = self

        class BoundaryModel:
            def __getattr__(self, key):
                return getattr(original, key)

            def generate(self, **kwargs):
                if 'logits_processor' in kwargs:
                    raise ValueError('unexpected existing discovery logits processor')
                eos = original.generation_config.eos_token_id
                self.boundary = RecordBoundary(owner.processor.tokenizer,
                    kwargs['input_ids'].shape[1], owner.config['max_new_tokens'], frame_id,
                    eos if isinstance(eos, list) else [eos])
                kwargs['logits_processor'] = LogitsProcessorList([self.boundary])
                output = original.generate(**kwargs)
                self.output_ids = output[0, kwargs['input_ids'].shape[1]:].detach().cpu().tolist()
                return output

        wrapper = BoundaryModel()
        self.model = wrapper
        try:
            row = super().discover(image_path, frame_id, question=question)
        finally:
            self.model = original
        emitted = len(row['detections'])
        unique = []
        for detection in row['detections']:
            if detection not in unique:
                unique.append(detection)
        row['detections'] = unique
        row['output_token_ids'] = wrapper.output_ids
        row['output_contract'] = dict(**wrapper.boundary.report(),
            original_prompt=True, original_image=True, original_output_cap=self.config['max_new_tokens'],
            emitted_records=emitted, retained_records=len(unique),
            removed_identical_records=emitted-len(unique),
            coordinate_perturbation=False, physical_identity_merge=False,
            original_duplicate_guard_retained=True,
            raw_output_modified=False, constraints_before_boundary=False)
        return row
