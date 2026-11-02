import unittest
from sandbox_vlm.qa.two_pass import candidate_from_generation, verification_messages, decide, eligible


def generation(raw, status='generated'):
    return dict(status=status, raw_output=raw)


class TwoPassTest(unittest.TestCase):
    def test_failed_draft_never_extracts_truncated_answer(self):
        self.assertIsNone(candidate_from_generation(generation('FINAL_ANSWER: A', 'answer_failed'))['answer'])

    def test_correction_requires_actual_frame_in_notes(self):
        draft = candidate_from_generation(generation('Ground first.\nFINAL_ANSWER: A'))
        for raw in ['F999 shows the subject.\nFINAL_ANSWER: B', 'Record I32 says B.\nFINAL_ANSWER: B',
                    'No citation.\nFINAL_ANSWER: F158']:
            self.assertEqual(decide(draft, generation(raw), [158])['prediction'], 'A')
        self.assertEqual(decide(draft, generation('Frame ID 158 shows it.\nFINAL_ANSWER: B'), [158])['prediction'], 'B')

    def test_failed_verification_retains_valid_draft(self):
        draft = candidate_from_generation(generation('FINAL_ANSWER: C'))
        self.assertEqual(decide(draft, generation('', 'answer_failed'), [1])['prediction'], 'C')

    def test_failed_draft_allows_new_valid_answer(self):
        draft = candidate_from_generation(generation('', 'answer_failed'))
        self.assertEqual(decide(draft, generation('FINAL_ANSWER: D'), [1])['prediction'], 'D')

    def test_injection_does_not_change_existing_evidence(self):
        base = [dict(role='user',content=[dict(type='text',text='header'),
            dict(type='image',evidence_image_index=0,pixel_sha256='map'),
            dict(type='text',text='Original question (verbatim):\nquestion'),dict(type='text',text='footer')])]
        after = verification_messages(base, dict(status='ok',answer='A',raw_output='test'))
        del after[0]['content'][-3]
        self.assertEqual(after, base)

    def test_dense_gate_does_not_read_task_or_geometry(self):
        self.assertFalse(eligible(dict(sparse=False), None, None))


if __name__ == '__main__':
    unittest.main()
