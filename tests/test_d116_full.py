from scripts.run_d116_full import tracking_state,restore_tracking
from sandbox_vlm.perception.coverage_short_track import CoverageShortTrack


def test_tracking_checkpoint_preserves_complete_continuation_state():
    a=CoverageShortTrack(None);a.attempted={'a','b'};a.new_seed_attempts=6
    a.processed_through={'a':15};a.failed_source_keys={'failed'};a.last_watermark=19
    a.accepted_edges=[dict(source='a',target='b')];a.new_edges=[dict(source='a',target='b')]
    a.seed_attempts=9;a.legacy_seed_attempts=3;a.history=[dict(seed_attempts=6)]
    b=CoverageShortTrack(None);restore_tracking(b,tracking_state(a))
    assert tracking_state(a)==tracking_state(b)
    assert b.attempted=={'a','b'}and b.failed_source_keys=={'failed'}


def test_tracking_checkpoints_do_not_serialize_models_or_images():
    t=CoverageShortTrack(object());state=tracking_state(t)
    assert 'tracker'not in state
    import json
    json.dumps(state)
