"""Contract tests with a synthetic policy; not a production threshold choice."""
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from sandbox_vlm.memory.continuous_scenes import ContinuousSceneMemory, SceneObservation, VoxelScenePolicy
from sandbox_vlm.memory.geometry import CameraParams
from sandbox_vlm.perception.continuous_stream import ContinuousStreamVGGT, tensor_storage_bytes
from sandbox_vlm.perception.scene_points import reliable_scene_points, scale_fixed_voxel_size
from sandbox_vlm.qa.continuous_selection import select_continuous_frames


class SyntheticPolicy:
    def initial_limit(self, bounds):
        return float(bounds[1, 0] - bounds[0, 0] + 2)

    def sufficient_overlap(self, occupied, incoming):
        return bool(occupied & incoming)

    def within_limit(self, bounds, fixed_limit):
        return bounds[1, 0] - bounds[0, 0] <= fixed_limit


def camera(fid, reference='video'):
    pose = np.eye(4)
    pose[0, 3] = fid
    return CameraParams(fid, pose, np.eye(3), (2, 2), reference)


class FakeModel(torch.nn.Module):
    """Tiny heads follow official shapes and mutate caches like upstream."""
    def __init__(self):
        super().__init__()
        self.indices = []
        self.aggregator = SimpleNamespace(depth=1)
        self.aggregator = self.Aggregator(self)
        self.camera_head = self.Camera()

    class Aggregator:
        depth = 1

        def __init__(self, owner):
            self.owner = owner

        def __call__(self, images, *, past_key_values, use_cache, past_frame_idx):
            assert use_cache and images.shape[:3] == (1, 1, 3)
            old = past_key_values[0]
            assert (old is None) == (past_frame_idx == 0)
            self.owner.indices.append(past_frame_idx)
            new = torch.full((1, 1, 1, 1, 1), float(past_frame_idx))
            key = new if old is None else torch.cat([old[0], new], dim=2)
            past_key_values[0] = (key, key.clone())
            return [images], 0, past_key_values

    class Camera:
        trunk_depth = 1

        def __call__(self, tokens, *, num_iterations, past_key_values_camera, use_cache):
            old = past_key_values_camera[0]
            for i in range(num_iterations):
                new = torch.full((1, 1, 1, 1, 1), float(self.calls * num_iterations + i))
                key = new if old is None else torch.cat([old[0], new], dim=2)
                old = (key, key.clone())
            past_key_values_camera[0] = old
            self.calls += 1
            return [torch.zeros(1, 1, 9)], past_key_values_camera

        def __init__(self):
            self.calls = 0

    def depth_head(self, tokens, **kwargs):
        return torch.ones(1, 1, 2, 2, 1), torch.ones(1, 1, 2, 2)

    def point_head(self, tokens, **kwargs):
        return torch.ones(1, 1, 2, 2, 3), torch.ones(1, 1, 2, 2)


class ContinuousTests(unittest.TestCase):
    def memory(self):
        return ContinuousSceneMemory(reference='video', policy=SyntheticPolicy())

    def add(self, mem, fid, voxels, lo=0, hi=1, observations=()):
        return mem.add(camera=camera(fid), reliable_points=np.array([[lo, 0, 0], [hi, 1, 1]]),
                       occupied_voxels=frozenset(voxels), observations=observations,
                       available_through=fid)

    def stream(self, limit=1000, frames=10, cache_frames=16):
        return ContinuousStreamVGGT(FakeModel().eval(), reference='video',
                                   frame_ids=tuple(range(0, 100, 10)),
                                   max_frames=frames, cache_limit_bytes=limit, cache_frames=cache_frames)

    def step(self, stream, i):
        return stream.step(torch.zeros(1, 3, 2, 2), frame_id=i * 10,
                           sample_index=i, available_through=i * 10)

    def test_middle_camera_and_duplicate_instance_across_scenes(self):
        mem = self.memory()
        self.add(mem, 0, [0], observations=(SceneObservation('a', 7, 0),))
        self.add(mem, 10, [0, 1])  # intermediate frame without an object
        self.add(mem, 20, [9], observations=(SceneObservation('b', 7, 20),))
        self.assertEqual(list(mem.scenes[0].cameras), [0, 10])
        self.assertTrue(mem.scenes[0].closed)
        self.assertEqual(mem.scenes[0].observations[0].instance_id,
                         mem.scenes[1].observations[0].instance_id)
        self.assertEqual(mem.occupied_voxels, frozenset([9]))
        np.testing.assert_array_equal(mem.scenes[1].cameras[20].T_ref_from_camera, camera(20).T_ref_from_camera)

    def test_fixed_initial_limit_and_revisit_new_scene(self):
        mem = self.memory()
        self.add(mem, 0, [0])
        self.add(mem, 1, [0], hi=2)
        self.assertEqual(mem.active.fixed_limit, 3)
        self.assertEqual(self.add(mem, 2, [0], hi=4)['status'], 'range_limit')
        self.add(mem, 3, [9])
        self.add(mem, 4, [0])
        self.assertEqual(len(mem.scenes), 4)
        mem.finish()
        self.assertEqual(mem.occupied_voxels, frozenset())

    def test_reference_rejected_and_missing_geometry_retains_null_assignment(self):
        mem = self.memory()
        with self.assertRaisesRegex(ValueError, 'reference'):
            mem.add(camera=camera(0, 'other'), reliable_points=np.ones((1, 3)),
                    occupied_voxels=frozenset([1]), observations=(), available_through=0)
        result = mem.add(camera=camera(0), reliable_points=np.empty((0, 3)),
                         occupied_voxels=frozenset(), observations=(), available_through=0)
        self.assertEqual(mem.scenes, [])
        self.assertIsNone(result['scene_id'])
        self.assertEqual(mem.views[0]['camera'].frame_id, 0)

    def test_scene_switch_does_not_reset_stream(self):
        stream, mem = self.stream(), self.memory()
        for i in range(3):
            result = self.step(stream, i)
            self.add(mem, i * 10, [i])
            self.assertEqual(result['reference'], 'video')
        self.assertEqual(stream.model.indices, [0, 1, 2])
        self.assertEqual(stream.camera_cache[0][0].shape[2], 12)
        self.assertEqual(stream.cache[0][0].shape[2], 3)
        self.assertEqual(len(mem.scenes), 3)

    def test_no_skipped_samples_or_future_frames(self):
        stream = self.stream()
        with self.assertRaisesRegex(ValueError, 'prefix'):
            self.step(stream, 1)
        with self.assertRaisesRegex(ValueError, 'prefix'):
            # Renumbering retrieved A/B endpoints cannot hide a skipped frame.
            stream.step(torch.zeros(1, 3, 2, 2), frame_id=20,
                        sample_index=0, available_through=20)
        with self.assertRaisesRegex(ValueError, 'prefix'):
            stream.step(torch.zeros(1, 3, 2, 2), frame_id=10,
                        sample_index=0, available_through=9)
        self.assertEqual(stream.model.indices, [])

    def test_cache_budget_stops_and_forbids_retry(self):
        stream = self.stream(limit=40)
        self.step(stream, 0)
        with self.assertRaisesRegex(RuntimeError, 'cache budget'):
            self.step(stream, 1)
        with self.assertRaisesRegex(RuntimeError, 'restart or retry'):
            self.step(stream, 1)

    def test_frame_budget_and_changed_grid(self):
        stream = self.stream(frames=1)
        self.step(stream, 0)
        with self.assertRaisesRegex(RuntimeError, 'frame budget'):
            self.step(stream, 1)
        stream = self.stream()
        self.step(stream, 0)
        with self.assertRaisesRegex(ValueError, 'grid'):
            stream.step(torch.zeros(1, 3, 3, 2), frame_id=10,
                        sample_index=1, available_through=10)

    def test_tensor_storage_counts_alias_once(self):
        value = torch.ones(10)
        self.assertEqual(tensor_storage_bytes([value, value[:2]]), 40)

    def test_fifo_many_rollovers_retains_complete_frames_and_absolute_index(self):
        stream = self.stream(cache_frames=3)
        for i in range(10):
            result = self.step(stream, i)
            first = max(0, i - 2)
            self.assertEqual(stream.cache[0][0].flatten().tolist(), list(range(first, i + 1)))
            self.assertEqual(stream.camera_cache[0][0].flatten().tolist(), list(range(first * 4, (i + 1) * 4)))
            self.assertEqual(result['cost']['retained_frame_ids'], list(range(first * 10, i * 10 + 1, 10)))
            self.assertLessEqual(result['cost']['cache_after_bytes'], 120)
        self.assertEqual(stream.model.indices, list(range(10)))
        self.assertEqual(stream.reference, 'video')

    def test_fifo_releases_old_storage_and_minimum_history(self):
        stream = self.stream(cache_frames=2)
        for i in range(5):
            self.step(stream, i)
        # A retained slice without clone would have larger underlying storage.
        self.assertEqual(tensor_storage_bytes(stream.cache), 16)
        with self.assertRaisesRegex(ValueError, 'at least two'):
            self.stream(cache_frames=1)

    def test_fifo_does_not_split_scene_or_delete_external_cameras_and_links(self):
        stream, mem = self.stream(cache_frames=3), self.memory()
        for i in range(10):
            self.step(stream, i)
            result = self.add(mem, i * 10, [0],
                              observations=(SceneObservation(str(i), 7, i * 10),))
            self.assertEqual(result['status'], 'initial' if i == 0 else 'joined')
        self.assertEqual(len(mem.scenes), 1)
        self.assertEqual(len(mem.active.cameras), 10)
        self.assertEqual(len(mem.active.observations), 10)
        self.assertEqual({o.instance_id for o in mem.active.observations}, {7})
        self.assertEqual(stream.cached_frame_ids, (70, 80, 90))

    def test_documented_voxel_fraction_and_diagonal(self):
        policy = VoxelScenePolicy(voxel_size=.1, overlap_threshold=.5, expansion_factor=3)
        self.assertTrue(policy.sufficient_overlap(frozenset(range(100)), frozenset([0, 200])))
        self.assertFalse(policy.sufficient_overlap(frozenset([0]), frozenset([0, 1, 2])))
        self.assertAlmostEqual(policy.initial_limit(np.zeros((2, 3))), .3)
        self.assertTrue(policy.within_limit(np.array([[0, 0, 0], [1, 2, 2]]), 3))
        self.assertFalse(policy.within_limit(np.array([[0, 0, 0], [1, 2, 3]]), 3))
        self.assertEqual(policy.voxelize(np.array([[-.01, 0, 0], [.01, 0, 0]])),
                         frozenset([(-1, 0, 0), (0, 0, 0)]))

    def test_explicit_reconstruction_boundary_never_compares_grids(self):
        mem = self.memory()
        self.add(mem, 0, [0])
        mem.begin_reference(reference='independent', policy=SyntheticPolicy())
        self.assertTrue(mem.scenes[0].closed)
        mem.add(camera=camera(1, 'independent'), reliable_points=np.array([[0, 0, 0], [1, 1, 1]]),
                occupied_voxels=frozenset([0]), observations=(), available_through=1)
        self.assertEqual(len(mem.scenes), 2)

    def test_reliable_points_are_deterministic_and_insufficient_is_empty(self):
        y, x = np.mgrid[:8, :8]
        points = np.stack([x, y, np.ones_like(x)], axis=-1).astype(float)
        conf = np.ones((8, 8))
        points[0, 0] = np.nan
        args = dict(confidence_quantile=.5, grid_stride=2, max_points=8, min_points=4)
        a, report = reliable_scene_points(points, conf, **args)
        b, _ = reliable_scene_points(points, conf, **args)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(len(a), 8)
        self.assertEqual(report['status'], 'reliable')
        self.assertGreater(scale_fixed_voxel_size(a, diagonal_fraction=.02), 0)
        args['min_points'] = 8
        conf[:] = 0
        self.assertEqual(len(reliable_scene_points(points, conf, **args)[0]), 0)

    def test_temporal_seeds_include_non_target_middle_then_cover_then_diversify(self):
        ids = tuple(range(9))
        features = {i: np.array([np.cos(i), np.sin(i)]) for i in ids}
        result = select_continuous_frames(ids, available_through=8, maximum=6,
            temporal_fraction=.5, relevant_instance_ids={7}, actual_frame_instances={1: {7}},
            descriptors=features)
        self.assertEqual([r['frame_id'] for r in result['choices'][:3]], [0, 4, 8])
        self.assertEqual(result['choices'][3]['stage'], 'instance_coverage')
        self.assertEqual(result['choices'][3]['frame_id'], 1)
        self.assertTrue(all(r['stage'] == 'visual_diversity' for r in result['choices'][4:]))
        self.assertEqual(len(result['frame_ids']), 6)
        self.assertEqual(result['uncovered_instance_ids'], [])

    def test_selector_rejects_future_and_missing_descriptors(self):
        args = dict(maximum=2, temporal_fraction=.5, relevant_instance_ids=set(),
                    actual_frame_instances={}, descriptors={})
        with self.assertRaisesRegex(ValueError, 'prefix'):
            select_continuous_frames((0, 1, 2), available_through=1, **args)
        with self.assertRaisesRegex(ValueError, 'descriptors'):
            select_continuous_frames((0, 1, 2), available_through=2, **args)


if __name__ == '__main__':
    unittest.main()
