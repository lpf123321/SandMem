"""Actual 3D Scene bounds, source-backed boxes, and intermediate camera path."""
from io import BytesIO
import numpy as np
from PIL import Image

from sandbox_vlm.memory.geometry import BBox3D


def render_continuous(memory, prepared, *, max_side):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    frames = set(prepared['selection']['frame_ids'])
    references = sorted({r['camera'].reference for r in memory.frames.values()})
    height = round(max_side * .75)
    fig = plt.figure(figsize=(max_side / 100, height / 100), dpi=100)
    panels = []
    edges = [(a, b) for a in range(8) for b in range(a + 1, 8) if (a ^ b) in (1, 2, 4)]
    try:
        for index, reference in enumerate(references):
            ax = fig.add_subplot(1, len(references), index + 1, projection='3d')
            clouds, scenes, plotted = [], [], []

            def box_lines(box, color, alpha=1, style='solid'):
                corners = box.corners()
                clouds.append(corners)
                ax.add_collection3d(Line3DCollection([(corners[a], corners[b]) for a, b in edges],
                    colors=color, alpha=alpha, linestyles=style, linewidths=.8))

            for scene in memory.scenes.scenes if memory.scenes else []:
                if scene.reference != reference:
                    continue
                lower, upper = scene.bounds
                box_lines(BBox3D((lower + upper) / 2, upper - lower, np.eye(3), reference),
                          'grey', .3, 'dashed')
                scenes.append(scene.scene_id)
            for row in prepared['payload']['geometry']:
                if row['reference'] != reference:
                    continue
                box = BBox3D.from_dict(row['bbox'])
                gid = row['instance_id']
                color = plt.get_cmap('tab10')(gid % 10)
                box_lines(box, color)
                ax.text(*box.center, f'{gid}: {memory.instances[gid]["name"]}', fontsize=6, color=color)
                plotted.append(gid)
            cameras = [(fid, row['camera']) for fid, row in memory.frames.items()
                       if row['camera'].reference == reference]
            positions = np.array([cam.T_ref_from_camera[:3, 3] for _, cam in cameras])
            if len(positions):
                ax.plot(*positions.T, color='steelblue', linewidth=1, alpha=.7)
                clouds.append(positions)
            for fid, camera in cameras:
                if fid in frames:
                    pos = camera.T_ref_from_camera[:3, 3]
                    ax.scatter(*pos, color='black', s=8)
                    ax.text(*pos, f'F{fid}', fontsize=6)
                    scale = max(float(np.ptp(positions, axis=0).max()) * .03, .01)
                    forward = camera.T_ref_from_camera[:3, 2] * scale
                    ax.quiver(*pos, *forward, color='black', arrow_length_ratio=.2)
            all_points = np.concatenate(clouds) if clouds else np.zeros((1, 3))
            lower, upper = all_points.min(0), all_points.max(0)
            center = (lower + upper) / 2
            radius = max(float((upper - lower).max()) * .55, 1e-6)
            for setter, coordinate in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), center):
                setter(coordinate - radius, coordinate + radius)
            ax.set_box_aspect((1, 1, 1))
            ax.set_title(reference + '\nShared coordinates across Scenes', fontsize=8)
            ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
            ax.tick_params(labelsize=5)
            panels.append(dict(reference=reference, scene_ids=scenes, plotted_ids=plotted,
                               trajectory_frame_ids=[fid for fid, _ in cameras],
                               labelled_frame_ids=[fid for fid, _ in cameras if fid in frames]))
        fig.suptitle('Estimated 3D sandbox | arbitrary units | dashed: Scene bounds | blue: camera path', fontsize=8)
        fig.subplots_adjust(left=.03, right=.97, bottom=.05, top=.88)
        output = BytesIO()
        fig.savefig(output, format='png', dpi=100)
        output.seek(0)
        diagram = Image.open(output).convert('RGB').copy()
    finally:
        plt.close(fig)
    return diagram, dict(image_size_hw=[height, max_side], panels=panels,
                         underlying_coordinates_unchanged=True, scale_is_metric=False)
