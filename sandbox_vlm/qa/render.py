"""One evidence canvas with isolated coordinate panels, never a cross-panel map."""
from io import BytesIO

import numpy as np
from PIL import Image

SPACE_NOTICE = ('每个空间框内的物体和视点使用共同坐标参照。不同空间框的排布仅用于展示，'
                '不表达框间的距离、方向或邻接关系，也不直接说明它们属于不同房间。'
                '请结合框内几何、实例图例、真实帧和关联注记回答。')


def render_sandbox(sandbox, selection, *, max_side: int, include_empty_extents=False,
                   include_unlinked_viewpoints=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    if type(max_side) is not int or max_side < 1 or sandbox.validate():
        raise ValueError('invalid rendering size or sandbox')
    ids, frames = set(selection.instance_ids), set(selection.frame_ids)
    extents = [e for e in sandbox.spatial_extents if e.instance_ids() & ids
               or include_empty_extents and not e.instances]
    width, height = max_side, max(1, round(max_side * .75))
    fig = plt.figure(figsize=(width / 100, height / 100), dpi=100, facecolor='white')
    panels = []
    edges = [(a, b) for a in range(8) for b in range(a + 1, 8) if (a ^ b) in (1, 2, 4)]
    try:
        if not extents:
            fig.text(.5, .5, 'No confirmed relevant 3D structure\nConsult real frames and unresolved notes.',
                     ha='center', va='center')
        for index, extent in enumerate(extents):
            ax = fig.add_subplot(1, len(extents), index + 1, projection='3d')
            ax.set_title(f'Extent {extent.extent_id}\n{extent.coordinate_reference}', fontsize=8)
            panel_instances, panel_views, points = [], [], [extent.bounds.corners()]

            def draw_box(box, color, style='solid', alpha=1):
                corners = box.corners()
                ax.add_collection3d(Line3DCollection([(corners[a], corners[b]) for a, b in edges],
                                                     colors=color, linestyles=style, alpha=alpha, linewidths=1))
            draw_box(extent.bounds, 'grey', 'dashed', .5)
            for instance in extent.instances:
                gid = instance.global_instance_id
                if gid not in ids:
                    continue
                color = plt.get_cmap('tab10')(gid % 10)
                draw_box(instance.bbox_3d, color)
                ax.text(*instance.bbox_3d.center, f'{gid}: {instance.name}', color=color, fontsize=7)
                points.append(instance.bbox_3d.corners())
                panel_instances.append(gid)
            arrow_length = max(float(np.linalg.norm(extent.bounds.extents)) * .08, 1e-6)
            for viewpoint in extent.viewpoints:
                if viewpoint.frame_id not in frames or (not (set(viewpoint.observed_instance_ids) & ids)
                                                       and not include_unlinked_viewpoints):
                    continue
                pose = viewpoint.camera.T_ref_from_camera
                origin, forward = pose[:3, 3], pose[:3, 2] * arrow_length
                ax.scatter(*origin, color='black', s=9)
                ax.quiver(*origin, *forward, color='black', arrow_length_ratio=.2)
                ax.text(*origin, f'F{viewpoint.frame_id}', fontsize=6)
                points.append(np.array([origin, origin + forward]))
                panel_views.append(viewpoint.frame_id)
            cloud = np.concatenate(points)
            lower, upper = cloud.min(axis=0), cloud.max(axis=0)
            center = (lower + upper) / 2
            radius = max(float((upper - lower).max()) / 2, 1e-6) * 1.08
            for setter, value in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), center):
                setter(value - radius, value + radius)
            ax.set_box_aspect((1, 1, 1))
            ax.view_init(elev=25, azim=-60)
            ax.set_xlabel('X', fontsize=7); ax.set_ylabel('Y', fontsize=7); ax.set_zlabel('Z', fontsize=7)
            ax.tick_params(labelsize=5)
            panels.append(dict(extent_id=extent.extent_id, coordinate_reference=extent.coordinate_reference,
                               instance_ids=panel_instances, viewpoint_frame_ids=panel_views,
                               display_limits_only=[(center - radius).tolist(), (center + radius).tolist()]))
        fig.suptitle('Independent reference panels | arbitrary scale | dashed: observed bounds', fontsize=8)
        fig.subplots_adjust(left=.03, right=.97, bottom=.06, top=.86, wspace=.25)
        output = BytesIO()
        fig.savefig(output, format='png', dpi=100)
        output.seek(0)
        image = Image.open(output).convert('RGB').copy()
    finally:
        plt.close(fig)
    return image, dict(image_size_hw=[image.height, image.width], panels=panels,
                       panel_layout_has_physical_meaning=False, scale_is_metric=False,
                       fixed_display_view=dict(elevation=25, azimuth=-60))
