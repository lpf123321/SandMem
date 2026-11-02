"""Remove only unselected instance evidence; preserve all local references."""
from copy import deepcopy
from sandbox_vlm.qa.evidence_binding import selected_catalog


def retrieve(payload, geometry, selection):
    selected = selected_catalog(payload, selection)
    keep = set(selection['relevant_ids'])
    result = deepcopy(geometry)
    base = result.get('original_D105', result)
    for extent in base['sandbox']['spatial_extents']:
        extent['instances'] = [o for o in extent['instances'] if o['global_instance_id'] in keep]
        for view in extent['viewpoints']:
            view['observed_instance_ids'] = [i for i in view['observed_instance_ids'] if i in keep]
    if 'box_sources' in base:
        base['box_sources']['rows'] = [r for r in base['box_sources']['rows'] if r[0] in keep]
    for panel in result.get('added_panels', []):
        panel['observations'] = [o for o in panel['observations'] if o['instance_id'] in keep]
    if 'added_ids' in result:
        result['added_ids'] = [i for i in result['added_ids'] if i in keep]
    # Keep empty panels, bounds and cameras: retrieval must not invent a new layout.
    return selected, result


def render_retrieved(geometry):
    if 'original_D105' in geometry:
        from sandbox_vlm.qa.complete_sandbox_repair import render_repaired
        return render_repaired(geometry)
    from sandbox_vlm.memory import Sandbox
    from sandbox_vlm.qa.complete_sandbox import render_complete
    return render_complete(Sandbox.from_dict(geometry['sandbox']))[0]
