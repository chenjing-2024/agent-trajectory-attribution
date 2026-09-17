"""Validate reusable AgentDojo components without changing content or IDs."""


def is_prepared(data):
    return isinstance(data, dict) and any(
        key in data for key in ('trajectory', 'components_metadata', 'component_scheme')
    )


def validate_prepared(data, workflow):
    if not isinstance(data.get('data_id'), str) or not data['data_id'].strip():
        raise ValueError('Prepared input requires a nonempty data_id')
    trajectory, metadata = data.get('trajectory'), data.get('components_metadata')
    if not isinstance(trajectory, list) or not trajectory:
        raise ValueError('Prepared input requires nonempty trajectory')
    if not isinstance(metadata, list) or len(metadata) != len(trajectory):
        raise ValueError('Prepared components_metadata must match trajectory length')
    if type(data.get('num_components')) is not int or data['num_components'] != len(trajectory):
        raise ValueError('Prepared num_components disagrees with trajectory')
    for number, (component, meta) in enumerate(zip(trajectory, metadata), 1):
        if not isinstance(component, dict) or component.get('role') not in {'system', 'user', 'assistant', 'tool'} or not isinstance(component.get('content'), str):
            raise ValueError('Invalid prepared component role/content')
        if not isinstance(meta, dict) or type(meta.get('component_id')) is not int or meta['component_id'] != number:
            raise ValueError('Prepared metadata IDs must be ordered 1..N; IDs will not be reassigned')
        if not isinstance(meta.get('component_type'), str) or type(meta.get('original_step_id')) is not int:
            raise ValueError('Missing prepared component type or original step provenance')
    if workflow == 'unsafe':
        if data.get('component_scheme') != 'action_result_pair_schemeA':
            raise ValueError('Unsafe requires action_result_pair_schemeA components')
        forward, backward = data.get('component_to_group_id'), data.get('group_to_component_ids')
        if not isinstance(forward, dict) or not isinstance(backward, dict):
            raise ValueError('Unsafe requires message-to-group mappings')
        expected_forward, expected_backward = {}, {}
        for number, meta in enumerate(metadata, 1):
            ids = meta.get('original_component_ids')
            if meta.get('group_id') != number or not isinstance(ids, list) or not ids:
                raise ValueError('Invalid unsafe group metadata')
            for cid in ids:
                if type(cid) is not int or cid <= 0 or str(cid) in expected_forward:
                    raise ValueError('Invalid or duplicate original component IDs')
                expected_forward[str(cid)] = number
            expected_backward[str(number)] = ids
        if forward != expected_forward or backward != expected_backward:
            raise ValueError('Prepared group mappings disagree with metadata')
    else:
        if data.get('component_scheme') is not None:
            raise ValueError('Task alignment requires its own normalized format, not unsafe grouped components')
        for meta in metadata:
            if meta['component_type'] == 'action_result_pair':
                steps = meta.get('original_step_ids')
                if not isinstance(steps, list) or len(steps) != 2 or any(type(n) is not int for n in steps) or steps[1] != steps[0] + 1 or meta['original_step_id'] != steps[0]:
                    raise ValueError('Task-alignment pair requires adjacent original_step_ids')
    return data
