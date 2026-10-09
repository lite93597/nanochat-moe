"""CPU-only, audited last-layer single-donor upcycle of actual step3600.

The source is immutable. Fourteen destination MLP matrices and their exact
world-eight Muon momentum rows change; the existing calibrated bias is replaced.
All optimizer IDs/groups/clocks, the held router and data position are retained.
"""
import argparse
import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'cloud'))
from model_spec import MOE20_R4_ROUTING_CONTROL, MOE20_R2_BIAS_KEY as BIAS_KEY, MOE20_HELD_ROUTER_KEY

STEP = 3600
DONOR = 0
MODEL_SHA = '1315bf32725e7a1ac47e0bb826eb9d390969bbfc4a8aa24d14c02bd69ef122d4'
META_SHA = '53f1e501295b0f7bbb2325d0eb33df2303d2d699157bf0f7be9498bf2c4b2f12'
R4_CONFIG = {'moe_load_bias_rate': .5, 'moe_load_bias_layer': 19,
             'moe_load_bias_mode': 'router_logit_probability_proportional',
             'moe_load_bias_max_step': .05, 'moe_router_hold_layer': 19}
POLICY = {'kind': 'last_layer_single_donor_upcycle', 'source_step': STEP,
          'layer': 19, 'donor': DONOR,
          'selection': 'source_endpoint_global_training_gate_mass_argmax',
          'optimizer_policy': 'reset_destination_muon_momentum_slices_only'}


def audit_module():
    path = Path(__file__).with_name('repair_moe20_checkpoint.py')
    spec = importlib.util.spec_from_file_location('frozen_r1_audit_for_r5', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.STEP = STEP
    old_meta, old_optimizer = module.validate_metadata, module.validate_optimizer
    def metadata(meta):
        module.require(meta.get('model_config') == {**module.ARCHITECTURE, **R4_CONFIG},
                       'R5 source must use the actual held-router R4 geometry and recipe')
        user = meta.get('user_config', {})
        module.require(user.get('router_lr') == .0008 and all(user.get(k) == v for k, v in R4_CONFIG.items()),
                       'R5 source user recipe differs from actual R4')
        module.require(meta.get('routing_control') in (None, MOE20_R4_ROUTING_CONTROL)
                       and not meta.get('routing_guard_stop_event') and 'expert_upcycle' not in meta,
                       'R5 source must be actual un-upcycled, finite R4')
        normalized = copy.deepcopy(meta)
        normalized['model_config'] = dict(module.ARCHITECTURE)
        normalized['user_config']['router_lr'] = None
        old_meta(normalized)
    def optimizer(data, mapping, rank):
        target = mapping['target_optimizer_id']
        module.require(type(data['state'][target].get('step')) is int and data['state'][target]['step'] == 400,
                       'Actual R4 last-router AdamW clock must be400')
        normalized = {'param_groups': data['param_groups'], 'state': {k: dict(v) for k, v in data['state'].items()}}
        normalized['state'][target]['step'] = STEP
        report = old_optimizer(normalized, mapping, rank)
        report['fingerprint'] = module.optimizer_fingerprint(data)
        report['actual_target_adamw_step'] = 400
        return report
    module.validate_metadata, module.validate_optimizer = metadata, optimizer
    return module


def expert_name(expert, matrix):
    return f'transformer.h.19.mlp.experts.{expert}.{matrix}.weight'


def destination_names():
    return [expert_name(e, matrix) for e in range(8) if e != DONOR for matrix in ('c_fc', 'c_proj')]


def momentum_plan(mapping):
    """Map names through actual group order to each group's state and rank row."""
    helper = audit_module()
    plan = []
    for name in destination_names():
        param = mapping['parameters'][name]
        group = mapping['groups'][param['group']]
        helper.require(group['expected_fields']['kind'] == 'muon', 'Destination expert must be an actual Muon parameter')
        index = group['parameter_names'].index(name)
        helper.require(group['parameter_ids'][index] == param['id'], 'Ordered name/ID mapping differs')
        chunk = math.ceil(len(group['parameter_names']) / 8)
        plan.append({'parameter': name, 'parameter_id': param['id'], 'group': param['group'],
                     'state_id': group['parameter_ids'][0], 'rank': index // chunk,
                     'row': index % chunk, 'chunk_size': chunk})
    helper.require(len(plan) == 14 and len({(p['rank'], p['state_id'], p['row']) for p in plan}) == 14,
                   'Destination Muon rows must be fourteen distinct actual slices')
    return plan


def donor_evidence(diagnostics, expected_sha=None):
    helper = audit_module()
    path = Path(diagnostics).resolve()
    helper.require(path.is_file() and not path.is_symlink(), 'Actual global training diagnostics required')
    digest = helper.file_sha(path)
    helper.require(expected_sha is None or digest == expected_sha, 'Endpoint diagnostics SHA differs from failed R4 history')
    records = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    endpoint = [r for r in records if r.get('optimizer_step') == STEP and r.get('stage') == 'pretrain']
    helper.require(len(endpoint) == 1, 'Exactly one actual step3600 training endpoint is required')
    record = endpoint[0]
    helper.require(record.get('metadata', {}).get('world_size') == 8
                   and record.get('valid_loss_tokens') == 524288
                   and record.get('metadata', {}).get('routing_detail') == 'full_training_forward_counters',
                   'Donor must be selected from the complete global training window')
    layers = [layer for layer in record['routing']['layers'] if layer.get('layer') == 19]
    helper.require(len(layers) == 1 and layers[0]['token_count'] == 524288
                   and layers[0]['assignment_count'] == 1048576, 'Actual endpoint routing is incomplete')
    experts = layers[0]['experts']
    helper.require(len(experts) == 8 and [e.get('expert') for e in experts] == list(range(8)), 'Eight actual experts required')
    masses = [e['gate_share'] for e in experts]
    helper.require(all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1 for v in masses)
                   and abs(sum(masses) - 1) < 1e-5, 'Actual normalized global gate mass is invalid')
    helper.require(masses.index(max(masses)) == DONOR and sum(v == max(masses) for v in masses) == 1,
                   'Predeclared donor0 must be the unique actual training gate-mass argmax')
    return {'rule': POLICY['selection'], 'predeclared_donor': DONOR, 'endpoint_step': STEP,
            'world_size': 8, 'valid_tokens': 524288, 'gate_shares': masses,
            'official_validation_used': False, 'diagnostics_path': str(path), 'diagnostics_sha256': digest}


def verify_prior_history(manifest_path):
    helper = audit_module()
    manifest_path = Path(manifest_path).resolve()
    root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    rows = manifest.get('prior_history', {})
    required = {'source_audit', 'derived_receipt', 'calibration', 'recovery_gate', 'trial_diagnostics', 'before', 'after', 'update100'}
    helper.require(set(rows) == required, 'R5 requires the immutable actual R4 intervention and failed endpoint history')
    docs = {}
    for label, row in rows.items():
        path = (root / row['path']).resolve()
        helper.require(path.is_relative_to(root / 'outputs/moe20_repair_r4') and not path.is_symlink()
                       and helper.file_sha(path) == row['sha256'], 'Actual R4 prior-history file differs: ' + label)
        if label != 'trial_diagnostics':
            docs[label] = json.loads(path.read_text(encoding='utf-8'))
    audit, derived, calibration, gate = (docs[k] for k in ('source_audit', 'derived_receipt', 'calibration', 'recovery_gate'))
    helper.require(audit.get('status') == 'passed' and audit.get('model_only') is False and audit.get('step') == 3500
                   and derived.get('kind') == 'moe20_router_intercept_r4' and derived.get('status') == 'derived'
                   and derived.get('step') == 3500 and derived.get('routing_control') == MOE20_R4_ROUTING_CONTROL
                   and calibration.get('status') == 'passed' and calibration.get('step') == 3500,
                   'Prior history is not the actual R4 source/derive/calibration')
    helper.require(gate.get('status') == 'failed' and gate.get('start_step') == 3500 and gate.get('endpoint_step') == STEP
                   and gate.get('trial_weights_sha256') == MODEL_SHA
                   and gate.get('diagnostics_sha256') == rows['trial_diagnostics']['sha256']
                   and gate.get('derived_receipt_sha256') == rows['derived_receipt']['sha256']
                   and gate.get('fixed_validation_sha256') == {k: rows[k]['sha256'] for k in ('before', 'after', 'update100')}
                   and docs['update100'].get('checkpoint_identity', {}).get('weights_sha256') == MODEL_SHA,
                   'Actual source3600 does not match the failed R4 endpoint')
    return {'references': rows, 'source_end_step': STEP, 'endpoint_model_sha256': MODEL_SHA, 'history_replayed': False}


def audit_source(source, runtime_source, manifest_path):
    helper = audit_module()
    source = Path(source).resolve()
    manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    helper.require(manifest.get('source_model_sha256') == MODEL_SHA and manifest.get('source_meta_sha256') == META_SHA,
                   'R5 manifest must bind actual R4 step3600')
    helper.require(helper.file_sha(source / 'model_003600.pt') == MODEL_SHA
                   and helper.file_sha(source / 'meta_003600.json') == META_SHA, 'Actual source3600 weights or metadata differ')
    pins = manifest.get('r4_source_sha256', {})
    helper.require(set(pins) == helper.REQUIRED_PINS, 'Actual frozen R4 eighteen source pins required')
    with tempfile.TemporaryDirectory(prefix='r5-source-pins-') as directory:
        pin_path = Path(directory) / 'pins.json'
        helper.atomic_json(pin_path, [{'path': k, 'sha256': v} for k, v in sorted(pins.items())])
        report = helper.audit_checkpoint(source, runtime_source, pin_path)
    helper.require(report['torch_version_matches_original'] and len(report['model_tensor_hashes']) == 298,
                   'Production source audit requires actual Torch2.9.1+cu128 and298 tensors')
    plan = momentum_plan(report['mapping'])
    history = verify_prior_history(manifest_path)
    diagnostics = Path(manifest_path).resolve().parent / history['references']['trial_diagnostics']['path']
    evidence = donor_evidence(diagnostics, history['references']['trial_diagnostics']['sha256'])
    report.update(kind='r5_actual_r4_source3600', audit_helper_sha256=helper.file_sha(Path(__file__).with_name('repair_moe20_checkpoint.py')),
                  audit_adapter_sha256=helper.file_sha(__file__), target_adamw_step=400, other_adamw_step=STEP,
                  source_routing_control=dict(MOE20_R4_ROUTING_CONTROL), momentum_reset_plan=plan,
                  donor_selection=evidence, prior_history=history)
    return report


def calibration_identity(source, output_dir):
    from calibrate_moe20_router import verify_calibration_receipt
    return verify_calibration_receipt(Path(source), Path(output_dir))


def transform_optimizer(data, plan, rank):
    """Copy only destination momentum tensors before zeroing their owned rows."""
    result = {'param_groups': copy.deepcopy(data['param_groups']), 'state': {k: dict(v) for k, v in data['state'].items()}}
    cloned = set()
    for row in plan:
        if row['rank'] != rank:
            continue
        state_id = row['state_id']
        if state_id not in cloned:
            for key in ('momentum_buffer', 'second_momentum_buffer'):
                result['state'][state_id][key] = result['state'][state_id][key].clone()
            cloned.add(state_id)
        for key in ('momentum_buffer', 'second_momentum_buffer'):
            result['state'][state_id][key][row['row']].zero_()
    return result


def verify_optimizer_transform(before, after, plan, rank):
    helper = audit_module()
    helper.require(set(before) == set(after) == {'state', 'param_groups'} and before['param_groups'] == after['param_groups']
                   and set(before['state']) == set(after['state']), 'Upcycle changed optimizer groups/IDs')
    expected = transform_optimizer(before, plan, rank)
    proofs = []
    for identifier, state in before['state'].items():
        helper.require(set(state) == set(after['state'][identifier]), 'Optimizer state keys changed')
        for key, value in state.items():
            actual, allowed = after['state'][identifier][key], expected['state'][identifier][key]
            if isinstance(value, torch.Tensor):
                helper.require(helper.tensor_identity(actual) == helper.tensor_identity(allowed),
                               f'Optimizer changed outside allowed destination momentum rows: rank{rank}/ID{identifier}/{key}')
            else:
                helper.require(type(actual) is type(value) and actual == value, 'Optimizer clock or scalar field changed')
    for row in plan:
        if row['rank'] != rank:
            continue
        for key in ('momentum_buffer', 'second_momentum_buffer'):
            old = before['state'][row['state_id']][key][row['row']]
            new = after['state'][row['state_id']][key][row['row']]
            proofs.append({**row, 'field': key, 'before': helper.tensor_identity(old),
                           'after': helper.tensor_identity(new), 'exactly_zero': bool((new == 0).all())})
    return {'rank': rank, 'groups_ids_clocks_unchanged': True, 'all_non_destination_state_unchanged': True,
            'reset_slices': proofs, 'fingerprint': helper.optimizer_fingerprint(after)}


def derive(source, destination, receipt_path, runtime_source, manifest_path, calibration_dir, diagnostics=None):
    helper = audit_module()
    source, destination = Path(source).resolve(), Path(destination).resolve()
    helper.require(not destination.exists() and source != destination and source not in destination.parents,
                   'R5 destination must be a new independent directory')
    report = audit_source(source, runtime_source, manifest_path)
    if diagnostics is not None:
        helper.require(donor_evidence(diagnostics) == report['donor_selection'], 'CLI donor evidence differs from pinned R4 history')
    calibration_dir = Path(calibration_dir).resolve()
    calibration = calibration_identity(source, calibration_dir)
    bias = torch.load(calibration_dir / 'calibrated_bias.pt', map_location='cpu', weights_only=True)
    helper.require(bias.dtype == torch.float32 and tuple(bias.shape) == (8,) and torch.isfinite(bias).all().item()
                   and abs(bias.mean().item()) <= 1e-5, 'Actual calibrated FP32[8] bias required')
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=destination.name + '.staging-', dir=destination.parent))
    published = False
    try:
        names = helper.checkpoint_names()
        weights = torch.load(source / names[0], map_location='cpu', weights_only=True, mmap=True)
        for e in range(8):
            if e != DONOR:
                for matrix in ('c_fc', 'c_proj'):
                    weights[expert_name(e, matrix)] = weights[expert_name(DONOR, matrix)].clone()
        weights[BIAS_KEY] = bias.clone()
        torch.save(weights, staging / names[0])
        del weights
        meta = copy.deepcopy(report['metadata'])
        meta['expert_upcycle'] = dict(POLICY)
        helper.atomic_json(staging / names[1], meta)
        ranks = []
        for rank, name in enumerate(names[2:]):
            before = torch.load(source / name, map_location='cpu', weights_only=True, mmap=True)
            if any(row['rank'] == rank for row in report['momentum_reset_plan']):
                after = transform_optimizer(before, report['momentum_reset_plan'], rank)
                torch.save(after, staging / name)
                del after
            else:
                os.link(source / name, staging / name)
            after = torch.load(staging / name, map_location='cpu', weights_only=True, mmap=True)
            ranks.append(verify_optimizer_transform(before, after, report['momentum_reset_plan'], rank))
            del before, after
        saved = torch.load(staging / names[0], map_location='cpu', weights_only=True, mmap=True)
        fingerprints = {name: helper.tensor_identity(t) for name, t in saved.items()}
        changed = set(destination_names()) | {BIAS_KEY}
        helper.require(set(saved) == set(report['model_tensor_hashes'])
                       and {n: t for n, t in fingerprints.items() if n not in changed}
                       == {n: t for n, t in report['model_tensor_hashes'].items() if n not in changed},
                       'Upcycle changed outside fourteen destination matrices and existing bias')
        for matrix in ('c_fc', 'c_proj'):
            values = [saved[expert_name(e, matrix)] for e in range(8)]
            helper.require(all(torch.equal(v, values[DONOR]) for v in values)
                           and len({v.data_ptr() for v in values}) == 8, 'Experts are unequal or share parameter storage')
        del saved
        helper.require(all(helper.file_identity(source / name) == row for name, row in report['source_files'].items()),
                       'Immutable source3600 changed during upcycle')
        receipt = {'format': 1, 'kind': 'moe20_expert_upcycle_r5', 'status': 'derived', 'world_size': 8, 'step': STEP,
            'horizon': 6641, 'source': str(source), 'destination': str(destination),
            'project_root': str(Path(manifest_path).resolve().parent), 'source_audit': report,
            'source_runtime': str(Path(runtime_source).resolve()),
            'source_files': report['source_files'], 'derived_files': {n: helper.file_identity(staging / n) for n in names},
            'routing_control': dict(MOE20_R4_ROUTING_CONTROL), 'expert_upcycle': {**POLICY, 'donor_selection': report['donor_selection'],
                'changed_weight_names': destination_names(), 'momentum_reset_plan': report['momentum_reset_plan']},
            'optimizer_ranks': ranks, 'model_tensor_fingerprints': fingerprints, 'derived_metadata': meta,
            'prior_history': report['prior_history'], 'router_hold_source': {'step': STEP, 'weights_sha256': MODEL_SHA,
                'router_tensor_identity': report['model_tensor_hashes'][MOE20_HELD_ROUTER_KEY]},
            'calibration': {'directory': str(calibration_dir), 'receipt_sha256': helper.file_sha(calibration_dir / 'calibration_receipt.json'),
                'content': calibration, 'bias_file_sha256': helper.file_sha(calibration_dir / 'calibrated_bias.pt'),
                'bias_tensor_identity': helper.tensor_identity(bias)},
            'proof': {'allowed_changed_model_tensors': sorted(changed), 'remaining283_tensors_unchanged': True,
                'all_eight_experts_equal_donor_at_derivation': True, 'eight_independent_parameter_slots': True,
                'optimizer_changes': 'only destination expert Muon momentum rows reset', 'source_unchanged': True,
                'data_cursor_and_clocks_unchanged': True, 'metadata_changes': ['expert_upcycle'],
                'calibration_train_only': True}, 'tool_sha256': helper.file_sha(__file__), 'limitations': report['data_resume_limitations']}
        helper.atomic_json(staging / 'repair_receipt.json', receipt)
        verify_derived_receipt(staging, staging / 'repair_receipt.json', check_destination=False)
        os.rename(staging, destination)
        published = True
        helper.atomic_json(receipt_path, receipt)
        return receipt
    finally:
        if not published and staging.exists():
            helper.require(staging.resolve().parent == destination.parent and staging.name.startswith(destination.name + '.staging-')
                           and not staging.is_symlink() and source != staging.resolve() and source not in staging.resolve().parents,
                           'Cleanup escaped R5 staging directory')
            shutil.rmtree(staging)


def verify_derived_receipt(directory, receipt_path, check_destination=True):
    helper = audit_module()
    directory = Path(directory).resolve()
    receipt = json.loads(Path(receipt_path).read_text(encoding='utf-8'))
    helper.require(receipt.get('kind') == 'moe20_expert_upcycle_r5' and receipt.get('status') == 'derived'
                   and receipt.get('world_size') == 8 and receipt.get('step') == STEP and receipt.get('horizon') == 6641
                   and receipt.get('routing_control') == MOE20_R4_ROUTING_CONTROL, 'Invalid exact R5 identity')
    helper.require(not check_destination or Path(receipt['destination']).resolve() == directory, 'R5 directory differs')
    audit = receipt['source_audit']
    helper.require(audit.get('status') == 'passed' and audit.get('model_only') is False
                   and audit.get('torch_version_matches_original') is True and len(audit.get('optimizer_ranks', [])) == 8
                   and audit.get('missing_optimizer_shards') == [] and audit.get('target_adamw_step') == 400
                   and audit.get('other_adamw_step') == STEP, 'Real complete actual3600 source audit required')
    helper.require(audit.get('audit_helper_sha256') == helper.file_sha(Path(__file__).with_name('repair_moe20_checkpoint.py'))
                   and audit.get('audit_adapter_sha256') == helper.file_sha(__file__), 'Source audit producer differs')
    helper.require(receipt['source_files'] == audit['source_files']
                   and receipt['source_files']['model_003600.pt']['sha256'] == MODEL_SHA
                   and receipt['source_files']['meta_003600.json']['sha256'] == META_SHA, 'R5 actual source differs')
    source = Path(receipt['source'])
    source_meta = json.loads((source / 'meta_003600.json').read_text(encoding='utf-8'))
    helper.validate_metadata(source_meta)
    helper.require(source_meta == audit['metadata'], 'Source audit metadata differs from actual source')
    runtime = Path(receipt['source_runtime'])
    helper.require(set(audit['source_pins']) == helper.REQUIRED_PINS
                   and all(helper.file_identity(runtime / name) == identity for name, identity in audit['source_pins'].items()),
                   'Source runtime pins changed')
    actual_mapping = helper.reconstruct_mapping(runtime, source_meta)
    helper.require(json.loads(json.dumps(actual_mapping)) == audit['mapping'], 'Actual ordered runtime mapping differs')
    names = helper.checkpoint_names()
    helper.require(set(receipt['derived_files']) == set(names), 'All ten derived files required')
    for name in names:
        helper.require(helper.file_identity(source / name) == receipt['source_files'][name]
                       and helper.file_identity(directory / name) == receipt['derived_files'][name], 'Source or derived actual file changed: ' + name)
    plan = momentum_plan(audit['mapping'])
    evidence = donor_evidence(audit['donor_selection']['diagnostics_path'], audit['donor_selection']['diagnostics_sha256'])
    helper.require(evidence == audit['donor_selection'] and receipt['expert_upcycle'] == {**POLICY,
                   'donor_selection': evidence, 'changed_weight_names': destination_names(), 'momentum_reset_plan': plan},
                   'R5 donor policy or exact momentum mapping differs')
    ranks = []
    for rank, name in enumerate(names[2:]):
        before = torch.load(source / name, map_location='cpu', weights_only=True, mmap=True)
        after = torch.load(directory / name, map_location='cpu', weights_only=True, mmap=True)
        ranks.append(verify_optimizer_transform(before, after, plan, rank))
        if not any(row['rank'] == rank for row in plan):
            helper.require(receipt['derived_files'][name] == receipt['source_files'][name], 'Unchanged optimizer rank bytes differ')
        del before, after
    helper.require(json.loads(json.dumps(ranks)) == receipt['optimizer_ranks'], 'Actual derived momentum slice proof differs')
    saved = torch.load(directory / names[0], map_location='cpu', weights_only=True, mmap=True)
    fingerprints = {n: helper.tensor_identity(t) for n, t in saved.items()}
    allowed = set(destination_names()) | {BIAS_KEY}
    helper.require(fingerprints == receipt['model_tensor_fingerprints'] and set(fingerprints) == set(audit['model_tensor_hashes'])
                   and {n: t for n, t in fingerprints.items() if n not in allowed}
                   == {n: t for n, t in audit['model_tensor_hashes'].items() if n not in allowed}, 'R5 model transform exceeded allowlist')
    for matrix in ('c_fc', 'c_proj'):
        values = [saved[expert_name(e, matrix)] for e in range(8)]
        helper.require(all(torch.equal(t, values[DONOR]) for t in values) and len({t.data_ptr() for t in values}) == 8,
                       'Derived expert weights must equal donor in independent storage')
    del saved
    calibration = receipt['calibration']
    caldir = Path(calibration['directory'])
    helper.require(helper.file_sha(caldir / 'calibration_receipt.json') == calibration['receipt_sha256']
                   and calibration_identity(source, caldir) == calibration['content']
                   and helper.file_sha(caldir / 'calibrated_bias.pt') == calibration['bias_file_sha256']
                   and fingerprints[BIAS_KEY] == calibration['bias_tensor_identity'], 'Actual calibrated intercept evidence differs')
    expected_meta = copy.deepcopy(audit['metadata'])
    expected_meta['expert_upcycle'] = dict(POLICY)
    helper.require(receipt['derived_metadata'] == expected_meta
                   and json.loads((directory / names[1]).read_text(encoding='utf-8')) == expected_meta,
                   'R5 metadata changes exceed explicit upcycle provenance')
    helper.require(receipt.get('router_hold_source') == {'step': STEP, 'weights_sha256': MODEL_SHA,
                   'router_tensor_identity': audit['model_tensor_hashes'][MOE20_HELD_ROUTER_KEY]}
                   and receipt.get('tool_sha256') == helper.file_sha(__file__), 'Held router or producer differs')
    helper.require(receipt['prior_history'] == audit['prior_history'], 'R5 prior actual R4 history differs')
    root = Path(receipt['project_root'])
    for row in receipt['prior_history']['references'].values():
        path = (root / row['path']).resolve()
        helper.require(path.is_relative_to(root.resolve() / 'outputs/moe20_repair_r4')
                       and helper.file_sha(path) == row['sha256'], 'Immutable prior R4 history changed')
    expected_proof = {'allowed_changed_model_tensors': sorted(allowed), 'remaining283_tensors_unchanged': True,
        'all_eight_experts_equal_donor_at_derivation': True, 'eight_independent_parameter_slots': True,
        'optimizer_changes': 'only destination expert Muon momentum rows reset', 'source_unchanged': True,
        'data_cursor_and_clocks_unchanged': True, 'metadata_changes': ['expert_upcycle'], 'calibration_train_only': True}
    helper.require(receipt.get('proof') == expected_proof, 'R5 transformation proof differs')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('audit', 'apply'))
    for flag in ('source', 'runtime-source', 'source-pins'):
        parser.add_argument('--' + flag, type=Path, required=True)
    for flag in ('output', 'destination', 'receipt', 'calibration-dir', 'diagnostics'):
        parser.add_argument('--' + flag, type=Path)
    args = parser.parse_args()
    helper = audit_module()
    if args.command == 'audit':
        helper.require(args.output is not None, 'Audit output required')
        helper.atomic_json(args.output, audit_source(args.source, args.runtime_source, args.source_pins))
    else:
        helper.require(args.destination is not None and args.receipt is not None and args.calibration_dir is not None,
                       'Apply requires a new destination, receipt output and actual calibration')
        derive(args.source, args.destination, args.receipt, args.runtime_source, args.source_pins, args.calibration_dir, args.diagnostics)


if __name__ == '__main__':
    main()
