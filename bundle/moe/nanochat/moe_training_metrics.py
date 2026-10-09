"""Local, token-weighted MoE training records, independent of WandB.

Only logged steps enable detailed routing counters. SFT also retains the most
recent step's existing selection counters, because epoch end is known only
after prefetching. No collective runs during gradient accumulation.
"""

import json
import hashlib
import importlib.util
import math
import os
import time

import torch
import torch.distributed as dist

from nanochat.moe_report import collect_moe_routing_report
from nanochat.gpt import moe_load_bias_recipe


class MoETrainingMetrics:
    def __init__(self, model, path, stage, interval, retain_final=False, diagnostics=False, guard_path=None):
        self.model, self.path, self.stage = model, os.fspath(path), stage
        self.interval, self.retain_final = interval, retain_final
        self.enabled = model.config.moe_num_experts > 0
        self.step, self.last_written = None, None
        self.selected, self.packet, self.routes = False, None, {}
        self.context = {}
        self.diagnostics, self.guard = diagnostics, None
        self.forward_diagnostics, self.optimizer_diagnostics = {}, None
        self.guard_stop_event = None
        self.loss_finite = None
        self.load_bias_feedback = []
        self.router_hold = None
        if guard_path:
            spec = importlib.util.spec_from_file_location('nanochat_runtime_routing_guard', guard_path)
            module = importlib.util.module_from_spec(spec)
            import sys
            sys.path.insert(0, os.path.dirname(os.path.abspath(guard_path)))
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            from dataclasses import asdict
            self.guard = module.RoutingGuard(architecture={key: value for key, value in asdict(model.config).items()
                if key not in ('moe_aux_loss_coef', 'moe_load_bias_rate', 'moe_load_bias_layer',
                               'moe_load_bias_mode', 'moe_load_bias_max_step', 'moe_router_hold_layer')}, snapshot_interval=interval,
                                             require_router_diagnostics=True)

    def begin(self, step, force=False):
        self.step = step
        self.selected = self.enabled and (force or step == 1 or step % self.interval == 0)
        self.packet, self.routes, self.context = None, {}, {}
        self.forward_diagnostics, self.optimizer_diagnostics = {}, None
        self.loss_finite = None
        self.load_bias_feedback = []
        self.router_hold = None
        if self.diagnostics:
            self.model.set_moe_router_diagnostics(self.selected)
        if self.selected:
            self.model.start_moe_stats(reset=True)

    def observe(self, loss, targets):
        if self.diagnostics:
            finite = torch.isfinite(loss.detach()).all()
            self.loss_finite = finite if self.loss_finite is None else self.loss_finite & finite
        if not self.enabled or not (self.selected or self.retain_final):
            return
        count = (targets != -1).sum().to(torch.float64)
        auxiliary = self.model.last_moe_aux_loss.detach().to(torch.float64)
        objective = loss.detach().to(torch.float64)
        ce = objective - self.model.config.moe_aux_loss_coef * auxiliary
        packet = torch.stack((ce * count, auxiliary * count, count, objective,
                              torch.ones_like(count)))
        self.packet = packet if self.packet is None else self.packet + packet
        if self.diagnostics and self.selected:
            for layer, stats in self.model.get_moe_router_diagnostics().items():
                value = torch.stack([stats[name].detach().to(torch.float64) for name in
                    ('logit_min', 'logit_max', 'prob_min', 'prob_max', 'logits_finite', 'probabilities_finite')])
                if layer in self.forward_diagnostics:
                    previous = self.forward_diagnostics[layer]
                    value = torch.stack([torch.minimum(previous[i], value[i]) if i in (0, 2, 4, 5)
                                         else torch.maximum(previous[i], value[i]) for i in range(6)])
                self.forward_diagnostics[layer] = value
        if self.retain_final and not self.selected:
            for layer, stats in self.model.get_moe_stats().items():
                n = stats['token_count'].detach().to(torch.float64)
                counters = torch.cat((n.reshape(1), stats['expert_counts'].detach().to(torch.float64),
                                      stats['mean_router_probs'].detach().to(torch.float64) * n))
                self.routes[layer] = counters if layer not in self.routes else self.routes[layer] + counters

    def capture_load_bias_feedback(self, reports):
        self.load_bias_feedback = reports

    def capture_router_hold(self, report):
        self.router_hold = report

    def capture_optimizer_diagnostics(self, optimizer):
        """Call after unscale and before optimizer.step/zero_grad on selected updates."""
        if not self.diagnostics or not self.selected:
            return
        routers = {i: block.mlp.router.weight for i, block in enumerate(self.model.transformer.h)
                   if hasattr(block.mlp, 'router')}
        layers = {}
        for i, parameter in routers.items():
            gradient = parameter.grad
            layers[i] = torch.stack((torch.isfinite(parameter).all().double(),
                torch.isfinite(gradient).all().double() if gradient is not None else parameter.new_zeros((), dtype=torch.float64),
                parameter.detach().double().norm(), gradient.detach().double().norm() if gradient is not None
                    else parameter.new_zeros((), dtype=torch.float64)))
        self.optimizer_diagnostics = {'layers': layers,
            'all_weights_finite': tensors_finite(self.model.parameters()),
            'all_gradients_finite': tensors_finite(p.grad for p in self.model.parameters() if p.grad is not None),
            'router_initial_lrs': [float(group['initial_lr']) for group in router_optimizer_groups(optimizer, self.model)]}

    def _diagnostic_report(self, distributed):
        if self.optimizer_diagnostics is None:
            raise RuntimeError('Selected router diagnostic update lacks pre-optimizer gradient evidence')
        result = self.optimizer_diagnostics
        flags = torch.stack((result['all_weights_finite'], result['all_gradients_finite'],
                             tensors_finite(self.model.parameters()))).double()
        if distributed:
            dist.all_reduce(flags, op=dist.ReduceOp.MIN)
        layers = []
        for layer, forward in sorted(self.forward_diagnostics.items()):
            forward = forward.clone()
            minimum = forward[[0, 2, 4, 5]].clone()
            maximum = forward[[1, 3]].clone()
            weight = result['layers'][layer].clone()
            finite, norms = weight[:2].clone(), weight[2:].clone()
            if distributed:
                dist.all_reduce(minimum, op=dist.ReduceOp.MIN)
                dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
                dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                dist.all_reduce(norms, op=dist.ReduceOp.MAX)
            values = [*minimum.tolist(), *maximum.tolist(), *norms.tolist()]
            def safe(value):
                return value if math.isfinite(value) else None
            layers.append({'layer': layer, 'logit_min': safe(values[0]), 'prob_min': safe(values[1]),
                'logits_finite': values[2] == 1, 'probabilities_finite': values[3] == 1,
                'logit_max': safe(values[4]), 'prob_max': safe(values[5]),
                'weights_finite': finite[0].item() == 1, 'gradients_finite': finite[1].item() == 1,
                'router_weight_norm': safe(values[6]), 'router_gradient_norm': safe(values[7])})
        return {'aggregation': 'all ranks and all gradient accumulation microbatches in this optimizer step',
            'norm_scope': 'maximum per-rank L2 norm; weights before update, gradients before optimizer',
            'all_weights_finite': flags[0].item() == 1 and flags[2].item() == 1,
            'all_gradients_finite': flags[1].item() == 1,
            'router_initial_lrs': result['router_initial_lrs'], 'layers': layers}

    def finish(self, force=False, **context):
        """All ranks call together; rank zero appends one JSON line."""
        self.context.update(context)
        if self.diagnostics and self.loss_finite is not None and self.last_written != self.step:
            finite = self.loss_finite.detach().double()
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if finite.item() != 1:
                self.guard_stop_event = {'decision': 'stop', 'step': self.step,
                    'issues': ['Nonfinite forward objective in the complete global update'],
                    'forward_objective_finite': False}
                if self.packet is None:
                    record = {'schema_version': 1, 'stage': self.stage, 'optimizer_step': self.step,
                        'recorded_at_epoch': time.time(), 'invalid_nonfinite_evidence': True,
                        'guard_stop_event': self.guard_stop_event,
                        'routing_unavailable': 'This nonfinite update was outside the routing snapshot schedule'}
                    if not dist.is_initialized() or dist.get_rank() == 0:
                        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
                        with open(self.path, 'a', encoding='utf-8') as stream:
                            stream.write(json.dumps(record, allow_nan=False) + '\n')
                    self.last_written = self.step
                    return record
        if not self.enabled or self.packet is None or self.last_written == self.step:
            return None
        if not (self.selected or force or self.guard_stop_event is not None):
            return None
        distributed = dist.is_available() and dist.is_initialized()
        layers = sorted(self.routes)
        packet = torch.cat((self.packet, *(self.routes[layer] for layer in layers))) if layers else self.packet.clone()
        if distributed:
            dist.all_reduce(packet, op=dist.ReduceOp.SUM)
        values = packet[:5].tolist()
        ce_sum, aux_sum, valid_tokens, objective_sum, microbatches = values
        denominator = max(valid_tokens, 1)
        if self.selected:
            routing = collect_moe_routing_report(self.model, f'{self.stage}_step_{self.step}', distributed=distributed)
            self.model.stop_moe_stats()
            detail = 'full_training_forward_counters'
        else:
            routing = self._selection_report(packet[5:], layers)
            detail = 'selection_counts_and_router_probabilities'
        record = {
            'schema_version': 1, 'stage': self.stage, 'optimizer_step': self.step,
            'recorded_at_epoch': time.time(), 'valid_loss_tokens': int(valid_tokens),
            'rank_microbatches': int(microbatches),
            'ce_loss_token_weighted': ce_sum / denominator,
            'aux_loss_token_weighted': aux_sum / denominator,
            'total_loss_token_weighted': (ce_sum + self.model.config.moe_aux_loss_coef * aux_sum) / denominator,
            'optimizer_objective_rank_microbatch_mean': objective_sum / max(microbatches, 1),
            'metadata': {
                'scope': 'supervised_loss_tokens', 'mask': 'targets != -1',
                'aggregation': 'all ranks and all gradient accumulation microbatches in this optimizer step',
                'aux_loss_definition': 'mean per-layer Switch load-balancing loss; no z-loss',
                'routing_detail': detail, 'interval_optimizer_steps': self.interval,
                'world_size': dist.get_world_size() if distributed else 1,
            },
            'routing': routing, **self.context,
        }
        if self.diagnostics and self.selected:
            record['router_diagnostics'] = self._diagnostic_report(distributed)
            self.model.set_moe_router_diagnostics(False)
        if self.model.config.moe_load_bias_rate > 0:
            record['metadata']['load_bias_recipe'] = moe_load_bias_recipe(self.model.config)
            record['load_bias_feedback'] = self.load_bias_feedback
        if self.model.config.moe_router_hold_layer >= 0:
            if self.router_hold is None:
                raise RuntimeError('Held router record has no actual optimizer-step proof')
            record['router_hold'] = self.router_hold
        nonfinite = nonfinite_paths(record)
        diagnostic = record.get('router_diagnostics', {})
        bad_flags = [name for name in ('all_weights_finite', 'all_gradients_finite')
                     if diagnostic.get(name) is False]
        bad_flags += [f"layer_{layer['layer']}.{name}" for layer in diagnostic.get('layers', [])
            for name in ('logits_finite', 'probabilities_finite', 'weights_finite', 'gradients_finite')
            if layer.get(name) is False]
        # Audit raw values before sanitizing a fault record for JSON. Never turn
        # an invalid record into a finite report or lose the save-current-step path.
        if self.guard and self.selected:
            decision = self.guard.observe(record)
            if decision['decision'] == 'stop':
                self.guard_stop_event = decision
        if nonfinite or bad_flags:
            self.guard_stop_event = self.guard_stop_event or {'decision': 'stop', 'step': self.step,
                'issues': ['Nonfinite training evidence'], 'nonfinite_paths': nonfinite, 'finite_flags_false': bad_flags}
            record['invalid_nonfinite_evidence'] = {'nonfinite_paths': nonfinite, 'finite_flags_false': bad_flags}
        if self.guard_stop_event:
            record['guard_stop_event'] = sanitize_nonfinite(self.guard_stop_event)
            record = sanitize_nonfinite(record)
        if not distributed or dist.get_rank() == 0:
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
            with open(self.path, 'a', encoding='utf-8') as handle:
                handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
        self.last_written = self.step
        return record

    def _selection_report(self, packet, layers):
        experts = self.model.config.moe_num_experts
        top_k = self.model.config.moe_top_k
        reports = []
        width = 1 + 2 * experts
        for i, layer in enumerate(layers):
            values = packet[i * width:(i + 1) * width].tolist()
            tokens, counts, probabilities = int(values[0]), values[1:1 + experts], values[1 + experts:]
            assignments = sum(counts)
            assert int(assignments) == tokens * top_k
            mean = assignments / experts
            reports.append({
                'layer': layer, 'token_count': tokens, 'assignment_count': int(assignments), 'top_k': top_k,
                'load_cv': math.sqrt(sum((n - mean) ** 2 for n in counts) / experts) / mean if mean else 0.0,
                'unused_experts': [j for j, n in enumerate(counts) if n == 0],
                'experts': [{'expert': j, 'selections': int(n),
                             'assignment_share': n / assignments if assignments else 0.0,
                             'token_hit_rate': n / tokens if tokens else 0.0,
                             'mean_router_probability': probabilities[j] / tokens if tokens else 0.0}
                            for j, n in enumerate(counts)],
            })
        return {'label': f'{self.stage}_step_{self.step}', 'num_experts': experts, 'top_k': top_k, 'layers': reports}


def tensors_finite(tensors):
    tensors = list(tensors)
    if not tensors:
        raise RuntimeError('Finite diagnostic requires actual tensors')
    return torch.stack([torch.isfinite(value.detach()).all() for value in tensors]).all()


def nonfinite_paths(value, path='record'):
    if isinstance(value, dict):
        return [item for key, child in value.items() for item in nonfinite_paths(child, f'{path}.{key}')]
    if isinstance(value, (list, tuple)):
        return [item for index, child in enumerate(value) for item in nonfinite_paths(child, f'{path}[{index}]')]
    return [path] if isinstance(value, float) and not math.isfinite(value) else []


def sanitize_nonfinite(value):
    if isinstance(value, dict):
        return {key: sanitize_nonfinite(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_nonfinite(child) for child in value]
    return None if isinstance(value, float) and not math.isfinite(value) else value


def router_optimizer_groups(optimizer, model):
    routers = {id(block.mlp.router.weight) for block in model.transformer.h if hasattr(block.mlp, 'router')}
    groups = [group for group in optimizer.param_groups if any(id(parameter) in routers for parameter in group['params'])]
    if not groups or sum(len(group['params']) for group in groups) != len(routers) or any(
            id(parameter) not in routers for group in groups for parameter in group['params']):
        raise ValueError('Actual optimizer router parameter mapping is inconsistent')
    return groups


def require_router_initial_lr(optimizer, expected, model):
    """Verify loaded optimizer metadata; CLI alone cannot override a resumed LR."""
    groups = router_optimizer_groups(optimizer, model)
    if not groups or not math.isfinite(expected) or expected <= 0:
        raise ValueError('Actual router optimizer groups and a positive expected initial_lr are required')
    values = [float(group.get('initial_lr', float('nan'))) for group in groups]
    if any(not math.isfinite(value) or not math.isclose(value, expected, rel_tol=1e-10, abs_tol=1e-15) for value in values):
        raise ValueError(f'Loaded router initial_lr differs from the required intervention: {values} vs {expected}')
    return values


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def fixed_validation_bpb(model, loader, eval_steps, token_bytes, output, *, label, step,
                         checkpoint_path, tokenizer_path, token_bytes_path, architecture, source_paths,
                         device_batch_size, sequence_len, tensor_shape_evidence, compute_dtype):
    """Fingerprint the actual same validation window on all ranks, then write BPB."""
    from nanochat.loss_eval import evaluate_bpb
    digest = hashlib.sha256()
    def measured():
        for inputs, targets in loader:
            for tensor in (inputs, targets):
                value = tensor.detach().cpu().contiguous()
                digest.update(str((str(value.dtype), tuple(value.shape))).encode('ascii'))
                digest.update(value.numpy().tobytes())
            yield inputs, targets
    model.eval()
    score = evaluate_bpb(model, measured(), eval_steps, token_bytes)
    distributed = dist.is_available() and dist.is_initialized()
    world = dist.get_world_size() if distributed else 1
    hashes = [None] * world
    if distributed:
        dist.all_gather_object(hashes, digest.hexdigest())
    else:
        hashes[0] = digest.hexdigest()
    model.train()
    if tensor_shape_evidence.get('tensor_shapes_verified') is not True:
        raise RuntimeError('Fixed validation requires actual checkpoint tensor-shape verification')
    record = {'schema_version': 1, 'label': label, 'checkpoint_step': step, 'bpb': score,
        'window_sha256': hashlib.sha256(json.dumps(hashes, separators=(',', ':')).encode()).hexdigest(),
        'rank_window_sha256': hashes, 'world_size': world, 'eval_steps': eval_steps,
        'device_batch_size': device_batch_size, 'sequence_len': sequence_len,
        'global_forward_positions': eval_steps * device_batch_size * sequence_len * world,
        'token_bytes_sha256': file_sha256(token_bytes_path),
        'checkpoint_identity': {'model_tag': 'moe_d20_e8_k2', 'step': step,
            'weights_sha256': file_sha256(checkpoint_path), 'tensor_shapes_verified': tensor_shape_evidence['tensor_shapes_verified'],
            'load_bias_recipe': moe_load_bias_recipe(model.config)},
        'evaluation_identity': {'architecture': architecture, 'tokenizer_sha256': file_sha256(tokenizer_path),
            'mask': 'targets != -1', 'dtype': str(compute_dtype), 'master_dtype': str(next(model.parameters()).dtype),
            'source_sha256': {name: file_sha256(path) for name, path in source_paths.items()}}}
    if not distributed or dist.get_rank() == 0:
        os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
        with open(output, 'w', encoding='utf-8') as stream:
            json.dump(record, stream, ensure_ascii=False, allow_nan=False, indent=2)
            stream.write('\n')
    return record
