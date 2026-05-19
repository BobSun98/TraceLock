from __future__ import annotations

import bisect
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tracelock.common.project import setup_project_root

setup_project_root(PROJECT_ROOT)

from tracelock.common.io_utils import load_jsonl, load_pt


@lru_cache(maxsize=128)
def _cached_load_pt(path_str: str) -> dict[str, Any]:
    return load_pt(path_str)


def infer_prompt_length(state_1d: torch.Tensor) -> int:
    prompt_positions = (state_1d == 0).nonzero(as_tuple=False).squeeze(-1)
    if prompt_positions.numel() == 0:
        return 0
    return int(prompt_positions.numel())


def list_sample_paths(samples_dir: str | Path) -> list[Path]:
    root = Path(samples_dir)
    if (root / 'train').is_dir() or (root / 'val').is_dir():
        return sorted(list(root.glob('train/*.pt')) + list(root.glob('val/*.pt')))
    return sorted(root.glob('*.pt'))


def list_split_sample_paths(samples_dir: str | Path, split: str) -> list[Path]:
    split_dir = Path(samples_dir) / split
    return sorted(split_dir.glob('*.pt'))


def sample_id_from_path(sample_path: str | Path) -> str:
    stem = Path(sample_path).stem.removeprefix('sample_')
    if '_step_' in stem:
        return stem.rsplit('_step_', 1)[0]
    return stem


def step_index_from_path(sample_path: str | Path) -> int:
    stem = Path(sample_path).stem
    if '_step_' not in stem:
        return 0
    return int(stem.rsplit('_step_', 1)[1])


class TraceSampleDataset(Dataset):
    def __init__(self, manifest_path: str | Path):
        self.manifest_path = Path(manifest_path)
        self.records = load_jsonl(self.manifest_path)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        payload = load_pt(record['file_path'])
        payload['_manifest'] = record
        return payload


class TracePositionDataset(Dataset):
    def __init__(self, manifest_path: str | Path, include_prompt_positions: bool = True):
        self.manifest_path = Path(manifest_path)
        self.records = load_jsonl(self.manifest_path)
        self.include_prompt_positions = include_prompt_positions

        self._cumulative = []
        total = 0
        for record in self.records:
            count = self._positions_per_sample(record)
            total += count
            self._cumulative.append(total)

    def _positions_per_sample(self, record: dict[str, Any]) -> int:
        seq_len = int(record['seq_len'])
        if self.include_prompt_positions:
            return seq_len
        return int(record['gen_len'])

    def __len__(self) -> int:
        return self._cumulative[-1] if self._cumulative else 0

    def _lookup(self, index: int) -> tuple[int, int]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        sample_idx = bisect.bisect_right(self._cumulative, index)
        start = 0 if sample_idx == 0 else self._cumulative[sample_idx - 1]
        return sample_idx, index - start

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample_idx, local_index = self._lookup(index)
        record = self.records[sample_idx]
        sample = load_pt(record['file_path'])

        state = sample['state']
        trace = sample['trace']
        label = sample['label']
        seq_len = int(trace.shape[1])
        prompt_length = infer_prompt_length(state)

        if self.include_prompt_positions:
            position_index = local_index
        else:
            position_index = prompt_length + local_index

        return {
            'hidden': trace[:, position_index],
            'label': label[position_index],
            'state': state[position_index],
            'is_prompt_position': state[position_index] == 0,
            'sample_id': record.get('sample_id', sample_id_from_path(record['file_path'])),
            'step_index': torch.tensor(record.get('step_index', step_index_from_path(record['file_path'])), dtype=torch.int32),
            'position_index': torch.tensor(position_index, dtype=torch.int32),
            'record': record,
            'seq_len': torch.tensor(seq_len, dtype=torch.int32),
        }


class TraceDirectorySampleDataset(Dataset):
    def __init__(self, samples_dir: str | Path):
        self.samples_dir = Path(samples_dir)
        self.sample_paths = list_sample_paths(self.samples_dir)

    def __len__(self) -> int:
        return len(self.sample_paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample_path = self.sample_paths[index]
        sample = _cached_load_pt(str(sample_path))
        return {
            'sample_path': str(sample_path),
            'sample_id': sample_id_from_path(sample_path),
            'step_index': torch.tensor(step_index_from_path(sample_path), dtype=torch.int32),
            'trace': sample['trace'],
            'label': sample['label'],
            'state': sample['state'],
        }


class TraceDirectoryStepDataset(Dataset):
    def __init__(self, samples_dir: str | Path | list[str | Path], sample_paths: list[str | Path] | None = None):
        if isinstance(samples_dir, list):
            if not samples_dir:
                raise ValueError('sample_paths list must not be empty when passed as the first argument.')
            self.sample_paths = [Path(path) for path in samples_dir]
            self.samples_dir = self.sample_paths[0].parent
        else:
            self.samples_dir = Path(samples_dir)
            if sample_paths is None:
                self.sample_paths = list_sample_paths(self.samples_dir)
            else:
                self.sample_paths = [Path(path) for path in sample_paths]

        if not self.sample_paths:
            raise ValueError(f'No sample files found for samples_dir={self.samples_dir}')

        first_sample = _cached_load_pt(str(self.sample_paths[0]))
        if 'x' in first_sample:
            base_hidden_layers = first_sample['x']
            if not isinstance(base_hidden_layers, torch.Tensor) or base_hidden_layers.ndim != 2:
                raise ValueError(
                    f'Expected 2D tensor `x` in {self.sample_paths[0]}, got '
                    f'{type(base_hidden_layers).__name__} with shape {getattr(base_hidden_layers, "shape", None)}'
                )
            base_feature_dim = int(base_hidden_layers.shape[-1])
        else:
            trace = first_sample['trace']
            if not isinstance(trace, torch.Tensor) or trace.ndim != 3:
                raise ValueError(
                    f'Expected 3D tensor `trace` in {self.sample_paths[0]}, got '
                    f'{type(trace).__name__} with shape {getattr(trace, "shape", None)}'
                )
            base_feature_dim = int(trace.shape[-1])

        self.use_confidence_feature = 'confidence' in first_sample
        self.base_feature_dim = base_feature_dim
        self.precomputed_input_dim = base_feature_dim + (1 if self.use_confidence_feature else 0)

    def __len__(self) -> int:
        return len(self.sample_paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample_path = self.sample_paths[index]
        sample = _cached_load_pt(str(sample_path))
        label = sample['label']
        state = sample['state']

        if 'x' in sample:
            hidden_layers = sample['x']
            seq_len = int(hidden_layers.shape[0])
        else:
            trace = sample['trace']
            hidden_layers = trace.permute(1, 0, 2).contiguous()
            seq_len = int(trace.shape[1])

        has_confidence = 'confidence' in sample
        if has_confidence != self.use_confidence_feature:
            raise ValueError(
                f'Inconsistent confidence feature presence for {sample_path}: '
                f'expected use_confidence_feature={self.use_confidence_feature}, got {has_confidence}.'
            )
        if has_confidence:
            confidence = sample['confidence']
            if not isinstance(confidence, torch.Tensor) or confidence.ndim != 1:
                raise ValueError(
                    f'Expected 1D tensor `confidence` in {sample_path}, got '
                    f'{type(confidence).__name__} with shape {getattr(confidence, "shape", None)}'
                )
            if int(confidence.shape[0]) != seq_len:
                raise ValueError(
                    f'confidence length {int(confidence.shape[0])} does not match seq_len {seq_len} for {sample_path}.'
                )
            hidden_layers = torch.cat(
                [hidden_layers, confidence.to(dtype=hidden_layers.dtype).unsqueeze(-1)],
                dim=-1,
            )

        prompt_length = infer_prompt_length(state)
        gen_len = seq_len - prompt_length

        return {
            'sample_path': str(sample_path),
            'sample_id': sample_id_from_path(sample_path),
            'step_index': torch.tensor(step_index_from_path(sample_path), dtype=torch.int32),
            'hidden_layers': hidden_layers,
            'label': label.to(torch.float32),
            'state': state.to(torch.int64),
            'seq_len': torch.tensor(seq_len, dtype=torch.int32),
            'prompt_length': torch.tensor(prompt_length, dtype=torch.int32),
            'gen_len': torch.tensor(gen_len, dtype=torch.int32),
        }


def tracelock_step_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    batch_size = len(batch)
    max_len = max(int(item['seq_len']) for item in batch)
    feature_rank = int(batch[0]['hidden_layers'].ndim)

    if feature_rank == 3:
        d_model = int(batch[0]['hidden_layers'].shape[-1])
        num_layers = int(batch[0]['hidden_layers'].shape[1])
        hidden_layers = torch.zeros(batch_size, max_len, num_layers, d_model, dtype=batch[0]['hidden_layers'].dtype)
    elif feature_rank == 2:
        feature_dim = int(batch[0]['hidden_layers'].shape[-1])
        hidden_layers = torch.zeros(batch_size, max_len, feature_dim, dtype=batch[0]['hidden_layers'].dtype)
    else:
        raise ValueError(f"Unsupported hidden_layers rank: {feature_rank}")
    state = torch.zeros(batch_size, max_len, dtype=torch.int64)
    label = torch.zeros(batch_size, max_len, dtype=torch.float32)
    sequence_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    step_index = torch.zeros(batch_size, dtype=torch.int32)
    seq_len = torch.zeros(batch_size, dtype=torch.int32)
    prompt_length = torch.zeros(batch_size, dtype=torch.int32)
    gen_len = torch.zeros(batch_size, dtype=torch.int32)
    sample_id: list[str] = []
    sample_path: list[str] = []

    for batch_idx, item in enumerate(batch):
        length = int(item['seq_len'])
        hidden_layers[batch_idx, :length] = item['hidden_layers']
        state[batch_idx, :length] = item['state']
        label[batch_idx, :length] = item['label']
        sequence_mask[batch_idx, :length] = True
        step_index[batch_idx] = item['step_index']
        seq_len[batch_idx] = item['seq_len']
        prompt_length[batch_idx] = item['prompt_length']
        gen_len[batch_idx] = item['gen_len']
        sample_id.append(item['sample_id'])
        sample_path.append(item['sample_path'])

    return {
        'hidden_layers': hidden_layers,
        'state': state,
        'label': label,
        'sequence_mask': sequence_mask,
        'step_index': step_index,
        'seq_len': seq_len,
        'prompt_length': prompt_length,
        'gen_len': gen_len,
        'sample_id': sample_id,
        'sample_path': sample_path,
    }
