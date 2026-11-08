"""Passive experiment records; never changes sampling or optimization state."""

import hashlib
import json
import random
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


def text_hash(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def rng_snapshot():
    return {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch_cpu': torch.get_rng_state(),
        'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
    }


class RunAudit:
    def __init__(self, save_dir, config, batch_indices, queries,
                 validation_indices=None):
        self.run_id = uuid.uuid4().hex
        self.directory = Path(save_dir) / 'audit' / self.run_id
        self.directory.mkdir(parents=True, exist_ok=False)
        repo = Path(__file__).resolve().parents[2]

        def git(*args):
            try:
                result = subprocess.run(
                    ['git', *args], cwd=repo, capture_output=True, text=True,
                    timeout=15, check=True,
                )
                return result.stdout
            except (OSError, subprocess.SubprocessError) as exc:
                return 'unavailable: ' + str(exc)

        source_dir = repo / 'attacks' / 'work1'
        sources = list(source_dir.rglob('*.py'))
        sources += list((repo / 'multimodalmodels').glob('work1_*.py'))
        sources.append(repo / 'multimodalmodels' / '__init__.py')
        for family in ('qwen2_5_vl', 'glm4_1v', 'qwen3_vl', 'llava', 'kimi_vl'):
            sources += list((repo / 'multimodalmodels' / family).rglob('*.py'))
        hashes = {
            str(path.relative_to(repo)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(sources)
            if '__pycache__' not in path.parts
        }
        # Include untracked Python sources too; git diff alone cannot reproduce them.
        for relative in hashes:
            target = self.directory / 'source' / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((repo / relative).read_bytes())
        metadata = {
            'run_id': self.run_id,
            'started_utc': datetime.now(timezone.utc).isoformat(),
            'python_executable': sys.executable,
            'torch_version': str(torch.__version__),
            'cuda_version': torch.version.cuda,
            'git_commit': git('rev-parse', 'HEAD').strip(),
            'git_status': git('status', '--short'),
            'source_sha256': hashes,
            'attack_config': config,
            'mcts_indices_in_training_subset': batch_indices,
            'mcts_query_sha256': [text_hash(queries[i]) for i in batch_indices],
            'mcts_validation_indices_in_training_subset': validation_indices or [],
            'mcts_validation_query_sha256': [
                text_hash(queries[i]) for i in (validation_indices or [])
            ],
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'generation_config_note': 'Downstream generation settings are not captured by this training record.',
        }
        (self.directory / 'metadata.json').write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8',
        )
        (self.directory / 'source.diff').write_text(
            git('diff', 'HEAD', '--', 'attacks/work1'), encoding='utf-8',
        )
        self.save_rng('start')

    def save_rng(self, label):
        torch.save(rng_snapshot(), self.directory / f'rng_{label}.pt')

    def append(self, stream, record):
        with (self.directory / f'{stream}.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps({'run_id': self.run_id, **record}, ensure_ascii=False) + '\n')
