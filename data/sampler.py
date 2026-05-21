import math
import random
from collections import defaultdict, deque

from torch.utils.data import Sampler


class PatientAwareBatchSampler(Sampler):
    """
    Round-robin batch sampler with foreground guarantee (nnUNet-style).

    Each batch has at least ceil(batch_size * fg_fraction) positive samples.
    When the patient-aware interleaved order doesn't naturally provide enough,
    extra positive indices are cycled in — same mechanism as nnUNet's
    foreground oversampling that prevents gradient collapse on rare objects.

    fg_fraction=0 disables the guarantee (plain round-robin).
    """

    def __init__(self, dataset, batch_size: int, drop_last: bool = True,
                 shuffle: bool = True, seed: int = 0,
                 fg_fraction: float = 1 / 3):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self._base_seed = seed
        self._epoch = 0
        self.fg_fraction = fg_fraction

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def _interleave(self) -> list:
        rng = random.Random(self._base_seed + self._epoch)

        groups = defaultdict(list)
        for i, p in enumerate(self.dataset.patient_ids):
            groups[p].append(i)

        # deterministic patient order, then shuffle once per epoch
        patients = sorted(groups.keys())
        if self.shuffle:
            for p in patients:
                rng.shuffle(groups[p])
            rng.shuffle(patients)

        queues = [deque(groups[p]) for p in patients]
        merged = []
        while queues:
            next_round = []
            for q in queues:
                merged.append(q.popleft())
                if q:
                    next_round.append(q)
            queues = next_round
        return merged

    def __iter__(self):
        rng = random.Random(self._base_seed + self._epoch)
        order = self._interleave()
        n_fg_required = math.ceil(self.batch_size * self.fg_fraction)

        # Build a cycling pool of positive indices for the fg-guarantee slots.
        if n_fg_required > 0:
            pos_indices = [
                i for i, s in enumerate(self.dataset.samples)
                if s["mask"] != "empty"
            ]
            if self.shuffle:
                rng.shuffle(pos_indices)
            repeats = (len(order) // max(len(pos_indices), 1)) + 2
            pos_cycle = pos_indices * repeats
            pos_ptr = [0]

            def next_pos():
                idx = pos_cycle[pos_ptr[0] % len(pos_cycle)]
                pos_ptr[0] += 1
                return idx

        for i in range(0, len(order), self.batch_size):
            batch = list(order[i:i + self.batch_size])
            if len(batch) < self.batch_size:
                if not self.drop_last:
                    yield batch
                break

            if n_fg_required > 0:
                n_pos_in_batch = sum(
                    1 for idx in batch
                    if self.dataset.samples[idx]["mask"] != "empty"
                )
                n_need = max(0, n_fg_required - n_pos_in_batch)
                if n_need > 0:
                    # Replace negative slots with guaranteed-positive samples.
                    neg_slots = [
                        j for j, idx in enumerate(batch)
                        if self.dataset.samples[idx]["mask"] == "empty"
                    ]
                    replace = rng.sample(neg_slots, min(n_need, len(neg_slots)))
                    for slot in replace:
                        batch[slot] = next_pos()
                    rng.shuffle(batch)

            yield batch

    def __len__(self) -> int:
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size
