"""Balanced retention/interference objective and delay-resolved diagnostics."""

from pathlib import Path

import torch

from dsrnn.data.selective_memory import UPDATE
from dsrnn.training.selective_memory import SelectiveMemoryModule, masked_query_mse


class LongMemoryModule(SelectiveMemoryModule):
    """Same recurrent models/readout, with an explicitly balanced task loss.

    Half the loss is the protected-target query; half is mean distractor query
    error. This prevents long delays from diluting target supervision.
    Single-device runner; conditional metrics are weighted by query counts.
    """

    def _shared_step(self, batch, stage):
        predictions, trace = self(batch["input"], return_routing=True)
        target = batch["target_query"]
        distractor = batch["query_mask"] & ~target
        target_loss = masked_query_mse(predictions, batch["target"], target)
        errors = (predictions - batch["target"]).square().mean(-1)
        distractor_loss = ((errors * distractor).sum(1)
                           / distractor.sum(1).clamp_min(1)).mean()
        loss = 0.5 * (target_loss + distractor_loss)
        self.log(f"{stage}/balanced_mse", loss, on_step=False, on_epoch=True,
                 batch_size=predictions.size(0), prog_bar=stage != "train")
        self.log(f"{stage}/distractor_mse", distractor_loss, on_step=False,
                 on_epoch=True, batch_size=predictions.size(0))
        groups = {"target": target, "distractor_query": distractor,
                  "query": batch["query_mask"],
                  "target_pure": target & ~batch["operations"][..., UPDATE],
                  "target_joint": target & batch["operations"][..., UPDATE]}
        for delay in batch["retention_delay"][target].unique().tolist():
            groups[f"delay_{delay}/target"] = target & (batch["retention_delay"] == delay)
        for name, mask in groups.items():
            count = int(mask.sum())
            if count:
                self.log(f"{stage}/{name}_mse", masked_query_mse(
                    predictions, batch["target"], mask), on_step=False,
                    on_epoch=True, batch_size=count)
        if stage != "train":
            read, write = trace["read_mask"] > 0.5, trace["write_mask"] > 0.5
            b, q = target.nonzero(as_tuple=True)
            source_t = batch["source_timestep"][b, q]
            source = write[b, source_t]
            protected_read = read[b, q] & ~write[b, q]
            counts = write.long().cumsum(1)
            survived = source & ((counts[b, q] - counts[b, source_t]) == 0)
            budget = source.sum(-1).clamp_min(1)
            alignments = {
                "source_read_fraction": (source & read[b, q]).sum(-1) / budget,
                "source_protected_read_fraction": (source & protected_read).sum(-1) / budget,
                "source_preserved_read_fraction": (survived & read[b, q]).sum(-1) / budget,
                "protected_read_excess": ((source & protected_read).sum(-1) / budget
                                          - protected_read.float().mean(-1)),
            }
            for name, value in alignments.items():
                self.log(f"{stage}/target_{name}", value.mean(), on_step=False,
                         on_epoch=True, batch_size=predictions.size(0))
            for name, mask in groups.items():
                count = int(mask.sum())
                if count:
                    for state, active in {"00": ~read & ~write, "10": read & ~write,
                                          "01": ~read & write, "11": read & write}.items():
                        self.log(f"{stage}/{name}_state{state}", active[mask].float().mean(),
                                 on_step=False, on_epoch=True, batch_size=count)
            # Per-sequence coordinate write counts and longest protection run.
            streak = torch.zeros_like(write[:, 0], dtype=torch.long)
            longest = streak.clone()
            for t in range(write.size(1)):
                streak = (streak + 1) * ~write[:, t]
                longest = torch.maximum(longest, streak)
            for name, value in {
                "writes_per_coordinate": write.float().sum(1).mean(),
                "longest_protection": longest.float().mean(),
            }.items():
                self.log(f"{stage}/{name}", value, on_step=False, on_epoch=True,
                         batch_size=predictions.size(0))
        if stage == "test" and self.hparams.analysis_path and not self._test_traces:
            # Bounded diagnostic sample: full test metrics still use all samples.
            record = {k: v.detach().cpu() for k, v in batch.items()}
            record.update({k: v.detach().cpu() for k, v in trace.items()})
            record["predictions"] = predictions.detach().cpu()
            record["task_loss"] = (errors * batch["query_mask"]).detach().cpu()
            self._test_traces.append(record)
        return {"loss": loss}

    def on_test_epoch_end(self):
        if self._test_traces:
            path = Path(self.hparams.analysis_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"trace_scope": "first_test_batch", "variant": self.hparams.variant,
                        "readout_source": "full_post_write_hidden_state",
                        **self._test_traces[0]}, path)
        self._test_traces.clear()
