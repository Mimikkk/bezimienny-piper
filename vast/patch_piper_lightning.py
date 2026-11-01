"""Make rhasspy/piper's VITS module train on Lightning 2 and a current CUDA torch.

The upstream training_step still takes optimizer_idx, which Lightning 2 removed.
"""
import sys
from pathlib import Path

OLD_STEP = """    def training_step(self, batch: Batch, batch_idx: int, optimizer_idx: int):
        if optimizer_idx == 0:
            return self.training_step_g(batch)

        if optimizer_idx == 1:
            return self.training_step_d(batch)
"""

NEW_STEP = """    def training_step(self, batch: Batch, batch_idx: int):
        opt_g, opt_d = self.optimizers()
        sch_g, sch_d = self.lr_schedulers()

        self.toggle_optimizer(opt_g)
        opt_g.zero_grad()
        loss_g = self.training_step_g(batch)
        self.manual_backward(loss_g)
        opt_g.step()
        self.untoggle_optimizer(opt_g)

        self.toggle_optimizer(opt_d)
        opt_d.zero_grad()
        loss_d = self.training_step_d(batch)
        self.manual_backward(loss_d)
        opt_d.step()
        self.untoggle_optimizer(opt_d)

        if self.trainer.is_last_batch:
            sch_g.step()
            sch_d.step()
        return loss_g
"""


def main():
    path = Path(sys.argv[1])
    text = path.read_text(encoding="utf-8")
    if "automatic_optimization = False" not in text:
        text = text.replace(
            "class VitsModel(pl.LightningModule):",
            "class VitsModel(pl.LightningModule):\n    automatic_optimization = False",
            1,
        )
    if OLD_STEP not in text:
        if "def training_step(self, batch: Batch, batch_idx: int):" in text:
            print(f"already patched {path}")
            return
        raise SystemExit(f"training_step block not found in {path}")
    text = text.replace(OLD_STEP, NEW_STEP, 1)
    path.write_text(text, encoding="utf-8")
    print(f"patched {path}")


if __name__ == "__main__":
    main()
