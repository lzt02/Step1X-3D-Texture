from dataclasses import dataclass, field
import numpy as np
import torch
from skimage import measure
from einops import repeat, rearrange

import step1x3d_geometry
from step1x3d_geometry.systems.base import BaseSystem
from step1x3d_geometry.utils.ops import generate_dense_grid_points
from step1x3d_geometry.utils.typing import *
from step1x3d_geometry.utils.misc import get_rank


@step1x3d_geometry.register("texture-autoencoder-system")
class ShapeAutoEncoderSystem(BaseSystem):
    @dataclass
    class Config(BaseSystem.Config):
        shape_model_type: str = None
        shape_model: dict = field(default_factory=dict)

        sample_posterior: bool = True

        # for mesh extraction
        bounds: float = 1.05
        mc_level: float = 0.0
        octree_resolution: int = 256

    cfg: Config

    def configure(self):
        super().configure()

        self.shape_model = step1x3d_geometry.find(self.cfg.shape_model_type)(
            self.cfg.shape_model
        )

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        # rand_points = batch["rand_points"]
        print("Batch keys:", batch.keys())
        rand_points = batch["feature"][:, :, -3:] #[1, 10, 1027] -> [1, 10, 3]
        if "sdf" in batch:
            target = batch["sdf"]
            criteria = torch.nn.MSELoss()
        elif "occupancies" in batch:
            target = batch["occupancies"]
            criteria = torch.nn.BCEWithLogitsLoss()
        elif "feature" in batch:
            target = batch["surface"]
            criteria = torch.nn.MSELoss()
        else:
            raise NotImplementedError

        # forward pass
        num_point_feats = 3 + self.cfg.shape_model.point_feats
        shape_latents, kl_embed, posterior = self.shape_model.encode(
            batch["surface"][..., :num_point_feats],
            sharp_surface=(
                batch["sharp_surface"][..., :num_point_feats]
                if "sharp_surface" in batch
                else None
            ),
            sample_posterior=self.cfg.sample_posterior,
        )
        latents = self.shape_model.decode(kl_embed)  # [B, num_latents, width]
        print(f"rand_points.shape:{rand_points.shape}")
        print(f"latents.shape:{latents.shape}")
        logits = self.shape_model.query(rand_points, latents).squeeze(  # pc, feats
            -1
        )  # [B, num_rand_points]
        print(f"logits.shape:{logits.shape}")
        if self.cfg.sample_posterior:
            loss_kl = posterior.kl()
            loss_kl = torch.sum(loss_kl) / loss_kl.shape[0]

            return {
                "loss_logits": criteria(logits, target).mean(),
                "loss_kl": loss_kl,
                "logits": logits,
                "target": target,
                "latents": latents,
            }
        else:
            return {
                "loss_logits": criteria(logits, target).mean(),
                "latents": latents,
                "logits": logits,
            }

    def training_step(self, batch, batch_idx):
        """
        Description:

        Args:
            batch:
            batch_idx:
        Returns:
            loss:
        """
        out = self(batch)

        loss = 0.0
        for name, value in out.items():
            if name.startswith("loss_"):
                self.log(f"train/{name}", value)
                loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))
        print(f"loss: {loss}")
        # 每 N 个 epoch 保存一次 logits 和 target
        if (self.current_epoch + 1) % 20000 == 0 and batch_idx == 0:
            logits = out["logits"].squeeze(0)  # 从 [1, N, 1027] 转换为 [N, 1027]
            target = out["target"].squeeze(0) 
            # 保存为文本文件
            import os
            save_dir = f"saved_arrays_{logits.shape[0]}_{logits.shape[1]}"
            os.makedirs(save_dir, exist_ok=True)
            # 提取 logits 和 target，并调整形状
            print(f"saving: {logits.shape}")
            np.savetxt(os.path.join(save_dir, f"logits_epoch_{self.current_epoch + 1}.txt"), logits.detach().cpu().numpy())
            np.savetxt(os.path.join(save_dir, f"target_epoch_{self.current_epoch + 1}.txt"), target.detach().cpu().numpy())
        return {"loss": loss}

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        self.eval()
        out = self(batch)

        meshes = self.shape_model.extract_geometry(
            out["latents"],
            bounds=self.cfg.bounds,
            mc_level=self.cfg.mc_level,
            octree_resolution=self.cfg.octree_resolution,
            enable_pbar=False,
        )
        for idx, name in enumerate(batch["uid"]):
            self.save_mesh(
                f"it{self.true_global_step}/{name}.obj",
                meshes[idx].verts,
                meshes[idx].faces,
            )

        threshold = 0
        outputs = out["logits"]
        labels = out["target"]
        pred = torch.zeros_like(outputs)
        pred[outputs >= threshold] = 1

        accuracy = (pred == labels).float().sum(dim=1) / labels.shape[1]
        accuracy = accuracy.mean()
        intersection = (pred * labels).sum(dim=1)
        union = (pred + labels).gt(0).sum(dim=1)
        iou = intersection * 1.0 / union + 1e-5
        iou = iou.mean()
        self.log("val/accuracy", accuracy)
        self.log("val/iou", iou)

        torch.cuda.empty_cache()

        return {
            "val/loss": out["loss_logits"],
            "val/accuracy": accuracy,
            "val/iou": iou,
        }

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        return
