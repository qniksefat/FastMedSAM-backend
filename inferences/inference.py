import torch
import numpy as np
import cv2
from matplotlib import pyplot as plt
from os import makedirs
from os.path import join, basename, isfile

from segment_anything.modeling import MaskDecoder, PromptEncoder, TwoWayTransformer
from tiny_vit_sam import TinyViT
from .model import MedSAM_Lite
from .utils import (
    resize_longest_side,
    pad_image,
    get_bbox,
    show_mask,
    show_box,
    resize_box,
)
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class Inference:
    def __init__(self, medsam_lite_checkpoint_path, device, bbox_shift=5):
        self.device = torch.device(device)
        self.bbox_shift = bbox_shift
        self.medsam_lite_model = self.load_model(medsam_lite_checkpoint_path)

    def load_model(self, checkpoint_path):
        logger.info(f"Loading model from checkpoint: {checkpoint_path}")
        medsam_lite_image_encoder = TinyViT(
            img_size=256,
            in_chans=3,
            embed_dims=[
                64, ## (64, 256, 256)
                128, ## (128, 128, 128)
                160, ## (160, 64, 64)
                320 ## (320, 64, 64) 
            ],
            depths=[2, 2, 6, 2],
            num_heads=[2, 4, 5, 10],
            window_sizes=[7, 7, 14, 7],
            mlp_ratio=4.0,
            drop_rate=0.0,
            drop_path_rate=0.0,
            use_checkpoint=False,
            mbconv_expand_ratio=4.0,
            local_conv_size=3,
            layer_lr_decay=0.8,
        )

        medsam_lite_prompt_encoder = PromptEncoder(
            embed_dim=256,
            image_embedding_size=(64, 64),
            input_image_size=(256, 256),
            mask_in_chans=16,
        )

        medsam_lite_mask_decoder = MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=256,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=256,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        )

        medsam_lite_model = MedSAM_Lite(
            image_encoder=medsam_lite_image_encoder,
            mask_decoder=medsam_lite_mask_decoder,
            prompt_encoder=medsam_lite_prompt_encoder,
        )

        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        medsam_lite_model.load_state_dict(checkpoint)
        medsam_lite_model.to(self.device)
        medsam_lite_model.eval()

        return medsam_lite_model

    def preprocess_image(self, image):
        image = resize_longest_side(image, 256)
        new_size = image.shape[:2]
        image = (image - image.min()) / np.clip(image.max() - image.min(), a_min=1e-8, a_max=None)
        image = pad_image(image, 256)
        image = torch.tensor(image).float().permute(2, 0, 1).unsqueeze(0).to(self.device)
        return image, new_size

    @torch.no_grad()
    def infer(self, img_embed, box_256, new_size, original_size):
        box_torch = torch.as_tensor(box_256[None, None, ...], dtype=torch.float, device=self.device)
        sparse_embeddings, dense_embeddings = self.medsam_lite_model.prompt_encoder(
            points=None,
            boxes=box_torch,
            masks=None,
        )
        low_res_logits, _ = self.medsam_lite_model.mask_decoder(
            image_embeddings=img_embed,
            image_pe=self.medsam_lite_model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        low_res_pred = self.medsam_lite_model.postprocess_masks(low_res_logits, new_size, original_size)
        low_res_pred = torch.sigmoid(low_res_pred).squeeze().cpu().numpy()
        return (low_res_pred > 0.5).astype(np.uint8)

    def process_file(self, gt_path_file, save_overlay, png_save_dir, overwrite):
        npz_name = basename(gt_path_file)
        png_file = join(png_save_dir, npz_name.split('.')[0] + '.png')
        if (not isfile(png_file)) or overwrite:
            npz_data = np.load(gt_path_file, 'r', allow_pickle=True)
            img_3D = npz_data['imgs'] # (Num, H, W)
            gt_3D = npz_data['gts'] # (Num, H, W)
            spacing = npz_data['spacing']
            seg_3D = np.zeros_like(gt_3D, dtype=np.uint8)
            box_list = [dict() for _ in range(img_3D.shape[0])]

            for i in range(img_3D.shape[0]):
                img_2d = img_3D[i, :, :] # (H, W)
                H, W = img_2d.shape[:2]
                img_3c = np.repeat(img_2d[:, :, None], 3, axis=-1) # (H, W, 3)
                img_256_tensor, new_size = self.preprocess_image(img_3c)

                with torch.no_grad():
                    image_embedding = self.medsam_lite_model.image_encoder(img_256_tensor)

                gt = gt_3D[i, :, :]
                label_ids = np.unique(gt)[1:]
                for label_id in label_ids:
                    gt2D = np.uint8(gt == label_id)
                    gt2D_resize = cv2.resize(
                        gt2D.astype(np.uint8), new_size, interpolation=cv2.INTER_NEAREST,
                    )
                    gt2D_padded = pad_image(gt2D_resize, 256)
                    if np.sum(gt2D_padded) > 0:
                        box = get_bbox(gt2D_padded, self.bbox_shift)
                        sam_mask = self.infer(image_embedding, box, new_size, (H, W))
                        seg_3D[i, sam_mask > 0] = label_id
                        box_list[i][label_id] = box

           # Visualize overlay, mask, and box
            if save_overlay:
                self.visualize_overlay(img_3D, gt_3D, seg_3D,
                                       box_list, new_size, (H, W),
                                       png_save_dir, npz_name)

    def visualize_overlay(
            self,
            img_3D,
            gt_3D,
            seg_3D,
            box_list,
            new_size,
            original_size,
            png_save_dir,
            npz_name,
    ):
        idx = int(seg_3D.shape[0] / 2)
        box_dict = box_list[idx]
        _, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.imshow(img_3D[idx], cmap='gray')
        ax.axis('off')

        _, box_256 = list(box_dict.items())[-1]
        color = np.random.rand(3)
        box_viz = resize_box(box_256, new_size, original_size)

        show_mask(gt_3D[idx], ax, mask_color=color)
        show_box(box_viz, ax, edgecolor=color)

        plt.tight_layout()
        plt.show()
        save_path = join(png_save_dir, npz_name.split('.')[0] + '.png')
        logger.info(f"Saving PNG file: {save_path}")
        plt.savefig(save_path, dpi=300)
        plt.close()
