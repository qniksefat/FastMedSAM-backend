import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from matplotlib import pyplot as plt
from os import makedirs
from os.path import join, basename, isfile

from segment_anything.modeling import MaskDecoder, PromptEncoder, TwoWayTransformer
from tiny_vit_sam import TinyViT
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class MedSAM_Lite(nn.Module):
    def __init__(
            self,
            image_encoder,
            mask_decoder,
            prompt_encoder
        ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder

    def forward(self, image, box_np):
        image_embedding = self.image_encoder(image) # (B, 256, 64, 64)
        # do not compute gradients for prompt encoder
        with torch.no_grad():
            box_torch = torch.as_tensor(box_np, dtype=torch.float32, device=image.device)
            if len(box_torch.shape) == 2:
                box_torch = box_torch[:, None, :] # (B, 1, 4)

        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=None,
            boxes=box_np,
            masks=None,
        )
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embedding, # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
            multimask_output=False,
          ) # (B, 1, 256, 256)

        return low_res_masks

    @torch.no_grad()
    def postprocess_masks(self, masks, new_size, original_size):
        """
        Do cropping and resizing

        Parameters
        ----------
        masks : torch.Tensor
            masks predicted by the model
        new_size : tuple
            the shape of the image after resizing to the longest side of 256
        original_size : tuple
            the original shape of the image

        Returns
        -------
        torch.Tensor
            the upsampled mask to the original size
        """
        # Crop
        masks = masks[..., :new_size[0], :new_size[1]]
        # Resize
        masks = F.interpolate(
            masks,
            size=(original_size[0], original_size[1]),
            mode="bilinear",
            align_corners=False,
        )

        return masks


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
        logger.info(f"Preprocessing image")
        image = resize_longest_side(image, 256)
        new_size = image.shape[:2]
        image = (image - image.min()) / np.clip(image.max() - image.min(), a_min=1e-8, a_max=None)
        image = pad_image(image, 256)
        image = torch.tensor(image).float().permute(2, 0, 1).unsqueeze(0).to(self.device)
        logger.info(f"Image preprocessed: {image.shape}")
        logger.info(f"New size: {new_size}")
        return image, new_size

    @torch.no_grad()
    def infer(self, img_embed, box_256, new_size, original_size):
        logger.info(f"Inferring segmentation")
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
        logger.info(f"Postprocessing segmentation")
        low_res_pred = self.medsam_lite_model.postprocess_masks(low_res_logits, new_size, original_size)
        logger.info(f"Segmentation postprocessed")
        low_res_pred = torch.sigmoid(low_res_pred).squeeze().cpu().numpy()
        logger.info(f"Segmentation completed")
        return (low_res_pred > 0.5).astype(np.uint8)

    def process_file(self, gt_path_file, pred_save_dir, save_overlay, png_save_dir, overwrite):
        npz_name = basename(gt_path_file)
        task_folder = gt_path_file.split('/')[-2]
        makedirs(join(pred_save_dir, task_folder), exist_ok=True)

        if (not isfile(join(pred_save_dir, task_folder, npz_name))) or overwrite:
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

                gt = gt_3D[i, :, :] # (H, W)
                label_ids = np.unique(gt)[1:]
                for label_id in label_ids:
                    gt2D = np.uint8(gt == label_id)
                    gt2D_resize = cv2.resize(gt2D.astype(np.uint8), new_size, interpolation=cv2.INTER_NEAREST)
                    gt2D_padded = pad_image(gt2D_resize, 256)
                    if np.sum(gt2D_padded) > 0:
                        box = get_bbox(gt2D_padded, self.bbox_shift)
                        sam_mask = self.infer(image_embedding, box, new_size, (H, W))
                        seg_3D[i, sam_mask > 0] = label_id
                        box_list[i][label_id] = box
                
            np.savez_compressed(
                join(pred_save_dir, task_folder, npz_name),
                segs=seg_3D, gts=gt_3D, spacing=spacing
            )

            # Visualize overlay, mask, and box
            if save_overlay:
                self.visualize_overlay(img_3D, gt_3D, seg_3D, 
                                       box_list, new_size, (H, W), 
                                       png_save_dir, npz_name)

    def visualize_overlay(self, img_3D, gt_3D, seg_3D, box_list, new_size, original_size, png_save_dir, npz_name):
        idx = int(seg_3D.shape[0] / 2)
        box_dict = box_list[idx]
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(img_3D[idx], cmap='gray')
        ax[1].imshow(img_3D[idx], cmap='gray')
        ax[2].imshow(img_3D[idx], cmap='gray')
        ax[0].set_title("Image")
        ax[1].set_title("Ground Truth")
        ax[2].set_title(f"Segmentation")
        ax[0].axis('off')
        ax[1].axis('off')
        ax[2].axis('off')
        for label_id, box_256 in box_dict.items():
            color = np.random.rand(3)
            box_viz = resize_box(box_256, new_size, original_size)
            show_mask(gt_3D[idx], ax[1], mask_color=color)
            show_box(box_viz, ax[1], edgecolor=color)
            show_mask(seg_3D[idx], ax[2], mask_color=color)
            show_box(box_viz, ax[2], edgecolor=color)
        plt.tight_layout()
        plt.savefig(join(png_save_dir, npz_name.split(".")[0] + '.png'), dpi=300)
        plt.close()

def resize_longest_side(image, target_length):
    long_side_length = target_length
    oldh, oldw = image.shape[0], image.shape[1]
    scale = long_side_length * 1.0 / max(oldh, oldw)
    newh, neww = int(oldh * scale + 0.5), int(oldw * scale + 0.5)
    return cv2.resize(image, (neww, newh), interpolation=cv2.INTER_AREA)

def pad_image(image, target_size):
    h, w = image.shape[0], image.shape[1]
    padh = target_size - h
    padw = target_size - w
    if len(image.shape) == 3:
        return np.pad(image, ((0, padh), (0, padw), (0, 0)))
    else:
        return np.pad(image, ((0, padh), (0, padw)))

def get_bbox(gt2D, bbox_shift=5):
    y_indices, x_indices = np.where(gt2D > 0)
    x_min, x_max = x_indices.min(), x_indices.max()
    y_min, y_max = y_indices.min(), y_indices.max()
    return np.array([x_min - bbox_shift, y_min - bbox_shift, 
                     x_max + bbox_shift + 1, y_max + bbox_shift + 1])

def resize_box(box: np.ndarray, 
               new_size: tuple, 
               original_size: tuple) -> np.ndarray:    
    new_box = np.zeros_like(box)
    scale = max(original_size) / max(new_size)
    for i in range(len(box)):
       new_box[i] = int(box[i] * scale)
    return new_box

def show_mask(mask, ax, mask_color=None, alpha=0.5):
    if mask_color:
        color = np.concatenate([mask_color, np.array([alpha])], axis=0)
    else:
        color = np.array([251 / 255, 252 / 255, 30 / 255, alpha])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax, edgecolor=(0, 1, 0)):
    x0, y0, x1, y1 = box
    ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, edgecolor=edgecolor, facecolor=(0, 0, 0, 0), lw=2))

