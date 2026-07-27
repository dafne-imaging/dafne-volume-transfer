import numpy as np
import torch

from sam2.build_sam import build_sam2_video_predictor_npz
from sam2_support_match.preprocessing import resize_grayscale_to_rgb_and_resize, IMG_SIZE

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)

class MedSAM2Segmenter: 
    def __init__(self, checkpoint: str, model_cfg: str, device: str='cuda'):
        """Load SAM2 video predictor from checkpoint/config onto device."""
        self.checkpoint = checkpoint
        self.predictor = build_sam2_video_predictor_npz(model_cfg, checkpoint)
        self.model_cfg = model_cfg
        self.device = 'cuda' if device.startswith('cuda') else 'cpu'

    def _preprocess(self, vol_u8: np.ndarray) -> torch.Tensor:
        '''
        process uint8 volume for sam2 throught IMAGENET normaliazion
        '''
        img_processed = resize_grayscale_to_rgb_and_resize(vol_u8, IMG_SIZE)
        t = torch.from_numpy(img_processed).to(self.device).float()
        mean = torch.tensor(_IMAGENET_MEAN, device=self.device)[:, None, None]
        std = torch.tensor(_IMAGENET_STD, device=self.device)[:, None, None]
        return (t - mean) / std

    @torch.inference_mode()
    def encoder_f_extractor(self, frame_u8: np.ndarray) -> torch.Tensor:
        '''
        frame_u8: [H,W] uint8 -> [C,h,w] SAM2 image-encoder feature (most-semantic
        FPN level)
        '''
        frame_u8_processed = self._preprocess(frame_u8[None]) # [1, 3, img_size, img_size]
        autocast = (torch.autocast(self.device, dtype=torch.bfloat16)
                    if self.device == 'cuda'
                    else torch.autocast(self.device, enabled=False))
        with autocast:
            out = self.predictor.forward_image(frame_u8_processed)
        return out["backbone_fpn"][-1][0]

    @torch.inference_mode()
    def encoder_frames_batched(self, frames_u8: list, chunk_size: int = 8) ->list:
        '''
        frames_u8: [B, H, W] uint8 -> [B, C, h, w] SAM2 image-encoder feature (most-semantic
                FPN level) -> batched frames by chunks
        '''
        autocast = (torch.autocast(self.device, dtype=torch.bfloat16) if self.device == 'cuda'
                    else torch.autocast(self.device, enabled=False))
        feats=[]
        for start in range (0, len(frames_u8), chunk_size):
            chunk = np.stack(frames_u8[start:start+chunk_size], axis=0)
            img = self._preprocess(chunk) #normalize chunk
            with autocast:
                out = self.predictor.forward_image(img)
            fpn = out['backbone_fpn'][-1]
            feats.extend(fpn[i] for i in range(fpn.shape[0]))
        return feats

    @torch.inference_mode()
    def segment_volume_mask(self, vol_u8: np.ndarray,
                            masks: dict[int, np.ndarray]) -> np.ndarray:
        """
        Mask-prompt variant of segment_volume (pseudo-label ablation, models/
        support_prompt.py multiclass_masks)
        vol_u8 : [Z,H,W] uint8 [0,255] (already cropped to the propagation range).
        masks  : {frame_idx -> [H,W] bool} in ORIGINAL (H,W) coords; one or more
                 prompted slices. Propagation conditions on all prompted frames.
        returns: [Z,H,W] uint8 binary mask.
        """
        Z, H, W = vol_u8.shape
        seg = np.zeros((Z, H, W), dtype=np.uint8)
        if not masks:
            return seg

        img_resized = self._preprocess(vol_u8)
        autocast = (torch.autocast(self.device, dtype=torch.bfloat16)
                    if self.device == "cuda"
                    else torch.autocast(self.device, enabled=False))
        with autocast:
            state = self.predictor.init_state(img_resized, H, W)
            for fidx, mask in sorted(masks.items()):
                self.predictor.add_new_mask(
                    inference_state=state, frame_idx=int(fidx), obj_id=1,
                    mask=mask) #add mask as label prompt for specific frame (slice idx)

            for reverse in (False, True):
                for fidx, _oids, logits in self.predictor.propagate_in_video(
                        state, reverse=reverse): #propagate in video
                    seg[fidx][(logits[0] > 0.0).cpu().numpy()[0]] = 1
        return seg
