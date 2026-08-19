"""
SAM2VideoPredictor variant whose init_state takes an already-preprocessed image tensor
directly, instead of a video path. Stock SAM2's init_state loads frames from a directory of
jpegs or an mp4 (sam2.utils.misc.load_video_frames); backbone.MedSAM2Segmenter does its own
frame prep (_preprocess) and hands the resulting tensor straight to init_state, so there is
never a video file on disk to load from.

Reimplemented here against pip-installed sam2's own SAM2VideoPredictor rather than depending
on bowang-lab/MedSAM2's fork (which this project used to vendor this trick from): init_state
is the only method overridden, copied from sam2.sam2_video_predictor.SAM2VideoPredictor's own
init_state minus the load_video_frames() call it replaces, so every other inherited method
keeps seeing exactly the inference_state shape the installed sam2 version expects.
"""

from collections import OrderedDict

import torch
from sam2.build_sam import _load_checkpoint
from sam2.sam2_video_predictor import SAM2VideoPredictor

from sam2_support_match.preprocessing import IMG_SIZE

_FEAT_STRIDE = 16  # SAM2's memory-attention feature grid is image_size / 16 per side


class SAM2VideoPredictorNPZ(SAM2VideoPredictor):
    @torch.inference_mode()
    def init_state(self, images: torch.Tensor, video_height: int, video_width: int,
                   offload_video_to_cpu: bool = False, offload_state_to_cpu: bool = False):
        """images: [Z,3,image_size,image_size] preprocessed tensor (see
        backbone.MedSAM2Segmenter._preprocess), image_size matching the model this predictor
        was built with (build_sam2_video_predictor_npz's image_size)."""
        compute_device = self.device
        inference_state = {
            "images": images,
            "num_frames": len(images),
            "offload_video_to_cpu": offload_video_to_cpu,
            "offload_state_to_cpu": offload_state_to_cpu,
            "video_height": video_height,
            "video_width": video_width,
            "device": compute_device,
            "storage_device": torch.device("cpu") if offload_state_to_cpu else compute_device,
            "point_inputs_per_obj": {},
            "mask_inputs_per_obj": {},
            "cached_features": {},
            "constants": {},
            "obj_id_to_idx": OrderedDict(),
            "obj_idx_to_id": OrderedDict(),
            "obj_ids": [],
            "output_dict_per_obj": {},
            "temp_output_dict_per_obj": {},
            "frames_tracked_per_obj": {},
        }
        # Warm up the visual backbone and cache the image feature on frame 0
        self._get_image_feature(inference_state, frame_idx=0, batch_size=1)
        return inference_state


def build_sam2_video_predictor_npz(config_file: str, ckpt_path: str = None, device: str = "cuda",
                                   mode: str = "eval", apply_postprocessing: bool = True,
                                   image_size: int = IMG_SIZE, **kwargs) -> SAM2VideoPredictorNPZ:
    """
    Same shape as sam2.build_sam.build_sam2_video_predictor, but instantiates
    SAM2VideoPredictorNPZ (see above) and overrides image_size -- and the memory-attention
    feat_sizes derived from it -- to `image_size`. Every stock sam2/sam2.1 hydra config
    (automatic.checkpoints.CHECKPOINT_MODELS) defaults to image_size=1024; this project
    always resizes frames to preprocessing.IMG_SIZE (512) instead, so the override is applied
    here rather than needing a separate config file per resolution.
    """
    from hydra import compose
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    feat_side = image_size // _FEAT_STRIDE
    hydra_overrides = [
        "++model._target_=sam2_support_match.automatic.video_predictor_npz.SAM2VideoPredictorNPZ",
        f"++model.image_size={image_size}",
        f"++model.memory_attention.layer.self_attention.feat_sizes=[{feat_side},{feat_side}]",
        f"++model.memory_attention.layer.cross_attention.feat_sizes=[{feat_side},{feat_side}]",
    ]
    if apply_postprocessing:
        hydra_overrides += [
            # dynamically fall back to multi-mask if the single mask is not stable
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
            # the sigmoid mask logits on interacted frames with clicks in the memory
            # encoder so that the encoded masks are exactly as what users see from clicking
            "++model.binarize_mask_from_pts_for_mem_enc=true",
            # fill small holes in the low-res masks up to `fill_hole_area` (before
            # resizing them to the original video resolution)
            "++model.fill_hole_area=8",
        ]

    cfg = compose(config_name=config_file, overrides=hydra_overrides)
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, ckpt_path)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model
