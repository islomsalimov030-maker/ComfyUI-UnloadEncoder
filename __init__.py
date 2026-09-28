"""
ComfyUI-UnloadEncoder

Frees a text encoder (CLIP) from RAM/VRAM right after it has produced the
conditioning, instead of letting it sit in the node cache for the whole run.

Nodes:
  * Unload Encoder From RAM  - passthrough; destroys the CLIP's weights once the
                               conditioning wired into it has been computed.
  * CLIPLoader (disposable)  - same as core CLIPLoader, but never served from
                               cache, so a destroyed encoder is never reused.

Wiring:
  CLIPLoader (disposable) -> clip  of the encode node (e.g. MiniMax H3 Reference To Video)
                          -> clip  of Unload Encoder From RAM
  encode node .positive   -> anything of Unload Encoder From RAM
  Unload .anything        -> conditioning of the Guider   (this forces the order:
                                                           unload BEFORE sampling)
"""

import ctypes
import gc
import logging
import os
import sys

import torch

import comfy.model_management as mm
import nodes

log = logging.getLogger("UnloadEncoder")


class AnyType(str):
    """Type string that matches any socket type."""

    def __ne__(self, other):
        return False


ANY = AnyType("*")


def _rss_gb():
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / 1024**3
    except Exception:
        return float("nan")


def _malloc_trim():
    """Ask glibc to hand freed heap pages back to the OS (Linux only)."""
    if sys.platform.startswith("linux"):
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass


_unpin = getattr(mm, "unpin_memory", None)


def _release_tensor(t):
    if t is None:
        return 0
    size = 0
    try:
        size = t.numel() * t.element_size()
    except Exception:
        pass
    if _unpin is not None:
        try:
            if t.device.type == "cpu":
                _unpin(t)
        except Exception:
            pass
    return size


def _strip_module(module):
    """Drop every parameter and buffer of every submodule (frees RAM and VRAM)."""
    freed, count = 0, 0
    for m in module.modules():
        for k, p in list(m._parameters.items()):
            if p is not None:
                freed += _release_tensor(p)
                m._parameters[k] = None
                count += 1
        for k, b in list(m._buffers.items()):
            if b is not None:
                freed += _release_tensor(b)
                m._buffers[k] = None
                count += 1
    return freed, count


def _forget_in_model_management(inner_model):
    """Remove the encoder from ComfyUI's loaded-model list WITHOUT moving it to
    CPU first (a normal unload would copy the whole encoder into RAM)."""
    removed = 0
    keep = []
    for lm in list(mm.current_loaded_models):
        patcher = getattr(lm, "model", None)
        target = getattr(patcher, "model", None) if patcher is not None else None
        if target is not None and target is inner_model:
            removed += 1
        else:
            keep.append(lm)
    mm.current_loaded_models[:] = keep
    return removed


class UnloadEncoderFromRAM:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "anything": (ANY, {"tooltip": "Conditioning (or anything) produced with this CLIP. Passed through unchanged."}),
                "clip": ("CLIP", {"tooltip": "The encoder to destroy. Must not be used anywhere else afterwards."}),
            }
        }

    RETURN_TYPES = (ANY, "STRING")
    RETURN_NAMES = ("anything", "report")
    FUNCTION = "run"
    CATEGORY = "utils/memory"

    def run(self, anything, clip):
        before = _rss_gb()

        patcher = getattr(clip, "patcher", None)
        inner = getattr(patcher, "model", None) or getattr(clip, "cond_stage_model", None)

        if inner is None:
            msg = "UnloadEncoder: nothing to free (encoder already released?)"
            log.info(msg)
            return (anything, msg)

        removed = _forget_in_model_management(inner)
        freed, count = _strip_module(inner)

        if patcher is not None:
            for attr in ("backup", "backup_buffers", "object_patches_backup", "patches", "pinned"):
                d = getattr(patcher, attr, None)
                if hasattr(d, "clear"):
                    try:
                        d.clear()
                    except Exception:
                        pass

        del inner
        gc.collect()
        mm.soft_empty_cache()
        _malloc_trim()

        after = _rss_gb()
        msg = (
            f"UnloadEncoder: dropped {count} tensors (~{freed / 1024**3:.1f} GB), "
            f"removed {removed} loaded-model entr{'y' if removed == 1 else 'ies'}. "
            f"Process RAM {before:.1f} GB -> {after:.1f} GB"
        )
        log.info(msg)
        print(msg)
        return (anything, msg)


NODE_CLASS_MAPPINGS = {"UnloadEncoderFromRAM": UnloadEncoderFromRAM}
NODE_DISPLAY_NAME_MAPPINGS = {"UnloadEncoderFromRAM": "Unload Encoder From RAM"}


_base_loader = getattr(nodes, "CLIPLoader", None)
if _base_loader is not None and hasattr(_base_loader, "INPUT_TYPES"):

    class DisposableCLIPLoader(_base_loader):
        """Core CLIPLoader that re-loads every run, so a destroyed encoder
        left in the node cache is never handed out again."""

        @classmethod
        def IS_CHANGED(cls, **kwargs):
            return float("nan")

    NODE_CLASS_MAPPINGS["DisposableCLIPLoader"] = DisposableCLIPLoader
    NODE_DISPLAY_NAME_MAPPINGS["DisposableCLIPLoader"] = "CLIPLoader (disposable)"
else:
    log.warning("UnloadEncoder: core CLIPLoader not found, 'CLIPLoader (disposable)' not registered")
