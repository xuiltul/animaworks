"""Local Diffusers-backed image generation helpers."""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from core.tools._base import logger

if TYPE_CHECKING:
    from core.config.models import ImageGenConfig

_HF_CACHE_ROOT = Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
_AUTO_MODEL_REPOS = (
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    "runwayml/stable-diffusion-v1-5",
)
_ASPECT_SIZES: dict[str, tuple[int, int]] = {
    "1:1": (896, 896),
    "3:4": (768, 1024),
    "4:3": (1024, 768),
}
_PIPELINE_CACHE: dict[tuple[str, str, str, str], Any] = {}
_IP_ADAPTER_LOADED: set[tuple[str, str, str, str]] = set()


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required for local Diffusers image generation.") from exc
    return torch


def _import_diffusers() -> tuple[Any, Any]:
    try:
        from diffusers import AutoPipelineForImage2Image, AutoPipelineForText2Image
    except ImportError as exc:
        raise RuntimeError("diffusers is required for local image generation. Install animaworks with Diffusers support.") from exc
    return AutoPipelineForText2Image, AutoPipelineForImage2Image


def _import_pil() -> tuple[Any, Any]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required for local Diffusers image generation.") from exc
    return Image, io.BytesIO


def _cache_dir_for_repo(repo_id: str) -> Path:
    return _HF_CACHE_ROOT / ("models--" + repo_id.replace("/", "--"))


def _resolve_snapshot_path(repo_id: str) -> str | None:
    cache_dir = _cache_dir_for_repo(repo_id)
    snapshots_dir = cache_dir / "snapshots"
    if not snapshots_dir.is_dir():
        return None

    main_ref = cache_dir / "refs" / "main"
    if main_ref.is_file():
        revision = main_ref.read_text(encoding="utf-8").strip()
        snapshot_dir = snapshots_dir / revision
        if snapshot_dir.is_dir():
            return str(snapshot_dir)

    snapshots = sorted((p for p in snapshots_dir.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True)
    if snapshots:
        return str(snapshots[0])
    return None


def _resolve_model_source(value: str | None) -> str:
    if value and value not in {"", "auto"}:
        return value

    env_model = os.getenv("ANIMAWORKS_DIFFUSERS_MODEL")
    if env_model:
        return env_model

    for repo_id in _AUTO_MODEL_REPOS:
        snapshot = _resolve_snapshot_path(repo_id)
        if snapshot:
            logger.info("Using cached Diffusers model snapshot: %s", snapshot)
            return snapshot

    return _AUTO_MODEL_REPOS[0]


class LocalDiffusersClient:
    """Local text/image generation using Hugging Face Diffusers."""

    def __init__(self, config: ImageGenConfig | None = None) -> None:
        from core.config.models import ImageGenConfig

        self._config = config or ImageGenConfig()
        self._device = self._resolve_device(getattr(self._config, "diffusers_device", "auto"))
        self._dtype_name = getattr(self._config, "diffusers_torch_dtype", "auto")
        self._local_files_only = bool(getattr(self._config, "diffusers_local_files_only", True))

        # Resolve text2img model: style-specific override > generic > auto
        image_style = getattr(self._config, "image_style", "realistic")
        style_model = ""
        if image_style == "realistic":
            style_model = getattr(self._config, "diffusers_text2img_model_realistic", "") or ""
        elif image_style == "anime":
            style_model = getattr(self._config, "diffusers_text2img_model_anime", "") or ""
        base_model = style_model or getattr(self._config, "diffusers_text2img_model", "auto")
        self._text2img_source = _resolve_model_source(base_model)

        img2img_value = getattr(self._config, "diffusers_img2img_model", "auto")
        if not img2img_value or img2img_value == "auto":
            img2img_value = self._text2img_source
        self._img2img_source = _resolve_model_source(img2img_value)
        logger.info("Diffusers model resolved: style=%s, text2img=%s", image_style, self._text2img_source)

    @staticmethod
    def _resolve_device(value: str) -> str:
        if value != "auto":
            return value
        torch = _import_torch()
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _resolve_torch_dtype(self) -> Any:
        torch = _import_torch()
        if self._device == "cpu":
            return torch.float32
        if self._dtype_name == "float32":
            return torch.float32
        if self._dtype_name == "bfloat16":
            return torch.bfloat16
        return torch.float16

    @staticmethod
    def _snap_size(width: int, height: int) -> tuple[int, int]:
        snapped_width = max(64, (width // 8) * 8)
        snapped_height = max(64, (height // 8) * 8)
        return snapped_width, snapped_height

    @staticmethod
    def _is_sdxl(model_source: str) -> bool:
        """Detect whether the model source points to an SDXL-class model."""
        lower = model_source.lower()
        sdxl_markers = ("stable-diffusion-xl", "sdxl", "realvis", "animagine")
        return any(m in lower for m in sdxl_markers)

    def _pipeline_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "torch_dtype": self._resolve_torch_dtype(),
            "local_files_only": self._local_files_only,
        }
        # safety_checker is only used by SD 1.x pipelines; SDXL ignores it
        if not self._is_sdxl(self._text2img_source):
            kwargs["safety_checker"] = None
            kwargs["requires_safety_checker"] = False
        return kwargs

    @staticmethod
    def _apply_scheduler(pipe: Any, model_source: str) -> None:
        """Replace the default scheduler with DPM++ 2M Karras for better quality."""
        try:
            from diffusers import DPMSolverMultistepScheduler
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                pipe.scheduler.config,
                algorithm_type="dpmsolver++",
                use_karras_sigmas=True,
            )
            logger.info("Scheduler set to DPM++ 2M Karras for %s", model_source)
        except Exception:
            logger.debug("Failed to set DPM++ scheduler, keeping default", exc_info=True)

    # VRAM thresholds for full-GPU mode.
    # RealVisXL_V5.0 (SDXL) memory budget in float16:
    #   UNet ~4.8 GB + VAE ~0.2 GB + CLIP ~1.3 GB = ~6.3 GB base
    #   Inference activations: ~0.7 GB with xFormers, ~3 GB without
    #   → need ≥7 GB free to run full-GPU with xFormers safely
    #   → need ≥10 GB free to run full-GPU without xFormers
    _VRAM_THRESHOLD_WITH_XFORMERS: int = 7 * 1024 ** 3
    _VRAM_THRESHOLD_NO_XFORMERS: int = 10 * 1024 ** 3

    @staticmethod
    def _xformers_available() -> bool:
        try:
            import xformers  # noqa: F401
            return True
        except ImportError:
            return False

    def _vram_offload_threshold(self) -> int:
        if self._xformers_available():
            return self._VRAM_THRESHOLD_WITH_XFORMERS
        return self._VRAM_THRESHOLD_NO_XFORMERS

    def _should_use_cpu_offload(self) -> bool:
        """Return True if free VRAM is below the threshold for full-VRAM mode."""
        if self._device != "cuda":
            return False
        try:
            torch = _import_torch()
            free, _ = torch.cuda.mem_get_info()
            threshold = self._vram_offload_threshold()
            needs_offload = free < threshold
            if needs_offload:
                logger.info(
                    "Free VRAM %.1fGB < %.0fGB threshold — enabling CPU offload for Diffusers",
                    free / 1024 ** 3,
                    threshold / 1024 ** 3,
                )
            return needs_offload
        except Exception:
            return False

    def _apply_memory_optimizations(self, pipe: Any, cpu_offload: bool) -> Any:
        """Apply VRAM-saving optimizations and optionally move to device.

        NOTE: xFormers and model_cpu_offload are mutually exclusive —
        do not enable xFormers when using CPU offload.
        """
        if cpu_offload:
            # CPU offload path: minimal VRAM usage, xFormers NOT compatible
            # VAE slicing still reduces peak decode VRAM
            try:
                pipe.vae.enable_slicing()
            except Exception:
                pass
            try:
                pipe.enable_model_cpu_offload()
                logger.info("Model CPU offload enabled (low-VRAM mode)")
                return pipe  # do NOT call .to(device) after cpu_offload
            except Exception:
                logger.warning("enable_model_cpu_offload failed, falling back to .to(device)", exc_info=True)
            return pipe.to(self._device)

        # Full-GPU path: use xFormers + slicing for maximum VRAM efficiency
        pipe = pipe.to(self._device)

        if self._xformers_available():
            try:
                pipe.enable_xformers_memory_efficient_attention()
                logger.info("xFormers memory-efficient attention enabled")
            except Exception:
                logger.debug("xFormers not applicable for this pipeline", exc_info=True)

        try:
            pipe.vae.enable_slicing()
            logger.debug("VAE slicing enabled")
        except Exception:
            pass

        try:
            pipe.enable_attention_slicing()
            logger.debug("Attention slicing enabled")
        except Exception:
            pass

        return pipe

    def _load_text2img_pipeline(self) -> Any:
        cache_key = ("text2img", self._text2img_source, self._device, self._dtype_name)
        cached = _PIPELINE_CACHE.get(cache_key)
        if cached is not None:
            return cached

        auto_text2img, _ = _import_diffusers()
        pipe = auto_text2img.from_pretrained(self._text2img_source, **self._pipeline_kwargs())
        cpu_offload = self._should_use_cpu_offload()
        pipe = self._apply_memory_optimizations(pipe, cpu_offload)
        self._apply_scheduler(pipe, self._text2img_source)
        _PIPELINE_CACHE[cache_key] = pipe
        return pipe

    def _load_img2img_pipeline(self) -> Any:
        cache_key = ("img2img", self._img2img_source, self._device, self._dtype_name)
        cached = _PIPELINE_CACHE.get(cache_key)
        if cached is not None:
            return cached

        _, auto_img2img = _import_diffusers()
        if self._img2img_source == self._text2img_source:
            base_pipe = self._load_text2img_pipeline()
            # Retire IP-Adapter before deriving img2img so bustup/expression
            # generation is not polluted by face-reference weights.
            text2img_key = ("text2img", self._text2img_source, self._device, self._dtype_name)
            self._retire_ip_adapter(text2img_key)
            pipe = auto_img2img.from_pipe(base_pipe)
        else:
            pipe = auto_img2img.from_pretrained(self._img2img_source, **self._pipeline_kwargs())
            cpu_offload = self._should_use_cpu_offload()
            pipe = self._apply_memory_optimizations(pipe, cpu_offload)
            self._apply_scheduler(pipe, self._img2img_source)
        _PIPELINE_CACHE[cache_key] = pipe
        return pipe

    @staticmethod
    def _to_png_bytes(image: Any) -> bytes:
        _, bytes_io_cls = _import_pil()
        buffer = bytes_io_cls()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    @staticmethod
    def _read_image(image_bytes: bytes) -> Any:
        image_cls, bytes_io_cls = _import_pil()
        image = image_cls.open(bytes_io_cls(image_bytes))
        return image.convert("RGB")

    @staticmethod
    def _crop_to_face(image: Any) -> Any:
        """Detect face via OpenCV Haar cascade and crop tightly around it.

        Returns the cropped face region (with margin) or the original image
        if detection fails.  This prevents background/clothing from leaking
        into IP-Adapter embeddings.
        """
        try:
            import cv2
            import numpy as np

            arr = np.array(image)
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml",
            )
            faces = cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
            )
            if len(faces) == 0:
                logger.info("No face detected in reference — using full image")
                return image

            # Pick the largest face
            x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
            # Expand by 50% for forehead/chin/cheek margin
            margin = int(max(w, h) * 0.5)
            ih, iw = arr.shape[:2]
            x1, y1 = max(0, x - margin), max(0, y - margin)
            x2, y2 = min(iw, x + w + margin), min(ih, y + h + margin)
            # Make square (IP-Adapter expects square input)
            side = max(x2 - x1, y2 - y1)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            x1 = max(0, cx - side // 2)
            y1 = max(0, cy - side // 2)
            x2 = min(iw, x1 + side)
            y2 = min(ih, y1 + side)
            # Adjust if clamped
            if x2 - x1 < side:
                x1 = max(0, x2 - side)
            if y2 - y1 < side:
                y1 = max(0, y2 - side)

            cropped = image.crop((x1, y1, x2, y2))
            logger.info(
                "Face detected: crop (%d,%d)-(%d,%d) from %dx%d",
                x1, y1, x2, y2, iw, ih,
            )
            return cropped
        except ImportError:
            logger.debug("OpenCV not available — skipping face crop")
            return image
        except Exception:
            logger.warning("Face detection failed — using full image", exc_info=True)
            return image

    @staticmethod
    def _make_generator(seed: int | None) -> Any:
        if seed is None:
            return None
        torch = _import_torch()
        return torch.Generator().manual_seed(seed)

    def _ensure_ip_adapter(self, pipe: Any, cache_key: tuple) -> None:
        """Load IP-Adapter face weights onto a pipeline (lazy, once per pipeline).

        Supports both SDXL and SD 1.5 class models with appropriate weights.
        IP-Adapter stays loaded across retries to avoid load/unload cycling
        which degrades UNet weights.  Call :meth:`_retire_ip_adapter` when
        switching to a non-face-reference generation mode.
        """
        if cache_key in _IP_ADAPTER_LOADED:
            return

        ip_model = getattr(self._config, "ip_adapter_model", "h94/IP-Adapter")

        if self._is_sdxl(self._text2img_source):
            ip_weight = "ip-adapter-plus-face_sdxl_vit-h.safetensors"
            subfolder = "sdxl_models"
        else:
            ip_weight = "ip-adapter-plus-face_sd15.bin"
            subfolder = "models"

        logger.info(
            "Loading IP-Adapter: %s (subfolder=%s, weight=%s, model=%s)",
            ip_model, subfolder, ip_weight, self._text2img_source,
        )

        # Try local cache first, then auto-download if not found.
        for attempt, local_only in enumerate((self._local_files_only, False)):
            try:
                pipe.load_ip_adapter(
                    ip_model,
                    subfolder=subfolder,
                    weight_name=ip_weight,
                    image_encoder_folder="models/image_encoder",
                    local_files_only=local_only,
                )
                _IP_ADAPTER_LOADED.add(cache_key)
                if attempt > 0:
                    logger.info("IP-Adapter downloaded and loaded successfully")
                else:
                    logger.info("IP-Adapter loaded from local cache")
                return
            except Exception:
                if attempt == 0 and self._local_files_only:
                    logger.info("IP-Adapter not in local cache — attempting download …")
                    continue
                logger.exception(
                    "Failed to load IP-Adapter (%s/%s) — face reference will use img2img fallback",
                    subfolder, ip_weight,
                )

    def _retire_ip_adapter(self, text2img_key: tuple) -> None:
        """Unload IP-Adapter from the cached text2img pipeline if present.

        Called once when switching away from face-reference mode (e.g. to
        plain text2img or img2img for bustup), NOT after every generation.
        """
        if text2img_key not in _IP_ADAPTER_LOADED:
            return
        pipe = _PIPELINE_CACHE.get(text2img_key)
        if pipe is None:
            _IP_ADAPTER_LOADED.discard(text2img_key)
            return
        try:
            pipe.unload_ip_adapter()
            logger.info("IP-Adapter retired (switching away from face reference mode)")
        except Exception:
            logger.warning("Failed to unload IP-Adapter", exc_info=True)
        _IP_ADAPTER_LOADED.discard(text2img_key)
        # Invalidate derived img2img cache so it is rebuilt cleanly.
        img2img_key = ("img2img", self._img2img_source, self._device, self._dtype_name)
        _PIPELINE_CACHE.pop(img2img_key, None)

    def _image2image_strength(self, requested: float | None = None) -> float:
        strength = requested
        if strength is None:
            strength = getattr(self._config, "diffusers_img2img_strength", 0.55)
        return max(0.15, min(0.85, float(strength)))

    def generate_fullbody(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 512,
        height: int = 768,
        seed: int | None = None,
        steps: int = 20,
        scale: float = 7.5,
        sampler: str = "k_euler_ancestral",
        vibe_image: bytes | None = None,
        vibe_strength: float = 0.6,
        vibe_info_extracted: float = 0.8,
        face_reference_image: bytes | None = None,
        step_callback: "Callable[[int, int], None] | None" = None,
    ) -> bytes:
        """Generate a full-body character image locally.

        If *face_reference_image* is provided, IP-Adapter (Plus Face) is used
        to inject the facial features from the reference into the generation.
        This takes priority over *vibe_image* when both are supplied.

        *step_callback(current_step, total_steps)* is called after each
        denoising step so callers can emit progress events.
        """
        del sampler, vibe_info_extracted

        # In CPU-offload mode, further reduce to keep generation under ~5 min.
        cpu_offload = self._should_use_cpu_offload()
        if cpu_offload:
            steps = min(steps, 10)             # cap at 10 steps
            width = min(width, 512)            # max 512×512
            height = min(height, 512)
            logger.info(
                "Low-VRAM CPU-offload mode: reduced to %d steps at %dx%d",
                steps, width, height,
            )

        width, height = self._snap_size(width, height)
        total_steps = max(1, steps)
        generator = self._make_generator(seed)

        # Build Diffusers callback for progress reporting
        _done_steps = [0]

        def _diffusers_callback(pipe_self: Any, step: int, timestep: Any, callback_kwargs: dict) -> dict:  # noqa: ARG001
            _done_steps[0] = step + 1
            if step_callback is not None:
                try:
                    step_callback(_done_steps[0], total_steps)
                except Exception:
                    pass
            return callback_kwargs

        common_kwargs: dict[str, Any] = {
            "prompt": prompt,
            "negative_prompt": negative_prompt or None,
            "guidance_scale": scale,
            "num_inference_steps": total_steps,
            "generator": generator,
            "callback_on_step_end": _diffusers_callback,
        }

        text2img_key = ("text2img", self._text2img_source, self._device, self._dtype_name)

        if face_reference_image is not None:
            # IP-Adapter face reference → text2img + adapter.
            # IP-Adapter stays loaded across retries to avoid repeated
            # load/unload cycles that degrade UNet weights.
            pipe = self._load_text2img_pipeline()
            self._ensure_ip_adapter(pipe, text2img_key)

            if text2img_key in _IP_ADAPTER_LOADED:
                # Use vibe_strength from the request so the UI slider
                # controls face influence.  Fall back to config default.
                ip_scale = vibe_strength if vibe_strength is not None else float(
                    getattr(self._config, "ip_adapter_scale", 0.6),
                )
                pipe.set_ip_adapter_scale(ip_scale)
                face_img = self._crop_to_face(self._read_image(face_reference_image))
                common_kwargs["ip_adapter_image"] = face_img
                logger.info("Generating with IP-Adapter face reference (scale=%.2f)", ip_scale)

                result = pipe(**common_kwargs, width=width, height=height)

                # Keep IP-Adapter loaded — no unload here.  Invalidate
                # img2img cache so bustup/expressions get a fresh pipeline
                # (via _retire_ip_adapter in _load_img2img_pipeline).
                img2img_key = ("img2img", self._img2img_source, self._device, self._dtype_name)
                _PIPELINE_CACHE.pop(img2img_key, None)
            else:
                # IP-Adapter unavailable (weights missing or download failed).
                # Fall back to img2img so the face reference still has an
                # effect rather than being silently ignored.
                #
                # Strength mapping is INVERTED for face reference: the UI
                # slider means "face influence" (higher = more face), but
                # img2img strength means "how much to regenerate" (higher =
                # less reference).  So we flip: slider 0.6 → strength 0.4.
                face_strength = 1.0 - (vibe_strength if vibe_strength is not None else 0.6)
                face_strength = max(0.15, min(0.85, face_strength))
                logger.warning(
                    "IP-Adapter not available — falling back to img2img with face reference "
                    "(strength=%.2f, i.e. %.0f%% face preserved)",
                    face_strength, (1 - face_strength) * 100,
                )
                pipe = self._load_img2img_pipeline()
                reference = self._crop_to_face(self._read_image(face_reference_image)).resize((width, height))
                result = pipe(
                    **common_kwargs,
                    image=reference,
                    strength=face_strength,
                )

        elif vibe_image is not None:
            # Switching away from face reference — retire IP-Adapter if loaded
            self._retire_ip_adapter(text2img_key)
            pipe = self._load_img2img_pipeline()
            reference = self._read_image(vibe_image).resize((width, height))
            result = pipe(
                **common_kwargs,
                image=reference,
                strength=self._image2image_strength(vibe_strength),
            )

        else:
            # Pure text2img — retire IP-Adapter if loaded
            self._retire_ip_adapter(text2img_key)
            pipe = self._load_text2img_pipeline()
            result = pipe(**common_kwargs, width=width, height=height)

        return self._to_png_bytes(result.images[0])

    def _run_img2img_pipeline(
        self,
        pipe: Any,
        prompt: str,
        negative_prompt: str,
        image: Any,
        guidance_scale: float,
        strength: float,
        generator: Any,
        steps: int,
    ) -> Any:
        """Run img2img with step fallback in case scheduler limits are hit."""
        max_attempts = 3
        attempt_steps = max(1, steps)
        kwargs = {
            "image": image,
            "guidance_scale": guidance_scale,
            "strength": strength,
            "generator": generator,
        }
        for attempt in range(max_attempts):
            pipe.scheduler.set_timesteps(attempt_steps)
            try:
                return pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt or None,
                    num_inference_steps=attempt_steps,
                    **kwargs,
                )
            except IndexError:
                if attempt == max_attempts - 1 or attempt_steps <= 5:
                    raise
                attempt_steps = max(5, attempt_steps - 2)


    def generate_from_reference(
        self,
        reference_image: bytes,
        prompt: str,
        aspect_ratio: str = "3:4",
        output_format: str = "png",
        guidance_scale: float = 5.5,
        seed: int | None = None,
        negative_prompt: str = "",
        strength: float | None = None,
    ) -> bytes:
        """Generate a derivative image from a reference image locally."""
        del output_format

        width, height = _ASPECT_SIZES.get(aspect_ratio, _ASPECT_SIZES["3:4"])
        generator = self._make_generator(seed)
        reference = self._read_image(reference_image).resize((width, height))
        pipe = self._load_img2img_pipeline()
        result = self._run_img2img_pipeline(
            pipe=pipe,
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=reference,
            guidance_scale=guidance_scale,
            strength=self._image2image_strength(strength),
            generator=generator,
            steps=max(1, int(getattr(self._config, "diffusers_num_inference_steps", 28))),
        )
        return self._to_png_bytes(result.images[0])


__all__ = [
    "LocalDiffusersClient",
]
