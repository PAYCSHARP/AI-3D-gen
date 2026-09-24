"""
Hunyuan3D 2.1 Cloud – Modly extension (v0.3, dvoukrokový režim).
Krok 1: tvar  -> Space `space_id`, endpoint shape_generation (60 s GPU, projde na free účtu)
Krok 2: textury -> tvůj Space `texture_space_id`, endpoint texture_only (< 120 s GPU)
Bez `texture_space_id` se použije původní generation_all (vyžaduje HF PRO).
Nastavení v config.json vedle tohoto souboru.
"""
import io
import json
import os
import random
import shutil
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Callable, Optional

from PIL import Image

from services.generators.base import BaseGenerator, smooth_progress, GenerationCancelled

_EXT_DIR = Path(__file__).resolve().parent
_DEFAULTS = {
    "space_id": "tencent/Hunyuan3D-2.1",
    "texture_space_id": "",
    "hf_token": "",
    "remove_background": True,
    "timeout_s": 900,
    "mode": "auto",  # auto = textury, při chybě vrátí aspoň tvar | textured | shape
}
MAX_SEED = 10_000_000


def _load_config() -> dict:
    cfg = dict(_DEFAULTS)
    try:
        cfg.update(json.loads((_EXT_DIR / "config.json").read_text(encoding="utf-8")))
    except Exception as exc:
        print(f"[Hunyuan3D21Cloud] config.json nelze načíst ({exc}), používám výchozí.")
    if os.environ.get("HF_TOKEN") and not cfg.get("hf_token"):
        cfg["hf_token"] = os.environ["HF_TOKEN"]
    return cfg


def _connect(space_id: str, token: Optional[str]):
    from gradio_client import Client
    try:
        return Client(space_id, hf_token=token)
    except TypeError:  # novější gradio_client
        return Client(space_id, token=token)


def _api_info(client) -> dict:
    try:
        return client.view_api(print_info=False, return_format="dict") or {}
    except Exception as exc:
        print(f"[Hunyuan3D21Cloud] view_api selhalo: {exc}")
        return {}


def _find_endpoint(info: dict, fn_name: str) -> str:
    names = list(info.get("named_endpoints", {}).keys())
    for n in names:
        if fn_name in n:
            return n
    if names:
        print(f"[Hunyuan3D21Cloud] Endpoint {fn_name} nenalezen, dostupné: {names}")
    return "/" + fn_name


class Hunyuan3D21CloudGenerator(BaseGenerator):
    MODEL_ID = "hunyuan3d21-cloud"
    DISPLAY_NAME = "Hunyuan3D 2.1 Cloud (HF Space)"
    VRAM_GB = 0

    # ---------------------------------------------------------------- #
    # Lifecycle
    # ---------------------------------------------------------------- #
    def is_downloaded(self) -> bool:
        return True

    def load(self) -> None:
        if self._model is not None:
            return
        cfg = _load_config()
        token = cfg.get("hf_token") or None
        print(f"[Hunyuan3D21Cloud] Připojuji se ke Space {cfg['space_id']} …")
        client = _connect(cfg["space_id"], token)
        self._info = _api_info(client)
        self._api_textured = _find_endpoint(self._info, "generation_all")
        self._api_shape = _find_endpoint(self._info, "shape_generation")

        self._tex_client = None
        if cfg.get("texture_space_id"):
            print(f"[Hunyuan3D21Cloud] Připojuji se k texturovacímu Space {cfg['texture_space_id']} …")
            self._tex_client = _connect(cfg["texture_space_id"], token)
            self._api_tex = _find_endpoint(_api_info(self._tex_client), "texture_only")

        self._cfg = cfg
        self._model = client
        print("[Hunyuan3D21Cloud] Připojeno.")

    def unload(self) -> None:
        super().unload()
        self._tex_client = None

    # ---------------------------------------------------------------- #
    # Inference
    # ---------------------------------------------------------------- #
    def generate(
        self,
        image_bytes: bytes,
        params: dict,
        progress_cb: Optional[Callable[[int, str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Path:
        from gradio_client import handle_file

        if self._model is None:
            self.load()

        seed = int(params.get("seed", -1))
        randomize = seed < 0
        if randomize:
            seed = random.randint(0, MAX_SEED)
        values = {
            "steps": int(params.get("num_inference_steps", 30)),
            "guidance_scale": float(params.get("guidance_scale", 5.0)),
            "seed": seed,
            "octree_resolution": int(params.get("octree_resolution", 256)),
            "check_box_rembg": bool(self._cfg.get("remove_background", True)),
            "num_chunks": 8000,
            "randomize_seed": randomize,
        }
        mode = str(self._cfg.get("mode", "auto")).lower()

        self._report(progress_cb, 3, "Připravuji obrázek…")
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp.close()
        Image.open(io.BytesIO(image_bytes)).convert("RGBA").save(tmp.name)

        stop_evt = threading.Event()
        if progress_cb:
            threading.Thread(
                target=smooth_progress,
                args=(progress_cb, 8, 90, "Generuji v cloudu (Hugging Face)…", stop_evt),
                daemon=True,
            ).start()

        try:
            if mode == "shape":
                glb = self._shape(tmp.name, values, cancel_event)
            elif self._tex_client is not None:
                # --- dvoukrokový režim ---
                shape_glb = self._shape(tmp.name, values, cancel_event)
                try:
                    res = self._run_job(self._tex_client, self._api_tex,
                                        [handle_file(shape_glb), handle_file(tmp.name)], cancel_event)
                    glb = self._pick_glb(res)
                except (RuntimeError, ValueError) as exc:
                    if mode == "auto":
                        print(f"[Hunyuan3D21Cloud] Textury selhaly, vracím jen tvar: {exc}")
                        glb = shape_glb
                    else:
                        raise
            else:
                # --- původní jednokrokový režim (PRO) ---
                try:
                    inputs = self._build_inputs(self._api_textured, handle_file(tmp.name), values)
                    glb = self._pick_glb(self._run_job(self._model, self._api_textured, inputs, cancel_event))
                except RuntimeError as exc:
                    if mode == "auto" and "duration" in str(exc).lower():
                        print("[Hunyuan3D21Cloud] Textury odmítnuty (limit GPU) -> jen tvar.")
                        glb = self._shape(tmp.name, values, cancel_event)
                    else:
                        raise
        finally:
            stop_evt.set()

        self._report(progress_cb, 95, "Ukládám GLB…")
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        out = self.outputs_dir / f"{int(time.time())}_{uuid.uuid4().hex[:8]}.glb"
        shutil.copyfile(glb, out)
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
        self._report(progress_cb, 100, "Hotovo")
        return out

    # ---------------------------------------------------------------- #
    # Helpers
    # ---------------------------------------------------------------- #
    def _shape(self, img_path, values, cancel_event) -> str:
        from gradio_client import handle_file
        inputs = self._build_inputs(self._api_shape, handle_file(img_path), values)
        return self._pick_glb(self._run_job(self._model, self._api_shape, inputs, cancel_event))

    def _run_job(self, client, api_name, inputs, cancel_event):
        try:
            job = client.submit(*inputs, api_name=api_name)
            deadline = time.time() + float(self._cfg.get("timeout_s", 900))
            while not job.done():
                if cancel_event is not None and cancel_event.is_set():
                    try:
                        job.cancel()
                    finally:
                        raise GenerationCancelled()
                if time.time() > deadline:
                    job.cancel()
                    raise RuntimeError("Vypršel časový limit čekání na Hugging Face Space.")
                time.sleep(1.0)
            return job.result()
        except GenerationCancelled:
            raise
        except Exception as exc:
            raise RuntimeError(
                f"Hugging Face Space vrátil chybu ({api_name}). Časté příčiny: Space je pozastavený, "
                "vyčerpaná denní ZeroGPU kvóta, limit délky GPU úlohy, nebo změněné API.\n"
                f"Původní chyba: {exc}"
            ) from exc

    def _build_inputs(self, api_name, image, values: dict) -> list:
        """Vstupy podle názvů parametrů z view_api (caption je gr.State a v API chybí)."""
        values = dict(values)
        values.update({"image": image, "caption": None,
                       "mv_image_front": None, "mv_image_back": None,
                       "mv_image_left": None, "mv_image_right": None})
        params = (self._info.get("named_endpoints", {}).get(api_name, {}) or {}).get("parameters", [])
        if not params:
            return [image, None, None, None, None,
                    values["steps"], values["guidance_scale"], values["seed"],
                    values["octree_resolution"], values["check_box_rembg"],
                    values["num_chunks"], values["randomize_seed"]]
        out, image_used = [], False
        for p in params:
            name = p.get("parameter_name") or p.get("label") or ""
            if name in values:
                out.append(values[name])
                image_used = image_used or name == "image"
            elif not image_used and "Image" in str(p.get("component", "")):
                out.append(image)
                image_used = True
            else:
                out.append(p.get("parameter_default"))
        return out

    @staticmethod
    def _pick_glb(result) -> str:
        items = list(result) if isinstance(result, (list, tuple)) else [result]

        def as_path(x):
            if isinstance(x, dict):
                x = x.get("value") or x.get("path") or x.get("name")
            return x if isinstance(x, str) and os.path.exists(x) else None

        # textured GLB bývá 2. výstup generation_all; jinak první .glb
        if len(items) > 1 and as_path(items[1]) and as_path(items[1]).lower().endswith(".glb"):
            return as_path(items[1])
        for x in items:
            p = as_path(x)
            if p and p.lower().endswith(".glb"):
                return p
        raise ValueError(f"Ve výstupu Space nebyl nalezen GLB soubor: {result!r}")

    @classmethod
    def params_schema(cls) -> list:
        manifest = json.loads((_EXT_DIR / "manifest.json").read_text(encoding="utf-8"))
        return manifest["nodes"][0]["params_schema"]
