"""
Hunyuan3D 2.1 Cloud – Modly extension.
Negeneruje lokálně: fotku pošle do Hugging Face Space (Gradio API, endpoint generation_all)
a stáhne hotový texturovaný GLB (PBR). Nastavení v config.json vedle tohoto souboru.
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
    "hf_token": "",
    "remove_background": True,
    "timeout_s": 900,
    "mode": "auto",   # "auto" = textury, při odmítnutí jen tvar | "textured" | "shape"
}
MAX_SEED = 10_000_000


def _load_config() -> dict:
    cfg = dict(_DEFAULTS)
    try:
        cfg.update(json.loads((_EXT_DIR / "config.json").read_text(encoding="utf-8")))
    except Exception as exc:
        print(f"[Hunyuan3D21Cloud] config.json nelze načíst ({exc}), používám výchozí.")
    token = os.environ.get("HF_TOKEN")
    if token and not cfg.get("hf_token"):
        cfg["hf_token"] = token
    return cfg


class Hunyuan3D21CloudGenerator(BaseGenerator):
    MODEL_ID = "hunyuan3d21-cloud"
    DISPLAY_NAME = "Hunyuan3D 2.1 Cloud (HF Space)"
    VRAM_GB = 0

    # ---------------------------------------------------------------- #
    # Lifecycle – nic se nestahuje, "model" je jen klient na Space
    # ---------------------------------------------------------------- #
    def is_downloaded(self) -> bool:
        return True

    def load(self) -> None:
        if self._model is not None:
            return
        from gradio_client import Client

        cfg = _load_config()
        token = cfg.get("hf_token") or None
        print(f"[Hunyuan3D21Cloud] Připojuji se ke Space {cfg['space_id']} …")
        try:
            client = Client(cfg["space_id"], hf_token=token)
        except TypeError:  # novější gradio_client používá 'token'
            client = Client(cfg["space_id"], token=token)
        self._cfg = cfg
        try:
            self._api_info = client.view_api(print_info=False, return_format="dict") or {}
        except Exception as exc:
            print(f"[Hunyuan3D21Cloud] view_api selhalo: {exc}")
            self._api_info = {}
        self._api_textured = self._find_endpoint(client, "generation_all")
        self._api_shape = self._find_endpoint(client, "shape_generation")
        self._model = client
        print(f"[Hunyuan3D21Cloud] Připojeno: {self._api_textured} | {self._api_shape}")

    def unload(self) -> None:
        super().unload()

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

        steps = int(params.get("num_inference_steps", 30))
        octree = int(params.get("octree_resolution", 256))
        guidance = float(params.get("guidance_scale", 5.0))
        seed = int(params.get("seed", -1))
        randomize = seed < 0
        if randomize:
            seed = random.randint(0, MAX_SEED)

        self._report(progress_cb, 3, "Připravuji obrázek…")
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp.close()
        Image.open(io.BytesIO(image_bytes)).convert("RGBA").save(tmp.name)

        self._report(progress_cb, 8, "Odesílám do Hugging Face Space (fronta)…")
        stop_evt = threading.Event()
        if progress_cb:
            threading.Thread(
                target=smooth_progress,
                args=(progress_cb, 10, 90, "Generuji tvar + PBR textury v cloudu…", stop_evt),
                daemon=True,
            ).start()

        mode = str(self._cfg.get("mode", "auto")).lower()
        args = {
            "steps": steps,
            "guidance_scale": guidance,
            "seed": seed,
            "octree_resolution": octree,
            "check_box_rembg": bool(self._cfg.get("remove_background", True)),
            "num_chunks": 8000,
            "randomize_seed": randomize,
        }
        try:
            if mode == "shape":
                result = self._run_job(self._api_shape, tmp.name, args, cancel_event)
            else:
                try:
                    result = self._run_job(self._api_textured, tmp.name, args, cancel_event)
                except RuntimeError as exc:
                    if mode == "auto" and "duration" in str(exc).lower():
                        print("[Hunyuan3D21Cloud] Textury odmítnuty (limit GPU času) -> generuji jen tvar.")
                        self._report(progress_cb, 50, "Limit GPU: generuji jen tvar (bez textur)…")
                        result = self._run_job(self._api_shape, tmp.name, args, cancel_event)
                    else:
                        raise
        finally:
            stop_evt.set()
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

        glb_src = self._pick_glb(result)
        self._report(progress_cb, 95, "Ukládám GLB…")
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        out = self.outputs_dir / f"{int(time.time())}_{uuid.uuid4().hex[:8]}.glb"
        shutil.copyfile(glb_src, out)
        self._report(progress_cb, 100, "Hotovo")
        return out

    # ---------------------------------------------------------------- #
    # Helpers
    # ---------------------------------------------------------------- #
    def _run_job(self, api_name, img_path, args, cancel_event):
        from gradio_client import handle_file
        try:
            inputs = self._build_inputs(api_name, handle_file(img_path), args)
            job = self._model.submit(*inputs, api_name=api_name)
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
                f"Space: {self._cfg.get('space_id')}\nPůvodní chyba: {exc}"
            ) from exc

    def _build_inputs(self, api_name, image, values: dict) -> list:
        """Sestaví vstupy podle názvů parametrů z view_api (pořadí se může lišit od kódu Space)."""
        values = dict(values)
        values.update({"image": image, "caption": None,
                       "mv_image_front": None, "mv_image_back": None,
                       "mv_image_left": None, "mv_image_right": None})
        params = (self._api_info.get("named_endpoints", {}).get(api_name, {}) or {}).get("parameters", [])
        if not params:
            # fallback: caption je gr.State, v API chybí -> začínáme obrázkem
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
        print(f"[Hunyuan3D21Cloud] {api_name} parametry: {[p.get('parameter_name') for p in params]}")
        return out

    @staticmethod
    def _find_endpoint(client, fn_name: str) -> str:
        try:
            info = client.view_api(print_info=False, return_format="dict")
            names = list((info or {}).get("named_endpoints", {}).keys())
        except Exception:
            names = []
        for n in names:
            if fn_name in n:
                return n
        if names:
            print(f"[Hunyuan3D21Cloud] Endpoint {fn_name} nenalezen, dostupné: {names}")
        return "/" + fn_name

    @staticmethod
    def _pick_glb(result) -> str:
        """Z výstupu generation_all vybere texturovaný GLB (2. výstup), jinak jakýkoli .glb."""
        items = list(result) if isinstance(result, (list, tuple)) else [result]

        def as_path(x):
            if isinstance(x, dict):
                x = x.get("value") or x.get("path") or x.get("name")
            return x if isinstance(x, str) and os.path.exists(x) else None

        if len(items) > 1 and as_path(items[1]) and as_path(items[1]).lower().endswith(".glb"):
            return as_path(items[1])
        for x in items:
            p = as_path(x)
            if p and p.lower().endswith(".glb"):
                return p
        raise RuntimeError(f"Ve výstupu Space nebyl nalezen GLB soubor: {result!r}")

    @classmethod
    def params_schema(cls) -> list:
        manifest = json.loads((_EXT_DIR / "manifest.json").read_text(encoding="utf-8"))
        return manifest["nodes"][0]["params_schema"]
