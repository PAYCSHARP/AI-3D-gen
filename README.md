# modly-hunyuan3d21-cloud

Modly rozšíření: Hunyuan3D 2.1 (tvar + PBR textury) přes Hugging Face Space.
Nepotřebuje lokální GPU – obrázek se pošle na Gradio API Space `generation_all`.

## Nastavení (`config.json`)
- `space_id` – Space s Hunyuan3D 2.1 (výchozí `tencent/Hunyuan3D-2.1`; lze použít běžící duplikát)
- `hf_token` – volitelný HF token (Read); s přihlášeným účtem je vyšší ZeroGPU kvóta
- `remove_background` – odstranění pozadí na straně Space
- `timeout_s` – max. čekání na výsledek

## Omezení
- ZeroGPU kvóta: free účet má jen pár minut GPU denně, jedna generace rezervuje až ~3 min.
- Pokud je Space „Paused“, nefunguje – nastav jiný `space_id`.
- Hunyuan3D licence: Tencent Hunyuan Non-Commercial.
