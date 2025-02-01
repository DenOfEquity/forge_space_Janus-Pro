## Forge2 Spaces implementation of Janus Pro 1B Chat ##
New Forge only.

an adaptation of:
* https://huggingface.co/spaces/deepseek-ai/Janus-Pro-7B

will download ~8GB because it fetches both pytorch model and safetensors model. Don't know why.

May need installation of some additional dependencies:
```
attrdict
```

I've reworked the UI.

Model is capable of generating variable height (including > 384px), but changing width causes distortion.

>[!NOTE]
>Install via *Extensions* tab; *Install from URL* sub-tab; use URL of this repo.

>[!TIP]
>You can edit `forge-app.py` line 17 to `use_7B = True` to use the larger model instead, quantized to 8 bit (comment out the two `quantization_config=` if you want), but it is too slow to be usable for me. Download size is 13.8GB.

I haven't added any form of manual offloading; standard diffusers model offloading or sequential offloading don't work.

---
example : 384x512

![example](https://github.com/user-attachments/assets/96ae122d-70f1-43ea-b5c9-a104474478d1)
