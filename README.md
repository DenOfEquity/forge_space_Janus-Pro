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

| prompt | size | image | note |
|---|---|---|---|
| The image depicts a unique landscape with geothermal pools in the foreground, surrounded by arid terrain and distant mountains under a starry sky. Above the landscape, there is a depiction of the Milky Way galaxy, with red stars and nebulae visible. | 384x512 | ![example](https://github.com/user-attachments/assets/3f465c68-4379-49a9-9cd5-5a23e3ba07a7) | low CFG, mid temperature |
| a beautiful woman with her face half covered by golden paste, the other half is dark purple. one eye is yellow and the other is green. closeup, professional shot (from [this reddit post](https://old.reddit.com/r/StableDiffusion/comments/1ieliyz/janus_pro_1b_offers_great_prompt_adherence/))  | 384x560 | ![example2](https://github.com/user-attachments/assets/101e2152-0c0c-4fc7-b96e-2131f53a3576) | increasing height too far tends to fade/blur the lower part |
| digital art shows a person with blue hair styled in a pony tail. They have a black choker around their neck and wear a red velvet jacket. The background is a bustling rustic market. | 384x384 | ![example3](https://github.com/user-attachments/assets/1bd43fbe-5560-4080-ad10-d55593e014d8) | mid CFG / higher temperature seems better for stylized images |



