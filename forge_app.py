import gradio as gr
import torch
from transformers import AutoConfig, AutoModelForCausalLM
from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig

from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images
from PIL import Image

import numpy as np
import os
import gc

import spaces
from modules.processing import get_fixed_seed

use_7B = False

# Load model and processor
if use_7B:
    model_path = "deepseek-ai/Janus-Pro-7B"
    config = AutoConfig.from_pretrained(model_path)
    language_config = config.language_config
    language_config._attn_implementation = 'eager'
    quant_config = TransformersBitsAndBytesConfig(load_in_8bit=True,)
    vl_gpt = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=quant_config,
                                                 trust_remote_code=False,
                                                 torch_dtype=torch.bfloat16,)

    vl_chat_processor = VLChatProcessor.from_pretrained(model_path, quantization_config=quant_config, 
                                                        torch_dtype=torch.bfloat16,)
else:
    model_path = "deepseek-ai/Janus-Pro-1B"
    config = AutoConfig.from_pretrained(model_path)
    language_config = config.language_config
    language_config._attn_implementation = 'eager'
    vl_gpt = MultiModalityCausalLM.from_pretrained(model_path,
                                                 torch_dtype=torch.bfloat16,).cuda()

    vl_chat_processor = VLChatProcessor.from_pretrained(model_path, torch_dtype=torch.bfloat16,)


@torch.inference_mode()
def multimodal_understanding(image, question, seed, top_p, temperature):
    torch.cuda.empty_cache()
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    
    conversation = [
        {
            "role": "<|User|>",
            "content": f"<image_placeholder>\n{question}",
            "images": [image],
        },
        {"role": "<|Assistant|>", "content": ""},
    ]
    
    pil_images = [Image.fromarray(image)]
    prepare_inputs = vl_chat_processor(
        conversations=conversation, images=pil_images, force_batchify=True
    ).to(device='cuda', dtype=torch.bfloat16)
    
    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)
    
    outputs = vl_gpt.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=vl_chat_processor.tokenizer.eos_token_id,
        bos_token_id=vl_chat_processor.tokenizer.bos_token_id,
        eos_token_id=vl_chat_processor.tokenizer.eos_token_id,
        max_new_tokens=512,
        do_sample=False if temperature == 0 else True,
        use_cache=True,
        temperature=temperature,
        top_p=top_p,
    )
    
    answer = vl_chat_processor.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    return answer

@torch.inference_mode()
def generate(input_ids,
             width,
             height,# w*h = 576*x*x

             temperature: float = 1,
             parallel_size: int = 5,
             cfg_weight: float = 5,
             image_token_num_per_image: int = 576,
             patch_size: int = 16):

    torch.cuda.empty_cache()
    
    tokens = torch.zeros((parallel_size * 2, len(input_ids)), dtype=torch.int).to('cuda')
    for i in range(parallel_size * 2):
        tokens[i, :] = input_ids
        if i % 2 != 0:
            tokens[i, 1:-1] = vl_chat_processor.pad_id
    inputs_embeds = vl_gpt.language_model.get_input_embeddings()(tokens)
    generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int).to('cuda')



    pkv = None
    for i in range(image_token_num_per_image):
        with torch.no_grad():
            outputs = vl_gpt.language_model.model(inputs_embeds=inputs_embeds,
                                                use_cache=True,
                                                past_key_values=pkv)
            pkv = outputs.past_key_values
            hidden_states = outputs.last_hidden_state
            logits = vl_gpt.gen_head(hidden_states[:, -1, :])
            logit_cond = logits[0::2, :]
            logit_uncond = logits[1::2, :]
            logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated_tokens[:, i] = next_token.squeeze(dim=-1)
            next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)

            img_embeds = vl_gpt.prepare_gen_img_embeds(next_token)
            inputs_embeds = img_embeds.unsqueeze(dim=1)

    

    patches = vl_gpt.gen_vision_model.decode_code(generated_tokens.to(dtype=torch.int),
                                                 shape=[parallel_size, 8, width // patch_size, height // patch_size])

    return generated_tokens.to(dtype=torch.int), patches

def unpack(dec, width, height, parallel_size=5):
    dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
    dec = np.clip((dec + 1) / 2 * 255, 0, 255)

    # visual_img = np.zeros((parallel_size, width, height, 3), dtype=np.uint8)
    visual_img = np.zeros(dec.shape, dtype=np.uint8)

    visual_img[:, :, :] = dec

    return visual_img


@torch.inference_mode()
def generate_image(prompt,
                   seed=None,
                   random=False,
                   image_count=1,
                   guidance=5,
                   t2i_temperature=1.0,
                   width=384,
                   height=384,
                   patch_size=16):
    # Clear CUDA cache and avoid tracking gradients
    torch.cuda.empty_cache()
    
    seed = get_fixed_seed(-1 if random else seed)

    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
    
    image_tokens = (width // patch_size) * (height // patch_size)
    
    parallel_size = image_count
    
    with torch.no_grad():
        messages = [{'role': '<|User|>', 'content': prompt},
                    {'role': '<|Assistant|>', 'content': ''}]
        text = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(conversations=messages,
                                                                   sft_format=vl_chat_processor.sft_format,
                                                                   system_prompt='')
        text = text + vl_chat_processor.image_start_tag
        
        input_ids = torch.LongTensor(vl_chat_processor.tokenizer.encode(text))
        output, patches = generate(input_ids,
                                   height // patch_size * patch_size,
                                   width // patch_size * patch_size,
                                   cfg_weight=guidance,
                                   parallel_size=parallel_size,
                                   temperature=t2i_temperature,
                                   image_token_num_per_image=image_tokens,
                                   patch_size=patch_size)
        images = unpack(patches,
                        height // patch_size * patch_size,
                        width // patch_size * patch_size,
                        parallel_size=parallel_size)

        return images, seed

def unload():
    global vl_gpt, vl_chat_processor
    del vl_gpt, vl_chat_processor
    gc.collect()
    torch.cuda.empty_cache()


# Gradio interface
with gr.Blocks(analytics_enabled=False) as demo:
    gr.Markdown(value="# Janus Pro (1B)")
    
    with gr.Tab("Multimodal Understanding"):
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(height="70vh")
                with gr.Row():
                    und_seed_input = gr.Number(label="Seed", precision=0, value=42)
                    top_p = gr.Slider(minimum=0, maximum=1, value=0.95, step=0.05, label="top_p")
                    temperature = gr.Slider(minimum=0.05, maximum=1, value=0.1, step=0.05, label="temperature")


            with gr.Column():
                question_input = gr.Textbox(label="Question")
                understanding_button = gr.Button("Chat")
                understanding_output = gr.Textbox(label="Response")

                def toggleChatOn():
                    return gr.Button("Chat", interactive=True, variant="primary")
                def toggleChatOff():
                    return gr.Button("...", interactive=False, variant="secondary")

                question_input.submit(fn=toggleChatOff, outputs=[understanding_button]).then(
                    multimodal_understanding,
                    inputs=[image_input, question_input, und_seed_input, top_p, temperature],
                    outputs=understanding_output
                ).then(fn=toggleChatOn, outputs=[understanding_button])
                understanding_button.click(fn=toggleChatOff, outputs=[understanding_button]).then(
                    multimodal_understanding,
                    inputs=[image_input, question_input, und_seed_input, top_p, temperature],
                    outputs=understanding_output
                ).then(fn=toggleChatOn, outputs=[understanding_button])

        examples_inpainting = gr.Examples(
            label="Multimodal Understanding examples",
            examples=[
                [
                    "explain this meme",
                    "doge.png",
                ],
                [
                    "Convert the formula into latex code.",
                    "equation.png",
                ],
            ],
            inputs=[question_input, image_input],
        )
        

    with gr.Tab("Text-to-Image"):
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    cfg_weight_input = gr.Slider(minimum=1, maximum=10, value=2, step=0.1, label="CFG Weight")
                    t2i_temperature = gr.Slider(minimum=0.05, maximum=1.0, value=0.5, step=0.05, label="temperature")

                prompt_input = gr.Textbox(label="Prompt (more detail can help produce better images!)")
                with gr.Row():
                    seed = gr.Number(label="Seed", precision=0, value=1234)
                    random = gr.Checkbox(label="Random seed", value=True)
                    count = gr.Number(label="Batch size", minimum=1, maximum=9, step=1, value=1, scale=0)

                with gr.Row(visible=True):
                    width  = gr.Slider(label="Width (fixed)",  step=16, value=384, minimum=160, maximum=768, interactive=False)
                    height = gr.Slider(label="Height", step=16, value=384, minimum=160, maximum=768)
                    patch_size =  gr.Number(label="Patch size", step=1, value=16, scale=1, visible=False)   # seems like hardcoded to 16 somewhere
                    # 576(tokens)=24*24; 24*16(patch) = 384(width, height)
                    # changing size: broken incoherent output

                examples_t2i = gr.Examples(
                    label="Text to image generation examples.",
                    examples=[
                        "Master shifu racoon wearing drip attire as a street gangster.",
                        "The face of a beautiful girl",
                        "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
                        "A cute and adorable baby fox with big brown eyes, autumn leaves in the background enchanting,immortal,fluffy, shiny mane,Petals,fairyism,unreal engine 5 and Octane Render,highly detailed, photorealistic, cinematic, natural colors.",
                        "The image features an intricately designed eye set against a circular backdrop adorned with ornate swirl patterns that evoke both realism and surrealism. At the center of attention is a strikingly vivid blue iris surrounded by delicate veins radiating outward from the pupil to create depth and intensity. The eyelashes are long and dark, casting subtle shadows on the skin around them which appears smooth yet slightly textured as if aged or weathered over time.\n\nAbove the eye, there's a stone-like structure resembling part of classical architecture, adding layers of mystery and timeless elegance to the composition. This architectural element contrasts sharply but harmoniously with the organic curves surrounding it. Below the eye lies another decorative motif reminiscent of baroque artistry, further enhancing the overall sense of eternity encapsulated within each meticulously crafted detail. \n\nOverall, the atmosphere exudes a mysterious aura intertwined seamlessly with elements suggesting timelessness, achieved through the juxtaposition of realistic textures and surreal artistic flourishes. Each component\u2014from the intricate designs framing the eye to the ancient-looking stone piece above\u2014contributes uniquely towards creating a visually captivating tableau imbued with enigmatic allure.",
                    ],
                    inputs=prompt_input,
                )

            with gr.Column():
                generation_button = gr.Button("Generate Images", variant="primary")

                image_output = gr.Gallery(label="Generated Images", columns=2, rows=2, height="60vh", preview=True)
        
                def toggleGenerateOn():
                    return gr.Button("Generate Images", interactive=True, variant="primary")
                def toggleGenerateOff():
                    return gr.Button("...", interactive=False, variant="secondary")
        
        
            prompt_input.submit(fn=toggleGenerateOff, outputs=[generation_button]).then(
                fn=generate_image,
                inputs=[prompt_input, seed, random, count, cfg_weight_input, t2i_temperature, width, height, patch_size],
                outputs=[image_output, seed]
            ).then(fn=toggleGenerateOn, outputs=[generation_button])
            generation_button.click(fn=toggleGenerateOff, outputs=[generation_button]).then(
                fn=generate_image,
                inputs=[prompt_input, seed, random, count, cfg_weight_input, t2i_temperature, width, height, patch_size],
                outputs=[image_output, seed]
            ).then(fn=toggleGenerateOn, outputs=[generation_button])
    
    demo.unload(fn=unload)


if __name__ == "__main__":
    demo.launch()
