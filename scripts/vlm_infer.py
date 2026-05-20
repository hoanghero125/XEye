"""VLM inference pipeline: SmolVLM-256M (image→EN) + opus-mt-en-vi (EN→VI)."""
import time
from pathlib import Path

import ctranslate2
import numpy as np
import onnxruntime as ort
from PIL import Image
from transformers import AutoConfig, AutoProcessor, MarianTokenizer

ROOT            = Path(__file__).resolve().parent.parent
VLM_DIR         = ROOT / "models" / "vlm"
NMT_EN_VI_DIR   = ROOT / "models" / "nmt" / "opus-mt-en-vi-int8"
NMT_VI_EN_DIR   = ROOT / "models" / "nmt" / "opus-mt-vi-en-int8"
ONNX_DIR        = VLM_DIR / "onnx"

MAX_NEW_TOKENS = 256


class VLMPipeline:
    def __init__(self):
        print("[VLM] Loading config and processor ...")
        self.config    = AutoConfig.from_pretrained(str(VLM_DIR))
        self.processor = AutoProcessor.from_pretrained(str(VLM_DIR))

        text_cfg             = self.config.text_config
        # Use tokenizer eos_token_id — covers added special tokens like <end_of_utterance>
        self.eos_token_id    = self.processor.tokenizer.eos_token_id
        self.image_token_idx = self.config.image_token_id
        self._num_kv_heads   = text_cfg.num_key_value_heads
        self._head_dim       = text_cfg.head_dim

        providers = ["CPUExecutionProvider"]
        so = ort.SessionOptions()
        so.inter_op_num_threads = 4
        so.intra_op_num_threads = 4

        print("[VLM] Loading ONNX sessions (FP32) ...")
        self.vision_sess  = ort.InferenceSession(str(ONNX_DIR / "vision_encoder.onnx"),       sess_options=so, providers=providers)
        self.embed_sess   = ort.InferenceSession(str(ONNX_DIR / "embed_tokens.onnx"),          sess_options=so, providers=providers)
        self.decoder_sess = ort.InferenceSession(str(ONNX_DIR / "decoder_model_merged.onnx"),  sess_options=so, providers=providers)

        # Infer KV cache keys from decoder input names
        self._kv_keys = sorted(
            [inp.name for inp in self.decoder_sess.get_inputs() if inp.name.startswith("past_key_values.")],
            key=lambda s: (int(s.split(".")[1]), s.split(".")[2]),
        )
        self._num_layers = text_cfg.num_hidden_layers

        print("[NMT] Loading EN→VI translator ...")
        self.en_vi       = ctranslate2.Translator(str(NMT_EN_VI_DIR), device="cpu", inter_threads=2)
        self.en_vi_tok   = MarianTokenizer.from_pretrained(str(NMT_EN_VI_DIR))

        print("[NMT] Loading VI→EN translator ...")
        self.vi_en       = ctranslate2.Translator(str(NMT_VI_EN_DIR), device="cpu", inter_threads=2)
        self.vi_en_tok   = MarianTokenizer.from_pretrained(str(NMT_VI_EN_DIR))

        print("[Pipeline] Ready.\n")

    def _init_cache(self, batch_size: int) -> dict:
        return {
            f"past_key_values.{layer}.{kv}": np.zeros(
                [batch_size, self._num_kv_heads, 0, self._head_dim], dtype=np.float32
            )
            for layer in range(self._num_layers)
            for kv in ("key", "value")
        }

    def describe(self, image: Image.Image, prompt: str = "Describe this image.") -> str:
        messages = [{
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": prompt}],
        }]
        text   = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=text, images=[image], return_tensors="np", do_image_splitting=False)

        input_ids      = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        position_ids   = np.cumsum(attention_mask, axis=-1)
        pixel_values   = inputs["pixel_values"]
        pix_attn_mask  = inputs["pixel_attention_mask"].astype(np.bool_)

        batch_size     = input_ids.shape[0]
        past_cache     = self._init_cache(batch_size)
        image_features = None
        generated      = np.array([[]], dtype=np.int64)

        t0 = time.time()
        for _ in range(MAX_NEW_TOKENS):
            embeds = self.embed_sess.run(None, {"input_ids": input_ids})[0]

            if image_features is None:
                print("[VLM] Running vision encoder ...", flush=True)
                image_features = self.vision_sess.run(
                    ["image_features"],
                    {"pixel_values": pixel_values, "pixel_attention_mask": pix_attn_mask},
                )[0]
                img_mask = inputs["input_ids"] == self.image_token_idx
                n_img_tokens = img_mask.sum()
                embeds[img_mask] = image_features.reshape(-1, image_features.shape[-1])
                print("[VLM] Prefill ...", flush=True)

            logits, *present_cache = self.decoder_sess.run(None, dict(
                inputs_embeds=embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **past_cache,
            ))

            input_ids      = logits[:, -1].argmax(-1, keepdims=True)
            attention_mask = np.concatenate([attention_mask, np.ones((batch_size, 1), dtype=attention_mask.dtype)], axis=-1)
            position_ids   = position_ids[:, -1:] + 1
            for j, key in enumerate(past_cache):
                past_cache[key] = present_cache[j]

            generated = np.concatenate([generated, input_ids], axis=-1)
            if (input_ids == self.eos_token_id).all():
                break

        elapsed = time.time() - t0
        n_tok   = generated.shape[-1]
        print(f"[VLM] {n_tok} tokens in {elapsed:.1f}s ({n_tok/elapsed:.1f} tok/s)")
        return self.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()

    def _translate(self, text: str, translator, tokenizer) -> str:
        encoded    = tokenizer([text], return_tensors=None, padding=False)
        src_tokens = [tokenizer.convert_ids_to_tokens(ids) for ids in encoded["input_ids"]]
        results    = translator.translate_batch(src_tokens)
        tgt_tokens = results[0].hypotheses[0]
        tgt_ids    = tokenizer.convert_tokens_to_ids(tgt_tokens)
        return tokenizer.decode(tgt_ids, skip_special_tokens=True)

    def vi_to_en(self, text: str) -> str:
        return self._translate(text, self.vi_en, self.vi_en_tok)

    def en_to_vi(self, text: str) -> str:
        return self._translate(text, self.en_vi, self.en_vi_tok)

    def run(self, image_path: str, question_vi: str = "") -> dict:
        image = Image.open(image_path).convert("RGB")
        image.thumbnail((1024, 512))

        if question_vi.strip():
            en_prompt = self.vi_to_en(question_vi)
            print(f"[NMT] VI→EN: {question_vi!r} → {en_prompt!r}")
        else:
            en_prompt = "Describe the scene in front of me."

        en_text = self.describe(image, en_prompt)
        vi_text = self.en_to_vi(en_text)
        return {"question_en": en_prompt, "en": en_text, "vi": vi_text}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to image file")
    parser.add_argument("--question", default="", help="Question in Vietnamese (optional)")
    args = parser.parse_args()

    pipeline = VLMPipeline()
    result   = pipeline.run(args.image, args.question)
    print(f"\n{result['vi']}")
