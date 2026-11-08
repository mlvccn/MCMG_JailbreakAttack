import re
import torch

# Match Qwen2.5-VL's inference chat template exactly: when callers omit an
# explicit system message, the processor inserts this default system turn.
qwen_chatbot_prompt = (
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n<ImageHere>%s<|im_end|>\n"
    "<|im_start|>assistant\n"
)


class Prompt:
    """Qwen-friendly prompt wrapper compatible with existing work1 attacker API.

    Notes:
    - Text segments are tokenized with Qwen tokenizer (processor.tokenizer preferred).
    - Image embeddings still require model.encode_img for gradient-based image attacks.
    """

    def __init__(self, model, text_prompts=None, img_prompts=None, device='cuda:0', max_new_tokens=300, max_length=2000):
        self.model = model
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.max_length = max_length

        self.text_prompts = text_prompts
        self.img_prompts = img_prompts

        self.text_embs = []
        self.img_embs = []
        self.context_embs = []

        self.offset = []
        self.adv_len = 0
        self.adv_suffix_tokens = None

        self.tokenizer = getattr(self.model, 'tokenizer', None)
        if self.tokenizer is None and hasattr(self.model, 'processor'):
            self.tokenizer = getattr(self.model.processor, 'tokenizer', None)
        if self.tokenizer is None and hasattr(self.model, 'llama_tokenizer'):
            self.tokenizer = self.model.llama_tokenizer
        if self.tokenizer is None:
            raise AttributeError('Model does not expose tokenizer or processor.tokenizer')

        self.text_model = getattr(self.model, 'language_model', None)
        if self.text_model is None:
            self.text_model = getattr(self.model, 'llama_model', self.model)

        if hasattr(self.text_model, 'model') and hasattr(self.text_model.model, 'embed_tokens'):
            self.embed_tokens = self.text_model.model.embed_tokens
        else:
            self.embed_tokens = self.text_model.get_input_embeddings()

        # Estimated image token length for token-gradient offset; configurable on model.
        self.image_token_estimate = int(getattr(self.model, 'image_token_estimate', 256))

        self.text_embs = self.generate_text_embedding(self.text_prompts)
        self.img_embs = self.generate_img_embedding(self.img_prompts)
        self.update_context_embs()

    def update_context_embs(self):
        if len(self.text_embs) == len(self.img_embs):
            self.context_embs = self.generate_context_embedding(self.text_embs, self.img_embs)
        else:
            self.context_embs = []

    def update_cand_context_embs(self):
        if len(self.text_embs) == len(self.img_embs):
            self.context_embs = self.generate_context_embedding(self.text_embs, self.img_embs, False)
        else:
            self.context_embs = []

    def update_text_prompt(self, text_prompts):
        self.text_prompts = text_prompts
        self.text_embs = self.generate_text_embedding(self.text_prompts)
        self.update_context_embs()

    def update_img_prompts(self, img_prompts):
        self.img_prompts = img_prompts
        self.img_embs = self.generate_img_embedding(self.img_prompts)
        self.update_context_embs()

    def generate_text_embedding(self, text_prompts):
        if text_prompts is None:
            return []

        self.offset = []
        text_embs = []

        for item in text_prompts:
            prompt_segs = re.split('<ImageHere>|<adv_split_start>|</adv_split_end>', item)
            seg_tokens = [
                self.tokenizer(seg, return_tensors='pt', add_special_tokens=(i == 0)).to(self.device).input_ids.long()
                for i, seg in enumerate(prompt_segs)
            ]

            # offset = prefix text + image placeholder tokens
            item_offset = seg_tokens[0].shape[1] + seg_tokens[1].shape[1] + self.image_token_estimate
            self.offset.append(item_offset)

            self.adv_len = seg_tokens[2].shape[1]
            self.adv_suffix_tokens = seg_tokens[2].squeeze()

            embs = [self.embed_tokens(seg_t) for seg_t in seg_tokens]
            text_embs.append(embs)

        return text_embs

    def generate_img_embedding(self, img_prompts):
        if img_prompts is None:
            return []

        if not hasattr(self.model, 'encode_img'):
            raise AttributeError(
                'Qwen white-box image attack currently requires model.encode_img for differentiable image embeddings'
            )

        img_embs = []
        for items in img_prompts:
            embs = []
            for img in items:
                feats, _ = self.model.encode_img(img)
                embs.append(feats)
            img_embs.append(embs)

        return img_embs

    def generate_context_embedding(self, batch_text_embs, batch_img_embs, require_grad=True):
        assert len(batch_text_embs) == len(batch_img_embs), 'Unmathced batch size of text and image prompts'

        batch_size = len(batch_text_embs)
        batch_context_embs = []

        for i in range(batch_size):
            text_embs = batch_text_embs[i]
            img_embs = batch_img_embs[i]

            num_text_segs = len(text_embs)
            num_img_segs = len(img_embs)

            if num_text_segs == 0 and num_img_segs == 0:
                mixed_embs = [torch.zeros([1, 0, 0], device=self.device)]
            elif num_text_segs == 0:
                mixed_embs = img_embs
            elif num_img_segs == 0:
                mixed_embs = text_embs
            else:
                s = t = 0
                mixed_embs = []
                while s < num_text_segs and t < num_img_segs:
                    mixed_embs.append(text_embs[s])
                    mixed_embs.append(img_embs[t])
                    s, t = s + 1, t + 1
                if s < num_text_segs:
                    mixed_embs += text_embs[s:]
                if t < num_img_segs:
                    mixed_embs += img_embs[t:]

            mixed_embs = torch.cat(mixed_embs, dim=1)

            current_max_len = mixed_embs.shape[1] + self.max_new_tokens
            if current_max_len - self.max_length > 0:
                print(
                    'Warning: The number of tokens in current conversation exceeds the max length. '
                    'The model will not see the contexts outside the range.'
                )

            begin_idx = max(0, current_max_len - self.max_length)
            mixed_embs = mixed_embs[:, begin_idx:]
            if require_grad:
                mixed_embs.retain_grad()
            batch_context_embs.append(mixed_embs)

        return batch_context_embs
