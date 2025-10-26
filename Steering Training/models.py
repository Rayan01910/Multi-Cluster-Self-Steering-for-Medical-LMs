import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import cfg

class CausalLMWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name, torch_dtype=torch.float16 if cfg.fp16 else None, device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # cache some shapes
        self.hidden_size = self.model.config.hidden_size
        # layers access
        self.num_hidden_layers = self.model.config.num_hidden_layers

    @torch.no_grad()
    def option_hidden(self, prompt: str, option_text: str) -> torch.Tensor:
        """
        Returns the last token hidden state from the penultimate layer for (prompt + option_text).
        """
        tok = self.tokenizer(prompt + option_text, return_tensors="pt", truncation=True, max_length=cfg.max_length).to(cfg.device)
        outputs = self.model.transformer(**{k: tok[k] for k in ["input_ids", "attention_mask"]}, output_hidden_states=True)
        h_all_layers = outputs.hidden_states  # tuple: layer0..last
        target_layer = -cfg.penultimate_layer_offset
        # last token index
        last_idx = tok["attention_mask"].sum(dim=1) - 1
        h = h_all_layers[target_layer][torch.arange(h_all_layers[target_layer].shape[0]), last_idx]  # (1, hidden)
        return h.squeeze(0)  # (hidden,)

    @torch.no_grad()
    def option_logprob(self, prompt: str, option_text: str) -> float:
        """
        Sum logprobs of option tokens conditioned on the whole prompt+option (causal LM).
        """
        tok = self.tokenizer(prompt, return_tensors="pt").to(cfg.device)
        opt = self.tokenizer(option_text, return_tensors="pt", add_special_tokens=False).to(cfg.device)
        # Concatenate
        input_ids = torch.cat([tok["input_ids"], opt["input_ids"]], dim=1)
        attn = torch.cat([tok["attention_mask"], opt["attention_mask"]], dim=1)
        with torch.no_grad():
            out = self.model(input_ids=input_ids, attention_mask=attn)
        # logprobs of option tokens given previous
        logits = out.logits[:, :-1, :]
        labels = input_ids[:, 1:]
        # mask to only option positions
        option_len = opt["input_ids"].shape[1]
        mask = torch.zeros_like(labels, dtype=torch.bool)
        mask[:, -option_len:] = True

        logprobs = torch.log_softmax(logits, dim=-1)
        token_lp = logprobs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        lp = token_lp[mask].sum().item()
        return lp
