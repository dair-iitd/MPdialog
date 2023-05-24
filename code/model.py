import torch
import os
from typing import Dict, List, Optional, Tuple, Any
from types import SimpleNamespace as SN
from transformers import (
    GPT2LMHeadModel,
    CLIPVisionModel,
    CLIPVisionConfig,
    EvalPrediction,
)
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions, ModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import WEIGHTS_NAME
from transformers.models.gpt2 import GPT2Config
from params import GenerationHparams

from transformers import GPT2LMHeadModel, EvalPrediction

import torch
import logging

logger = logging.getLogger(__name__)


class LinearPooler(torch.nn.Module):
    def __init__(self, in_features, out_features, n_tokens) -> None:
        super().__init__()
        self.W = torch.nn.Linear(in_features, n_tokens * out_features)
        self.in_features = in_features
        self.out_features = out_features
        self.n_tokens = n_tokens

    def forward(self, x):
        wx = self.W(x)
        wx = wx.reshape(-1, self.out_features).requires_grad_()
        return wx


class MultiHeadAttentionPooler(torch.nn.Module):
    def __init__(self, query_size, embed_dim, **attn_kwargs):
        super().__init__()
        self.query_size = query_size
        self.embed_dim = embed_dim
        self.query = torch.nn.parameter.Parameter(torch.empty((query_size, embed_dim)))
        self.attn = torch.nn.MultiheadAttention(
            embed_dim, batch_first=True, **attn_kwargs
        )
        self._reset_parameters()

    def _reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.query)

    def forward(self, sequence_output: torch.FloatTensor) -> torch.FloatTensor:
        query = self.query.expand(
            sequence_output.shape[0], self.query_size, self.embed_dim
        )
        attn_output, _ = self.attn(query, sequence_output, sequence_output)
        return attn_output.reshape(-1, self.embed_dim).requires_grad_()


class FrozenConfig(PretrainedConfig):
    def __init__(self, n_visual_tokens=1, mapper_type=None, **kwargs):
        super().__init__(**kwargs)
        self.n_visual_tokens = n_visual_tokens
        self.mapper_type = mapper_type


class Frozen(PreTrainedModel):
    def __init__(
        self,
        lm_config: GPT2Config,
        visual_config: CLIPVisionConfig,
        config: FrozenConfig,
        lm_cls=GPT2LMHeadModel,
        visual_cls=CLIPVisionModel,
    ) -> None:

        super().__init__(config=config)
        self.config = config
        self.lm_config = lm_config
        self.visual_config = visual_config

        self.lm_cls = lm_cls
        self.visual_cls = visual_cls
        self.visual_mapper = None
        self._init_models()

    def _init_models(self):
        self.lm = self.lm_cls(self.lm_config)
        self.visual = self.visual_cls(self.visual_config)
        self.embed_layer_norm = torch.nn.LayerNorm(self.lm_config.n_embd)

        if self.config.n_visual_tokens > 1:
            d1 = self.visual_config.hidden_size
            d2 = self.lm_config.n_embd
            if self.config.mapper_type == "linear":
                self.visual_mapper = LinearPooler(
                    in_features=d1,
                    out_features=d2,
                    n_tokens=self.config.n_visual_tokens,
                )
            elif self.config.mapper_type == "attn":
                self.visual_mapper = MultiHeadAttentionPooler(
                    query_size=self.config.n_visual_tokens,
                    embed_dim=d2,
                    kdim=d1,
                    vdim=d1,
                    num_heads=8,
                )
            else:
                raise NotImplementedError(
                    f"Mapper type {self.mapper_type} not implemented"
                )

    @staticmethod
    def from_pretrained(
        dir_path=None,
        lm_path=None,
        visual_path=None,
        mapping_model_dir=None,
        config: FrozenConfig = None,
        lm_cls=GPT2LMHeadModel,
        visual_cls=CLIPVisionModel,
    ):
        lm_path, visual_path, mapping_model_dir = Frozen._make_paths(
            dir_path, lm_path, visual_path, mapping_model_dir
        )

        lm_config = GPT2Config.from_pretrained(lm_path)
        visual_config = CLIPVisionConfig.from_pretrained(visual_path)

        if not config:
            if not dir_path:
                raise ValueError("dir_path must be provided if config is not provided")
            config = FrozenConfig.from_pretrained(dir_path)

        model = Frozen(
            lm_config=lm_config,
            visual_config=visual_config,
            config=config,
            lm_cls=lm_cls,
            visual_cls=visual_cls,
        )
        if dir_path and os.path.exists(os.path.join(dir_path, WEIGHTS_NAME)):
            model.load_state_dict(torch.load(os.path.join(dir_path, WEIGHTS_NAME)))
            return model

        model.lm = lm_cls.from_pretrained(lm_path)
        model.visual = visual_cls.from_pretrained(visual_path)
        if mapping_model_dir and model.visual_mapper:
            try:
                model.visual_mapper.load_state_dict(
                    torch.load(os.path.join(mapping_model_dir, WEIGHTS_NAME))
                )
            except:
                logger.warn("Could not load mapping model weights")
        return model

    @staticmethod
    def _make_paths(dir_path, lm_path, visual_path, mapping_model_dir):
        if not dir_path:
            return lm_path, visual_path, mapping_model_dir
        lm_path = os.path.join(dir_path, "lm") if lm_path is None else lm_path
        visual_path = (
            os.path.join(dir_path, "visual") if visual_path is None else visual_path
        )
        mapping_model_dir = (
            os.path.join(dir_path, "visual_mapper")
            if mapping_model_dir is None
            else mapping_model_dir
        )

        return lm_path, visual_path, mapping_model_dir

    def save_pretrained(
        self,
        dir_path,
        lm_path=None,
        visual_path=None,
        mapping_model_dir=None,
        state_dict=None,
    ):
        lm_path, visual_path, mapping_model_dir = Frozen._make_paths(
            dir_path, lm_path, visual_path, mapping_model_dir
        )

        os.makedirs(lm_path, exist_ok=True)
        os.makedirs(visual_path, exist_ok=True)

        self.lm.save_pretrained(lm_path)
        self.visual.save_pretrained(visual_path)

        if self.visual_mapper is not None:
            os.makedirs(mapping_model_dir, exist_ok=True)
            torch.save(
                self.visual_mapper, os.path.join(mapping_model_dir, WEIGHTS_NAME)
            )
        self.config.save_pretrained(dir_path)
        state_dict = self.state_dict()
        torch.save(state_dict, os.path.join(dir_path, WEIGHTS_NAME))
        print("Weights saved to: ", os.path.join(dir_path, WEIGHTS_NAME))

    def freeze_lm(self):
        for _, param in self.lm.transformer.named_parameters():
            param.requires_grad = False
        print("Frozen LM!")
        return self

    def unfreeze_lm(self):
        for _, param in self.lm.transformer.named_parameters():
            param.requires_grad = True
        print("Unfrozen LM!")
        return self

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        images: Optional[torch.FloatTensor] = None,
        image_mask: Optional[
            torch.BoolTensor
        ] = None,  # a tensor of same shape as input_ids indicating where image embeddings should be placed
        attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        gen_hparams: Optional[GenerationHparams] = None,
        **kwargs,
    ):
        if gen_hparams is not None:
            gen_hparams = gen_hparams.__dict__

            outputs = self.generate(
                input_ids=input_ids,
                images=images,
                image_mask=image_mask,
                attention_mask=attention_mask,
                labels=labels,
                **gen_hparams,
            )
            dummy_loss = torch.Tensor([0.0])
            dummy_loss.requires_grad = False

            return {
                "loss": dummy_loss,
                "logits": outputs,
            }

        assert image_mask.shape == input_ids.shape
        assert (
            self.config.n_visual_tokens * images.shape[0] == image_mask.sum().item()
        ), f"Expected {self.config.n_visual_tokens * images.shape[0]} image tokens, got {image_mask.sum().item()}"
        return_dict = return_dict = (
            return_dict if return_dict is not None else self.lm.config.use_return_dict
        )

        token_embeds = self._get_embeds(
            image_mask=image_mask, input_ids=input_ids, images=images
        )
        transformer_outputs = self.lm.transformer(
            input_ids=None,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            # token_type_ids=image_mask.long()
            # if image_mask is not None
            # else token_type_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=token_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = transformer_outputs[0]
        lm_logits = self.lm.lm_head(hidden_states)
        lm_logits = lm_logits.contiguous()
        loss = None

        if labels is not None:
            # make sure of this in dataloader:  labels[image_mask] = -100
            # Shift so that tokens < n predict n
            loss = Frozen.compute_loss(lm_logits, labels)

        if not return_dict:
            output = (lm_logits,)
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            # past_key_values=transformer_outputs.past_key_values,
            # hidden_states=transformer_outputs.hidden_states,
            # attentions=transformer_outputs.attentions,
            # cross_attentions=transformer_outputs.cross_attentions,
        )

    @staticmethod
    def compute_loss(logits, labels):
        m = logits.shape[-2]
        n = labels.shape[-1]
        min_ = min(m, n - 1)
        shift_logits = logits[..., :min_, :].contiguous()
        shift_labels = labels[..., 1 : min_ + 1].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        return loss

    @staticmethod
    def compute_metrics(preds: EvalPrediction):
        # import pdb

        # pdb.set_trace()
        loss = Frozen.compute_loss(
            torch.Tensor(preds.predictions).float(),
            torch.Tensor(preds.label_ids).long(),
        )
        ppl = torch.exp(loss)
        return {"eval_ppl": ppl.item(), "loss": loss.item()}

    def _expand_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        attention_mask: Optional[torch.LongTensor] = None,
        encoder_outputs: Optional[ModelOutput] = None,
        images: Optional[torch.FloatTensor] = None,
        image_mask: Optional[torch.BoolTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **model_kwargs,
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        expanded_return_idx = (
            torch.arange(input_ids.shape[0])
            .view(-1, 1)
            .repeat(1, expand_size)
            .view(-1)
            .to(input_ids.device)
        )

        input_ids = input_ids.index_select(0, expanded_return_idx)

        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = token_type_ids.index_select(
                0, expanded_return_idx
            )

        if images is not None:
            expanded_return_image_idx = (
                torch.arange(images.shape[0])
                .view(-1, 1)
                .repeat(1, expand_size)
                .view(-1)
                .to(images.device)
            )
            model_kwargs["images"] = images.index_select(0, expanded_return_image_idx)

        if labels is not None:
            model_kwargs["labels"] = labels.index_select(0, expanded_return_idx)
        if image_mask is not None:
            model_kwargs["image_mask"] = image_mask.index_select(0, expanded_return_idx)
        if attention_mask is not None:
            model_kwargs["attention_mask"] = attention_mask.index_select(
                0, expanded_return_idx
            )

        if is_encoder_decoder:
            if encoder_outputs is None:
                raise ValueError(
                    "If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined."
                )
            encoder_outputs[
                "last_hidden_state"
            ] = encoder_outputs.last_hidden_state.index_select(
                0, expanded_return_idx.to(encoder_outputs.last_hidden_state.device)
            )
            model_kwargs["encoder_outputs"] = encoder_outputs
        return input_ids, model_kwargs

    def _get_embeds(self, image_mask, input_ids, images):
        token_embeds = self.lm.transformer.wte(input_ids)
        if image_mask.sum() == 0:
            return token_embeds
        clip_outputs = self.visual(pixel_values=images)

        if self.config.n_visual_tokens > 1:
            if self.config.mapper_type == "linear":
                image_embeds = clip_outputs.pooler_output
            elif self.config.mapper_type == "attn":
                image_embeds = clip_outputs.last_hidden_state
            image_embeds = self.visual_mapper(image_embeds)
        else:
            image_embeds = clip_outputs.pooler_output

        image_embeds = self.embed_layer_norm(image_embeds)
        token_embeds[image_mask] = image_embeds
        # token_embeds = self.embed_layer_norm(token_embeds)
        return token_embeds

    def prepare_inputs_for_generation(
        self, input_ids: torch.LongTensor, **kwargs
    ) -> Dict[str, Any]:

        return {**kwargs, "input_ids": input_ids}

    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
    ) -> Dict[str, Any]:
        if "past_key_values" in outputs:
            model_kwargs["past"] = outputs.past_key_values
        elif "mems" in outputs:
            model_kwargs["past"] = outputs.mems
        elif "past_buckets_states" in outputs:
            model_kwargs["past"] = outputs.past_buckets_states
        else:
            model_kwargs["past"] = None

        if "image_mask" in model_kwargs:
            image_mask = model_kwargs["image_mask"]
            model_kwargs["image_mask"] = torch.cat(
                [image_mask, image_mask.new_zeros((image_mask.shape[0], 1))], dim=-1
            )

        if "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            model_kwargs["attention_mask"] = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))],
                dim=-1,
            )
        return model_kwargs


class GPTLMHeadForGeneration(GPT2LMHeadModel):
    def forward(
        self,
        gen_hparams=None,
        input_ids=None,
        attention_mask=None,
        labels=None,
        **kwargs,
    ):
        if gen_hparams is not None:

            outputs = self.generate(
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
                pad_token_id=self.config.eos_token_id,
                **gen_hparams.__dict__,
            )
            dummy_loss = torch.Tensor([0.0])
            dummy_loss.requires_grad = False
            return {
                "loss": dummy_loss,
                "logits": outputs,
            }
        return super().forward(input_ids=input_ids, labels=labels, **kwargs)
