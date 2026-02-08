# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025
# SPDX-License-Identifier: Apache-2.0

from ..base import BaseQModel


class Eagle3QModel(BaseQModel):
    """
    EAGLE3-LLaMA3.1 的 GPTQ 模型定义
    基于 vLLM 的实现进行适配，支持 EAGLE3 的特殊架构
    """
    
    # 语言模型头的名称
    lm_head = "lm_head"
    
    # 语言模型头前的归一化模块
    pre_lm_head_norm_module = "model.norm"
    
    # AWQ 缩放优化需要形状匹配的模块
    awq_scale_optimize_shape_dependent_modules = ["self_attn.o_proj"]
    
    # 模块树定义
    # EAGLE3 的特殊结构：
    # - 第一层：使用 embeddings + hidden_states 的拼接输入
    # - 后续层：使用 hidden_states 输入
    # - 包含特殊的 fc 层用于组合辅助隐藏状态
    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "hidden_norm": ("hidden_norm:!",),
            "self_attn": ("qkv_proj:0", "o_proj:1"),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp": ("gate_proj:0", "up_proj:0", "down_proj:1"),
        }
    ]
    
    # 支持的语言模型类型
    supported_lm_types = ["llama", "eagle3"]
    
    # 需要信任远程代码
    require_trust_remote_code = True
    
    # 支持的 VRAM 策略
    supported_vram_strategies = [
        VramStrategy.EXCLUSIVE,
        VramStrategy.BALANCED,
        VramStrategy.OPTIMIZED
    ]
    
    # 模态支持
    modality = [MODALITY.TEXT]
    
    def __init__(self, model, quantized, quantize_config, tokenizer=None, **kwargs):
        super().__init__(model, quantized, quantize_config, tokenizer, **kwargs)
        
        # EAGLE3 特定的初始化
        self._setup_eagle3_specifics()
    
    def _setup_eagle3_specifics(self):
        """设置 EAGLE3 特定的属性和配置"""
        config = getattr(self.model, 'config', None)
        
        # 设置 EAGLE3 特定的配置
        self.use_aux_hidden_state = True
        
        # 从配置中获取 EAGLE3 特定设置
        if config:
            eagle_config = getattr(config, 'eagle_config', None)
            if eagle_config and isinstance(eagle_config, dict):
                self.use_aux_hidden_state = eagle_config.get('use_aux_hidden_state', True)
            
            # 设置草稿词汇表大小
            if not hasattr(config, 'draft_vocab_size'):
                config.draft_vocab_size = getattr(config, 'vocab_size', 128256)
        
        # 检查是否包含必要的 EAGLE3 组件
        self._validate_eagle3_structure()
    
    def _validate_eagle3_structure(self):
        """验证 EAGLE3 结构完整性"""
        required_components = [
            'model.embed_tokens',
            'model.layers',
            'model.norm',
            'lm_head',
        ]
        
        for component in required_components:
            if not self._has_component(component):
                log.warning(f"EAGLE3 model missing component: {component}")
    
    def _has_component(self, component_path):
        """检查组件是否存在"""
        try:
            parts = component_path.split('.')
            obj = self.model
            for part in parts:
                obj = getattr(obj, part)
            return True
        except AttributeError:
            return False
    
    def combine_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        组合辅助隐藏状态
        这是 EAGLE3 特有的功能，用于处理多个辅助隐藏状态
        """
        if not self.use_aux_hidden_state:
            return hidden_states
        
        # 检查是否有 fc 层用于组合隐藏状态
        if hasattr(self.model.model, 'fc'):
            return self.model.model.fc(hidden_states)
        elif hasattr(self.model, 'fc'):
            return self.model.fc(hidden_states)
        
        log.warning("EAGLE3 fc layer not found for combining hidden states")
        return hidden_states
    
    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        计算 logits，处理草稿到目标 token 的映射
        EAGLE3 特有的草稿 token 映射功能
        """
        logits = self.lm_head(hidden_states)
        
        # 如果有草稿到目标的映射
        if hasattr(self, 'draft_id_to_target_id'):
            config = getattr(self.model, 'config', None)
            if config:
                draft_vocab_size = getattr(config, 'draft_vocab_size', 128256)
                vocab_size = getattr(config, 'vocab_size', 128256)
                
                base = torch.arange(draft_vocab_size, device=logits.device)
                targets = base + self.draft_id_to_target_id
                
                logits_new = logits.new_full(
                    (logits.shape[0], vocab_size),
                    float("-inf"),
                )
                logits_new[:, targets] = logits
                return logits_new
        
        return logits
    
    def forward_special(self, input_ids, positions, hidden_states, input_embeds=None):
        """
        EAGLE3 特殊的前向传播
        模仿 vLLM 中 EAGLE3 的前向传播逻辑
        """
        if input_embeds is None:
            input_embeds = self.model.embed_input_ids(input_ids)
        
        # 验证形状
        assert hidden_states.shape[-1] == input_embeds.shape[-1]
        
        residual = None
        for i, layer in enumerate(self.model.model.layers):
            # 第一层的特殊处理
            if i == 0:
                embeds = layer.input_layernorm(input_embeds)
                hidden_states = layer.hidden_norm(hidden_states)
                hidden_states = torch.cat([embeds, hidden_states], dim=-1)
            else:
                hidden_states = layer.input_layernorm(hidden_states)
            
            # 注意力机制
            hidden_states = layer.self_attn(
                positions=positions,
                hidden_states=hidden_states,
            )
            
            hidden_states = layer.post_attention_layernorm(hidden_states)
            
            # MLP
            hidden_states = layer.mlp(hidden_states)
        
        # 最终归一化
        hidden_states = self.model.model.norm(hidden_states)
        
        return hidden_states
    
    @classmethod
    def get_model_config(cls, config):
        """获取 EAGLE3 模型配置"""
        model_config = {
            "hidden_size": 4096,
            "intermediate_size": 14336,
            "num_hidden_layers": 1,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "vocab_size": 128256,
            "max_position_embeddings": 4096,
            "rms_norm_eps": 1e-5,
            "draft_vocab_size": 128256,
        }
        
        if config:
            # 从 HF 配置获取
            for key in model_config:
                if hasattr(config, key):
                    model_config[key] = getattr(config, key)
            
            # 处理 EAGLE3 特定配置
            if hasattr(config, 'eagle_config'):
                eagle_config = config.eagle_config
                if isinstance(eagle_config, dict):
                    for key, value in eagle_config.items():
                        model_config[f"eagle_{key}"] = value
        
        return model_config
    
    @classmethod
    def simple_layer_modules(cls, model_config, quantize_config, is_awq_quantize=False, include_capture_only=False):
        """重写以支持 EAGLE3 的特殊模块结构"""
        layer_modules = super().simple_layer_modules(
            model_config, quantize_config, is_awq_quantize, include_capture_only
        )
        
        # EAGLE3 特有的调整
        # 确保包含所有必要的模块
        return layer_modules
    
    def before_model_load(self, load_quantized_model):
        """在模型加载前执行的操作"""
        # EAGLE3 特定的预处理
        log.info("Initializing EAGLE3 model structure...")
    
    def after_model_load(self, model, load_quantized_model):
        """在模型加载后执行的操作"""
        # 确保 EAGLE3 特定的属性被正确设置
        model = super().after_model_load(model, load_quantized_model)
        
        # 添加 EAGLE3 特定的后处理
        if not load_quantized_model:
            self._setup_eagle3_components(model)
        
        return model
    
    def _setup_eagle3_components(self, model):
        """设置 EAGLE3 特定的组件"""
        # 确保有 draft_id_to_target_id 映射
        if not hasattr(model, 'draft_id_to_target_id'):
            config = getattr(model, 'config', None)
            if config:
                draft_vocab_size = getattr(config, 'draft_vocab_size', 128256)
                model.draft_id_to_target_id = torch.zeros(
                    draft_vocab_size, dtype=torch.long
                )
    
    @classmethod
    def map_weights(cls, weights):
        """权重映射函数"""
        mapped = {}
        
        # EAGLE3 特定的权重映射规则
        mapping_rules = {
            # 嵌入层
            "embed_tokens.weight": "model.embed_tokens.weight",
            
            # 层权重 - 注意 EAGLE3 使用 qkv_proj 而不是分开的 q_proj, k_proj, v_proj
            r"layers\.(\d+)\.self_attn\.q_proj\.weight": r"model.layers.\1.self_attn.qkv_proj.weight_q",
            r"layers\.(\d+)\.self_attn\.k_proj\.weight": r"model.layers.\1.self_attn.qkv_proj.weight_k",
            r"layers\.(\d+)\.self_attn\.v_proj\.weight": r"model.layers.\1.self_attn.qkv_proj.weight_v",
            r"layers\.(\d+)\.self_attn\.o_proj\.weight": r"model.layers.\1.self_attn.o_proj.weight",
            
            # MLP
            r"layers\.(\d+)\.mlp\.gate_proj\.weight": r"model.layers.\1.mlp.gate_proj.weight",
            r"layers\.(\d+)\.mlp\.up_proj\.weight": r"model.layers.\1.mlp.up_proj.weight",
            r"layers\.(\d+)\.mlp\.down_proj\.weight": r"model.layers.\1.mlp.down_proj.weight",
            
            # 归一化层
            r"layers\.(\d+)\.input_layernorm\.weight": r"model.layers.\1.input_layernorm.weight",
            r"layers\.(\d+)\.post_attention_layernorm\.weight": r"model.layers.\1.post_attention_layernorm.weight",
            r"layers\.(\d+)\.hidden_norm\.weight": r"model.layers.\1.hidden_norm.weight",
            
            # 特殊权重
            "fc.weight": "model.fc.weight",
            "norm.weight": "model.norm.weight",
            "lm_head.weight": "lm_head.weight",
            "draft_id_to_target_id": "draft_id_to_target_id",
        }
        
        import re
        for src_key, weight in weights.items():
            for pattern, replacement in mapping_rules.items():
                if re.match(pattern, src_key):
                    dst_key = re.sub(pattern, replacement, src_key)
                    mapped[dst_key] = weight
                    break
            else:
                # 如果没有匹配的规则，直接使用
                mapped[src_key] = weight
        
        return mapped
    
    def __str__(self):
        """字符串表示"""
        config = getattr(self.model, 'config', {})
        return f"Eagle3QModel({config.model_type if hasattr(config, 'model_type') else 'eagle3'})"