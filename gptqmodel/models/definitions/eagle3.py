from .base import BaseGPTQModel
import torch.nn as nn

class Eagle3MidLayer(nn.Module):
    """EAGLE3的单层结构包装器"""
    def __init__(self, midlayer_module):
        super().__init__()
        self.midlayer = midlayer_module
        
    @property
    def self_attn(self):
        return self.midlayer.self_attn
        
    @property 
    def mlp(self):
        return self.midlayer.mlp
        
    @property
    def input_layernorm(self):
        return self.midlayer.input_layernorm
        
    @property
    def post_attention_layernorm(self):
        return self.midlayer.post_attention_layernorm
        
    @property
    def hidden_norm(self):
        return self.midlayer.hidden_norm

class Eagle3GPTQ(BaseGPTQModel):
    """
    EAGLE3-LLaMA3.1特殊架构的GPTQ模型定义
    这是一个单层模型，结构不同于标准Llama
    """
    
    # 根级别模块
    non_layer_modules = [
        "d2t",          # 可能是token到embedding的映射
        "t2d",          # 可能是embedding到token的映射  
        "fc",           # 全连接层
        "norm",         # 最终层归一化
        "lm_head",      # 语言模型头
    ]
    
    # 只有一个层节点
    layers_node = "midlayer"
    
    # 自定义层类型
    layer_type = Eagle3MidLayer
    
    # 层内模块执行顺序
    # 根据权重文件中的模块顺序
    layer_modules = [
        # 注意力机制的QKV投影
        ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
        # 注意力输出投影
        ["self_attn.o_proj"],
        # MLP的门控和上投影
        ["mlp.gate_proj", "mlp.up_proj"],
        # MLP的下投影
        ["mlp.down_proj"],
        # 各种归一化层
        ["hidden_norm", "input_layernorm", "post_attention_layernorm"],
    ]
    
    # 模型配置（从权重文件推断）
    @classmethod
    def get_model_config(cls, config):
        """获取模型配置"""
        model_config = {
            "hidden_size": 4096,
            "intermediate_size": 14336,
            "num_hidden_layers": 1,  # EAGLE3是单层
            "num_attention_heads": 32,  # 需要根据实际计算
            "num_key_value_heads": 8,   # 需要根据实际计算
            "vocab_size": 128256,       # 从t2d的形状推断
            "max_position_embeddings": 4096,  # 默认值
        }
        
        # 如果配置文件中有，则使用配置文件的值
        if config:
            for key in model_config:
                if hasattr(config, key):
                    model_config[key] = getattr(config, key)
                    
        return model_config
    
    # 重写层获取方法
    def get_layers(self):
        """获取所有层 - EAGLE3只有单层"""
        if hasattr(self.model, 'midlayer'):
            # 包装单层
            return [Eagle3MidLayer(self.model.midlayer)]
        return []
    
    def get_layer(self, prefix: str):
        """获取特定层"""
        layers = self.get_layers()
        if layers:
            return layers[0]
        return None
    
    # 需要特殊处理的权重映射
    @staticmethod
    def map_weights(weights):
        """将权重映射到标准格式"""
        mapped = {}
        
        # 直接映射的权重
        direct_mapping = {
            "d2t": "d2t",
            "t2d": "t2d", 
            "norm.weight": "norm.weight",
            "fc.weight": "fc.weight",
            "lm_head.weight": "lm_head.weight",
        }
        
        # midlayer的映射
        midlayer_mapping = {
            "midlayer.self_attn.q_proj.weight": "model.midlayer.self_attn.q_proj.weight",
            "midlayer.self_attn.k_proj.weight": "model.midlayer.self_attn.k_proj.weight",
            "midlayer.self_attn.v_proj.weight": "model.midlayer.self_attn.v_proj.weight",
            "midlayer.self_attn.o_proj.weight": "model.midlayer.self_attn.o_proj.weight",
            "midlayer.mlp.gate_proj.weight": "model.midlayer.mlp.gate_proj.weight",
            "midlayer.mlp.up_proj.weight": "model.midlayer.mlp.up_proj.weight",
            "midlayer.mlp.down_proj.weight": "model.midlayer.mlp.down_proj.weight",
            "midlayer.hidden_norm.weight": "model.midlayer.hidden_norm.weight",
            "midlayer.input_layernorm.weight": "model.midlayer.input_layernorm.weight",
            "midlayer.post_attention_layernorm.weight": "model.midlayer.post_attention_layernorm.weight",
        }
        
        # 应用映射
        for src_key, dst_key in {**direct_mapping, **midlayer_mapping}.items():
            if src_key in weights:
                mapped[dst_key] = weights[src_key]
                
        return mapped
    
    # 量化配置
    @classmethod
    def get_quantize_config(cls, config):
        """获取量化配置"""
        from ..quantize import QuantizeConfig
        
        return QuantizeConfig(
            bits=4,
            group_size=128,
            desc_act=False,
            sym=False,
        )