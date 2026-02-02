import torch
from torch import nn
import torch.nn.functional as F
from modules.lora import LoRALayer


class Mlp(nn.Module):
    """
    多层感知机(MLP)模块
    功能：实现一个两层的全连接神经网络，包含激活函数和dropout。   主要用于特征转换和非线性变换
    
    参数说明：
    - in_features: 输入特征维度
    - hidden_features: 隐藏层特征维度，默认为输入维度
    - out_features: 输出特征维度，默认为输入维度
    - act_layer: 激活函数类型，默认为GELU
    - drop: dropout比率，默认为0
    """
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
        lora_rank=32,
        lora_alpha=1,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc1_lora = LoRALayer(in_features, hidden_features, rank=lora_rank, lora_alpha=lora_alpha)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.fc2_lora = LoRALayer(hidden_features, out_features, rank=lora_rank, lora_alpha=lora_alpha)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x) + self.fc1_lora(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x) + self.fc2_lora(x)
        x = self.drop(x)
        return x


class Block_softmoe(nn.Module):
    """
    软混合专家(Soft MoE)注意力块
    功能：实现多模态注意力机制，包含音频(a)、文本(t)和视觉(v)三个模态的Transformer
    
    参数说明：
    - dim: 输入特征维度
    - num_heads: 注意力头数，默认为8
    - attn_drop: 注意力dropout比率
    - proj_drop: 投影dropout比率
    - mlp_ratio: MLP扩展比率
    """
    def __init__(
            self,
            dim,
            num_heads=8,
            attn_drop=0,
            proj_drop=0,
            mlp_ratio=1,
            qkv_bias=False,
            lora_rank=32,
            lora_alpha=1,
    ):
        super().__init__()
        mlp_hidden_dim = int(dim * mlp_ratio)
        # 为每个模态创建独立的注意力模块
        self.Transformer_a = Attention(
            dim,
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
        )
        self.Transformer_t = Attention(
            dim,
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
        )
        self.Transformer_v = Attention(
            dim,
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
        )

    def forward(self, x, cross_modality='atv', mask_modality=None, mask=None):
        """
        前向传播函数
        
        参数说明：
        - x: 输入张量，形状为[B, s, C]
        - cross_modality: 跨模态类型，可选'a'/'t'/'v'
        - mask_modality: 掩码模态
        - mask: 注意力掩码
        """
        B, s, C = x.shape
        if cross_modality == 'a':
            x_a_mlp = self.Transformer_a(x, mask_modality, mask)
            return x_a_mlp
        if cross_modality == 't':
            x_t_mlp = self.Transformer_t(x, mask_modality, mask)
            return x_t_mlp
        if cross_modality == 'v':
            x_v_mlp = self.Transformer_v(x, mask_modality, mask)
            return x_v_mlp


class Attention(nn.Module):
    """
    多头自注意力机制模块
    功能：实现标准的Transformer注意力机制，包含Q、K、V投影和MLP
    
    参数说明：
    - dim: 输入特征维度
    - num_heads: 注意力头数，默认为8
    - attn_drop: 注意力dropout比率
    - proj_drop: 投影dropout比率
    - mlp_ratio: MLP扩展比率
    """
    def __init__(
            self,
            dim,
            num_heads=8,
            attn_drop=0.0,
            proj_drop=0.0,
            mlp_ratio=1.0,
            qkv_bias=False,
            lora_rank=32,
            lora_alpha=1,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.scale = head_dim ** -0.5
        # 定义Q、K、V的线性投影层
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        
        # 冻结原始投影层参数
        for param in self.qkv.parameters():
            param.requires_grad = False
        for param in self.proj.parameters():
            param.requires_grad = False

        # 添加LoRA层
        self.qkv_lora = LoRALayer(dim, dim * 3, rank=lora_rank, lora_alpha=lora_alpha)
        self.proj_lora = LoRALayer(dim, dim, rank=lora_rank, lora_alpha=lora_alpha)

        self.attn_drop = nn.Dropout(attn_drop)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            drop=proj_drop,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
        )

    def forward(self, x, mask_modality, mask=None):
        """
        前向传播函数
        
        参数说明：
        - x: 输入张量，形状为[B, seq_len, C]
        - mask_modality: 掩码模态
        - mask: 注意力掩码
        """
        B, seq_len, C = x.shape

        # 计算Q、K、V并重塑维度
        qkv = self.qkv(x) + self.qkv_lora(x)
        qkv = qkv.reshape(B, seq_len, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        q = q * self.scale
        attn = (q.float() @ k.float().transpose(-2, -1))  # [B, heads, s, s]

        # 应用掩码（如果存在）
        if mask is not None:
            mask = mask.bool()
            mask = {'a':mask[:, :seq_len], 't':mask[:, seq_len:2*seq_len], 'v':mask[:, 2*seq_len:3*seq_len]}
            mask = mask[mask_modality]
            attn = self.attn_drop(attn.masked_fill(~mask[:, None, None, :], float("-inf")).softmax(dim=-1).type_as(x))
            attn = torch.where(torch.isnan(attn), torch.full_like(attn, 0), attn)

        # 计算注意力输出并应用MLP
        x_out = (attn @ v).transpose(1, 2).reshape(B, seq_len, C)
        x_out = x_out + self.mlp(x_out)

        # 结合原始投影和LoRA投影
        x_out = self.proj(x_out) + self.proj_lora(x_out)
        x_out = self.mlp.drop(x_out)

        return x_out


class Block(nn.Module):
    """
    多层Transformer块
    功能：实现多层Soft MoE注意力块的堆叠，支持第一阶段和第二阶段的不同处理方式
    
    参数说明：
    - dim: 输入特征维度
    - num_heads: 注意力头数
    - mlp_ratio: MLP扩展比率，默认为4.0
    - drop: dropout比率
    - attn_drop: 注意力dropout比率
    - depth: Transformer块的深度，默认为4
    """
    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.0,
            drop=0.0,
            attn_drop=0.0,
            depth=4,
            qkv_bias=False,
            lora_rank=32,
            lora_alpha=1,
    ):
        super().__init__()
        self.drop = drop

        # 创建多层Soft MoE注意力块
        self.blocks = nn.ModuleList(
            [
                Block_softmoe(dim,
                              num_heads=num_heads,
                              attn_drop=attn_drop,
                              proj_drop=drop,
                              mlp_ratio=mlp_ratio,
                              qkv_bias=qkv_bias,
                              lora_rank=lora_rank,
                              lora_alpha=lora_alpha,)
                for i in range(depth)
            ]
        )

    def forward(self, x, first_stage, mask=None, modality=None):
        """
        前向传播函数
        参数说明：
        - x: 输入张量
        - first_stage: 是否为第一阶段处理
        - mask: 注意力掩码
        - modality: 模态类型
        """
        # 只保留第一阶段单模态残差连接
        for layer_idx, block in enumerate(self.blocks):
            x = x + block(x, cross_modality=modality, mask_modality=modality, mask=mask)
        return x
        
        
        
        
        # if first_stage:
        #     # 第一阶段：顺序处理每个块
        #     for layer_idx, block in enumerate(self.blocks):
        #         x = x + block(x, cross_modality=modality, mask_modality=modality, mask=mask) #残差连接
        #     return x
        # else:
        #     # 第二阶段：并行处理三个模态
        #     x_cross_a, x_cross_t, x_cross_v = torch.clone(x), torch.clone(x), torch.clone(x)
        #     for layer_idx, block in enumerate(self.blocks):
        #         x_cross_a = x_cross_a + block(x_cross_a, cross_modality='a', mask_modality=modality, mask=mask)
        #         x_cross_t = x_cross_t + block(x_cross_t, cross_modality='t', mask_modality=modality, mask=mask)
        #         x_cross_v = x_cross_v + block(x_cross_v, cross_modality='v', mask_modality=modality, mask=mask)
        #     return torch.cat([x_cross_a, x_cross_t, x_cross_v], dim=-1)  #拼接
