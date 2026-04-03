import torch
from torch import nn
import math


# ============ LoRA 网络结构 ============
class LoRA(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.rank = rank  # LoRA的秩（rank），控制低秩矩阵的大小
        self.A = nn.Linear(in_features, rank, bias=False)  # 低秩矩阵A
        self.B = nn.Linear(rank, out_features, bias=False)  # 低秩矩阵B
        # 矩阵A高斯初始化
        self.A.weight.data.normal_(mean=0.0, std=0.02)
        # 矩阵B全0初始化
        self.B.weight.data.zero_()

    def forward(self, x):
        return self.B(self.A(x))


# ============ DoRA 网络结构 ============
class DoRA(nn.Module):
    """
    DoRA: Weight-Decomposed Low-Rank Adaptation
    将权重分解为 magnitude 和 direction 两部分
    W = m * (W0 + B·A) / ||W0 + B·A||_c
    """
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.rank = rank
        self.in_features = in_features
        self.out_features = out_features

        # LoRA 部分（用于方向更新）
        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=False)

        # 矩阵A高斯初始化
        self.A.weight.data.normal_(mean=0.0, std=0.02)
        # 矩阵B全0初始化
        self.B.weight.data.zero_()

        # Magnitude 向量（可训练参数）
        # 初始化为1，表示初始时保持原始权重的幅度
        self.magnitude = nn.Parameter(torch.ones(out_features))

    def forward(self, x, weight):
        """
        Args:
            x: 输入 [batch, seq, in_features]
            weight: 原始权重 W0 [out_features, in_features]

        Returns:
            output: DoRA 输出
        """
        # 1. 计算 LoRA 更新：B·A·x
        lora_output = self.B(self.A(x))

        # 2. 计算完整的方向向量：V = W0 + B·A
        # B.weight: [out_features, rank], A.weight: [rank, in_features]
        lora_weight = self.B.weight @ self.A.weight  # [out_features, in_features]
        combined_weight = weight + lora_weight

        # 3. 计算列范数（column-wise L2 norm）
        # 对每一列（输出维度）计算范数
        column_norm = torch.norm(combined_weight, p=2, dim=1, keepdim=True).t()  # [1, out_features]
        column_norm = column_norm.clamp(min=1e-8)  # 避免除零

        # 4. 归一化方向
        normalized_weight = combined_weight / column_norm.t()  # [out_features, in_features]

        # 5. 应用 magnitude
        dora_weight = self.magnitude.unsqueeze(1) * normalized_weight  # [out_features, in_features]

        # 6. 计算最终输出
        output = torch.nn.functional.linear(x, dora_weight)

        return output


# ============ QLoRA 相关实现 ============
# NF4 量化表（针对正态分布优化的16个值）
NF4_QUANT_TABLE = torch.tensor([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0
], dtype=torch.float32)


class NF4Quantizer:
    """
    NF4 (4-bit NormalFloat) 量化器
    实现与 bitsandbytes 相同的分块量化策略
    """

    @staticmethod
    def quantize_blockwise(tensor, blocksize=64, double_quant=True):
        """
        分块量化：将张量分成多个block，每个block独立量化

        Args:
            tensor: 输入张量 [out_features, in_features]
            blocksize: 每个block的大小（默认64）
            double_quant: 是否对scale进行二次量化（节省显存）

        Returns:
            quantized: uint8 张量，每个元素存储0-15的索引
            absmax: 每个block的缩放因子
            absmax_quantized: 如果double_quant=True，返回量化后的absmax
        """
        # 展平张量
        original_shape = tensor.shape
        tensor_flat = tensor.flatten()
        n_elements = tensor_flat.numel()

        # 计算需要多少个block
        n_blocks = math.ceil(n_elements / blocksize)

        # Padding到blocksize的整数倍
        padded_size = n_blocks * blocksize
        if n_elements < padded_size:
            tensor_flat = torch.cat([tensor_flat, torch.zeros(padded_size - n_elements, device=tensor.device, dtype=tensor.dtype)])

        # 重塑为 [n_blocks, blocksize]
        tensor_blocks = tensor_flat.reshape(n_blocks, blocksize)

        # 每个block计算absmax（缩放因子）
        absmax = tensor_blocks.abs().max(dim=1, keepdim=True)[0]
        absmax = absmax.clamp(min=1e-8)  # 避免除零

        # 归一化到 [-1, 1]
        normalized = tensor_blocks / absmax

        # 量化：找到最近的NF4值
        nf4_table = NF4_QUANT_TABLE.to(tensor.device)
        distances = (normalized.unsqueeze(-1) - nf4_table).abs()
        quantized = distances.argmin(dim=-1).to(torch.uint8)

        # 去除padding
        quantized = quantized.flatten()[:n_elements].reshape(original_shape)
        absmax = absmax.squeeze().reshape(-1)

        # 双重量化：对absmax也进行量化（FP32 -> FP8）
        absmax_quantized = None
        if double_quant:
            absmax_quantized, absmax_scale = NF4Quantizer.quantize_fp8(absmax)
            return quantized, absmax_quantized, absmax_scale

        return quantized, absmax, None

    @staticmethod
    def quantize_fp8(tensor):
        """
        将FP32的scale量化为FP8（双重量化）
        简化实现：使用线性量化到int8
        """
        scale = tensor.abs().max()
        scale = scale.clamp(min=1e-8)
        quantized = (tensor / scale * 127).round().clamp(-128, 127).to(torch.int8)
        return quantized, scale

    @staticmethod
    def dequantize_blockwise(quantized, absmax, absmax_scale=None, blocksize=64, original_shape=None):
        """
        分块反量化

        Args:
            quantized: uint8 量化值
            absmax: 缩放因子（可能是量化的）
            absmax_scale: 如果absmax是量化的，这是它的scale
            blocksize: block大小
            original_shape: 原始形状
        """
        # 如果absmax是量化的，先反量化
        if absmax_scale is not None:
            absmax = NF4Quantizer.dequantize_fp8(absmax, absmax_scale)

        # 展平
        quantized_flat = quantized.flatten()
        n_elements = quantized_flat.numel()
        n_blocks = math.ceil(n_elements / blocksize)

        # Padding
        padded_size = n_blocks * blocksize
        if n_elements < padded_size:
            quantized_flat = torch.cat([quantized_flat, torch.zeros(padded_size - n_elements, device=quantized.device, dtype=torch.uint8)])

        # 重塑为 [n_blocks, blocksize]
        quantized_blocks = quantized_flat.reshape(n_blocks, blocksize)

        # 查表反量化
        nf4_table = NF4_QUANT_TABLE.to(quantized.device)
        dequantized_blocks = nf4_table[quantized_blocks.long()]

        # 乘以scale
        absmax = absmax.reshape(-1, 1).to(dequantized_blocks.device)
        dequantized_blocks = dequantized_blocks * absmax

        # 去除padding并恢复形状
        dequantized = dequantized_blocks.flatten()[:n_elements]
        if original_shape is not None:
            dequantized = dequantized.reshape(original_shape)

        return dequantized

    @staticmethod
    def dequantize_fp8(quantized, scale):
        """反量化FP8"""
        return quantized.to(torch.float32) / 127.0 * scale


class QuantizedLinear(nn.Module):
    """
    4-bit NF4 量化的 Linear 层
    实现与 bitsandbytes Linear4bit 相同的功能
    """

    def __init__(self, in_features, out_features, bias=True, blocksize=64, double_quant=True, compute_dtype=torch.bfloat16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.blocksize = blocksize
        self.double_quant = double_quant
        self.compute_dtype = compute_dtype

        # 注册buffer存储量化后的权重
        self.register_buffer('weight_quantized', None)
        self.register_buffer('weight_absmax', None)
        self.register_buffer('weight_absmax_scale', None)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

    def quantize_weight(self, weight):
        """量化权重"""
        if self.double_quant:
            quantized, absmax_q, absmax_scale = NF4Quantizer.quantize_blockwise(
                weight, blocksize=self.blocksize, double_quant=True
            )
            self.weight_quantized = quantized
            self.weight_absmax = absmax_q
            self.weight_absmax_scale = absmax_scale
        else:
            quantized, absmax, _ = NF4Quantizer.quantize_blockwise(
                weight, blocksize=self.blocksize, double_quant=False
            )
            self.weight_quantized = quantized
            self.weight_absmax = absmax
            self.weight_absmax_scale = None

    def forward(self, x):
        # 反量化权重
        weight_dequantized = NF4Quantizer.dequantize_blockwise(
            self.weight_quantized,
            self.weight_absmax,
            self.weight_absmax_scale,
            blocksize=self.blocksize,
            original_shape=(self.out_features, self.in_features)
        )

        # 使用指定精度计算
        output = torch.nn.functional.linear(
            x.to(self.compute_dtype),
            weight_dequantized.to(self.compute_dtype),
            self.bias.to(self.compute_dtype) if self.bias is not None else None
        )

        return output


# ============ LoRA 应用函数 ============
def apply_lora(model, rank=16):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.weight.shape[0] == module.weight.shape[1]:
            lora = LoRA(module.weight.shape[0], module.weight.shape[1], rank=rank).to(model.device)
            setattr(module, "lora", lora)
            original_forward = module.forward

            # 显式绑定
            def forward_with_lora(x, layer1=original_forward, layer2=lora):
                return layer1(x) + layer2(x)

            module.forward = forward_with_lora


# ============ QLoRA 应用函数 ============
def apply_qlora(model, rank=16, blocksize=64, double_quant=True, compute_dtype=torch.bfloat16):
    """
    应用 QLoRA 到模型

    Args:
        model: 基础模型
        rank: LoRA 秩
        blocksize: 量化block大小
        double_quant: 是否双重量化
        compute_dtype: 计算精度
    """
    quantized_count = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and module.weight.shape[0] == module.weight.shape[1]:
            # 1. 创建量化层
            quant_layer = QuantizedLinear(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                blocksize=blocksize,
                double_quant=double_quant,
                compute_dtype=compute_dtype
            ).to(module.weight.device)

            # 2. 量化原始权重
            quant_layer.quantize_weight(module.weight.data)
            if module.bias is not None:
                quant_layer.bias.data = module.bias.data

            # 3. 替换模块
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            if parent_name:
                parent = model.get_submodule(parent_name)
            else:
                parent = model
            setattr(parent, child_name, quant_layer)

            # 4. 添加 LoRA 适配器
            lora = LoRA(quant_layer.in_features, quant_layer.out_features, rank=rank).to(model.device)
            setattr(quant_layer, "lora", lora)

            # 5. 重写 forward
            original_forward = quant_layer.forward
            def forward_with_lora(x, layer1=original_forward, layer2=lora):
                return layer1(x) + layer2(x)
            quant_layer.forward = forward_with_lora

            quantized_count += 1


# ============ DoRA 应用函数 ============
def apply_dora(model, rank=16):
    """
    应用 DoRA 到模型

    Args:
        model: 基础模型
        rank: DoRA 秩
    """
    dora_count = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.weight.shape[0] == module.weight.shape[1]:
            # 创建 DoRA 适配器
            device = module.weight.device
            dora = DoRA(module.in_features, module.out_features, rank=rank).to(device)

            # 初始化 magnitude 为原始权重的列范数
            with torch.no_grad():
                column_norm = torch.norm(module.weight.data, p=2, dim=1)
                dora.magnitude.data = column_norm

            setattr(module, "dora", dora)

            # 保存原始权重（冻结）
            module.weight.requires_grad = False
            original_forward = module.forward
            original_weight = module.weight

            # 重写 forward
            def forward_with_dora(x, dora_layer=dora, weight=original_weight, original_fn=original_forward):
                # DoRA 输出
                dora_output = dora_layer(x, weight)
                # 如果有 bias，加上
                if hasattr(original_fn.__self__, 'bias') and original_fn.__self__.bias is not None:
                    dora_output = dora_output + original_fn.__self__.bias
                return dora_output

            module.forward = forward_with_dora
            dora_count += 1

    print(f"[DoRA] 成功应用 DoRA 到 {dora_count} 个 Linear 层")


def load_lora(model, path):
    state_dict = torch.load(path, map_location=model.device)
    state_dict = {(k[7:] if k.startswith('module.') else k): v for k, v in state_dict.items()}

    for name, module in model.named_modules():
        # 支持 LoRA
        if hasattr(module, 'lora'):
            lora_state = {k.replace(f'{name}.lora.', ''): v for k, v in state_dict.items() if f'{name}.lora.' in k}
            module.lora.load_state_dict(lora_state)
        # 支持 DoRA
        if hasattr(module, 'dora'):
            dora_state = {k.replace(f'{name}.dora.', ''): v for k, v in state_dict.items() if f'{name}.dora.' in k}
            module.dora.load_state_dict(dora_state)


def save_lora(model, path):
    raw_model = getattr(model, '_orig_mod', model)
    state_dict = {}
    for name, module in raw_model.named_modules():
        # 保存 LoRA 权重
        if hasattr(module, 'lora'):
            clean_name = name[7:] if name.startswith("module.") else name
            lora_state = {f'{clean_name}.lora.{k}': v.cpu().half() for k, v in module.lora.state_dict().items()}
            state_dict.update(lora_state)
        # 保存 DoRA 权重
        if hasattr(module, 'dora'):
            clean_name = name[7:] if name.startswith("module.") else name
            dora_state = {f'{clean_name}.dora.{k}': v.cpu().half() for k, v in module.dora.state_dict().items()}
            state_dict.update(dora_state)
    torch.save(state_dict, path)


def merge_lora(model, lora_path, save_path):
    load_lora(model, lora_path)
    raw_model = getattr(model, '_orig_mod', model)
    state_dict = {k: v.cpu().half() for k, v in raw_model.state_dict().items() if '.lora.' not in k}
    for name, module in raw_model.named_modules():
        if isinstance(module, nn.Linear) and '.lora.' not in name:
            state_dict[f'{name}.weight'] = module.weight.data.clone().cpu().half()
            if hasattr(module, 'lora'):
                state_dict[f'{name}.weight'] += (module.lora.B.weight.data @ module.lora.A.weight.data).cpu().half()
    torch.save(state_dict, save_path)
