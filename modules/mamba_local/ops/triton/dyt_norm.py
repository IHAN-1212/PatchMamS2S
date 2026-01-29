import torch
import triton
import triton.language as tl


# 使用triton.jit装饰器定义一个内核函数
@triton.jit
def tanh_transform_kernel(
        x_ptr, output_ptr,
        num_elements,
        alpha, weight, bias,
        BLOCK_SIZE: tl.constexpr,
):
    # 获取当前程序正在处理的元素索引
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements

    # 加载数据
    x = tl.load(x_ptr + offsets, mask=mask)

    # 应用变换: tanh(alpha * x) * weight + bias
    # 注意：在Triton中，weight和bias可以是标量或向量
    output = tl.math.tanh(alpha * x) * weight + bias

    # 存储结果
    tl.store(output_ptr + offsets, output, mask=mask)


# 包装函数，用于调用Triton内核
def tanh_transform(x, alpha=0.5, weight=1.0, bias=0.0):
    # 确定输入大小
    num_elements = x.numel()

    # 创建输出张量
    output = torch.empty_like(x)

    # 启动内核
    grid = lambda meta: (triton.cdiv(num_elements, meta['BLOCK_SIZE']),)
    tanh_transform_kernel[grid](
        x, output,
        num_elements,
        alpha, weight, bias,
        BLOCK_SIZE=1024
    )

    return output