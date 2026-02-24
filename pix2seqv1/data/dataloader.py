from typing import Dict, List, Iterator, Any
import torch
from torch.utils.data import DataLoader


class InterleavedDataLoader:
    """
    多任务混合加载器 (Interleaved DataLoader)
    
    功能: 
    管理多个 PyTorch DataLoader, 根据指定的权重交替采样.
    
    工作流程：
    1.  初始化时接收多个 DataLoader (例如 {"det": loader_A, "seg": loader_B}) 和对应的权重
    2.  每次调用 __next__, 先根据权重随机选择 Task
    3.  从该任务的 DataLoader 中提取 Batch
    4.  如果该 DataLoader 数据耗尽 (StopIteration), 会自动重置 (Re-initialize), 实现无限循环
    """

    def __init__(self, loaders: Dict[str, DataLoader], task_configs: List[Any]):
        """
        Args:
            loaders: 字典, Key为任务名, Value为对应的DataLoader
            task_configs: 配置列表，包含 'name' 和 'weight'
        """
        self.loaders = loaders
        self.task_names = list(loaders.keys())
        
        # 提取并整理任务权重
        raw_weights = []
        for name in self.task_names:
            # 按任务名匹配配置, 找到对应的权重
            cfg = next((t for t in task_configs if t.name == name), None)
            if cfg:
                raw_weights.append(cfg.weight)
            else:
                raw_weights.append(1.0) # 默认权重

        # 转换为 Tensor 方便使用 multinomial 采样
        self.weights = torch.tensor(raw_weights, dtype=torch.float32)
        
        # 为每个 DataLoader 创建迭代器
        self.iterators: Dict[str, Iterator] = {
            k: iter(v) for k, v in loaders.items()
        }
        
        # 计算 Epoch 长度 (Epoch Length)
        # 策略：取所有 Loader 长度之和，这代表模型在一个 Epoch 里见过的样本总量
        self._len = sum(len(l) for l in loaders.values())

    def __iter__(self):
        self._count = 0
        return self

    def __next__(self):
        # 是否达到 Epoch 边界
        if self._count >= self._len:
            raise StopIteration

        # 1. 根据权重选择一个任务索引
        task_idx = torch.multinomial(self.weights, 1).item()
        task_name = self.task_names[task_idx]
        
        # 2. 获取该任务的迭代器
        task_iter = self.iterators[task_name]
        
        try:
            # 3. 尝试获取 Batch
            batch = next(task_iter)
        except StopIteration:
            # 4. 迭代器耗尽时自动重置
            task_iter = iter(self.loaders[task_name])
            self.iterators[task_name] = task_iter
            batch = next(task_iter)
            
        return batch

    def __len__(self):
        return self._len