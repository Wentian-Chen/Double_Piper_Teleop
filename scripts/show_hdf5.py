

import sys
import argparse
from pathlib import Path

try:
    import h5py
    import numpy as np
except ImportError:
    print("错误：需要安装 h5py 和 numpy")
    print("请运行: pip install h5py numpy")
    sys.exit(1)


class H5Viewer:
    def __init__(self, filepath, max_preview=10, show_data=False):
        self.filepath = Path(filepath)
        self.max_preview = max_preview
        self.show_data = show_data
        self.indent = "  "
        
    def view(self):
        """主入口：查看整个H5文件"""
        if not self.filepath.exists():
            print(f"错误：文件不存在 {self.filepath}")
            return
            
        print(f"\\n{'='*60}")
        print(f"📁 文件: {self.filepath}")
        print(f"{'='*60}\\n")
        
        try:
            with h5py.File(self.filepath, 'r') as f:
                # 显示文件级别属性
                self._print_attributes(f, "文件属性")
                
                # 递归遍历所有组和数据集
                print(f"\\n📂 结构层级:")
                self._visit_group(f, 0)
                
                # 显示所有数据集摘要
                print(f"\\n{'='*60}")
                print("📊 数据集摘要")
                print(f"{'='*60}")
                self._summarize_datasets(f)
                
        except Exception as e:
            print(f"读取文件时出错: {e}")
    
    def _visit_group(self, obj, level):
        """递归遍历组"""
        items = list(obj.items())
        
        for name, item in items:
            prefix = self.indent * level
            
            if isinstance(item, h5py.Group):
                # 这是一个组
                print(f"{prefix}📁 {name}/")
                self._print_attributes(item, f"{prefix}   [属性]", inline=True)
                self._visit_group(item, level + 1)
                
            elif isinstance(item, h5py.Dataset):
                # 这是一个数据集
                shape_str = str(item.shape) if item.shape else "标量"
                dtype_str = str(item.dtype)
                size_str = self._format_size(item.nbytes)
                
                print(f"{prefix}📄 {name}")
                print(f"{prefix}   形状: {shape_str} | 类型: {dtype_str} | 大小: {size_str}")
                
                # 显示属性
                self._print_attributes(item, f"{prefix}   [属性]", inline=True)
                
                # 显示数据预览
                if self.show_data:
                    self._preview_data(item, prefix + "   ")
    
    def _print_attributes(self, obj, label, inline=False):
        """打印对象的属性"""
        attrs = dict(obj.attrs)
        if not attrs:
            return
            
        if inline:
            attr_str = ", ".join([f"{k}={v}" for k, v in attrs.items()])
            print(f"{label}: {attr_str}")
        else:
            print(f"\\n{label}:")
            for key, value in attrs.items():
                print(f"  {key}: {value}")
    
    def _preview_data(self, dataset, prefix):
        """预览数据集内容"""
        try:
            if dataset.shape == ():
                # 标量
                print(f"{prefix}值: {dataset[()]}")
            else:
                # 数组 - 显示前几个元素
                data = dataset[()]
                
                if dataset.ndim == 1:
                    preview = data[:self.max_preview]
                    if len(data) > self.max_preview:
                        print(f"{prefix}预览 [{len(data)}]: {preview} ...")
                    else:
                        print(f"{prefix}数据: {preview}")
                        
                elif dataset.ndim == 2:
                    rows, cols = dataset.shape
                    preview_rows = min(self.max_preview, rows)
                    preview_cols = min(self.max_preview, cols)
                    preview = data[:preview_rows, :preview_cols]
                    print(f"{prefix}预览 [{rows}x{cols}]:")
                    for row in preview:
                        print(f"{prefix}  {row}")
                    if rows > preview_rows or cols > preview_cols:
                        print(f"{prefix}  ...")
                        
                else:
                    # 高维数组
                    print(f"{prefix}预览 [{dataset.shape}]:")
                    print(f"{prefix}  {data.flat[:self.max_preview]} ...")
                    
        except Exception as e:
            print(f"{prefix}[无法预览: {e}]")
    
    def _summarize_datasets(self, f):
        """汇总所有数据集信息"""
        datasets = []
        total_size = 0
        
        def collect_datasets(name, obj):
            if isinstance(obj, h5py.Dataset):
                datasets.append((name, obj))
                nonlocal total_size
                total_size += obj.nbytes
        
        f.visititems(collect_datasets)
        
        print(f"总计: {len(datasets)} 个数据集, {self._format_size(total_size)}")
        print()
        
        for name, ds in datasets:
            shape_str = str(ds.shape) if ds.shape else "标量"
            print(f"  {name:40s} {shape_str:20s} {str(ds.dtype):15s}")
    
    def _format_size(self, nbytes):
        """格式化字节大小"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if nbytes < 1024.0:
                return f"{nbytes:.2f} {unit}"
            nbytes /= 1024.0
        return f"{nbytes:.2f} PB"


def extract_dataset(filepath, dataset_path, output=None):
    """提取特定数据集到numpy数组或文件"""
    try:
        with h5py.File(filepath, 'r') as f:
            if dataset_path not in f:
                print(f"错误：数据集 '{dataset_path}' 不存在")
                print("可用数据集:")
                f.visititems(lambda n, o: print(f"  {n}") if isinstance(o, h5py.Dataset) else None)
                return
            
            data = f[dataset_path][()]
            print(f"已提取: {dataset_path}")
            print(f"形状: {data.shape}, 类型: {data.dtype}")
            
            if output:
                np.save(output, data)
                print(f"已保存到: {output}")
            else:
                print(f"数据预览: {data}")
                
    except Exception as e:
        print(f"提取失败: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='HDF5/H5文件查看器',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python h5viewer.py model.h5                    # 查看文件结构
  python h5viewer.py data.h5 -d                 # 查看结构并显示数据预览
  python h5viewer.py model.h5 -n 5              # 预览前5个元素
  python h5viewer.py data.h5 -e /group/dataset  # 提取特定数据集
        """
    )
    
    parser.add_argument('file', help='HDF5文件路径')
    parser.add_argument('-d', '--data', action='store_true', 
                       help='显示数据集内容预览')
    parser.add_argument('-n', '--num', type=int, default=10,
                       help='预览元素数量 (默认: 10)')
    parser.add_argument('-e', '--extract', metavar='PATH',
                       help='提取特定数据集路径 (如: /weights/layer1)')
    parser.add_argument('-o', '--output',
                       help='提取数据的输出文件 (.npy格式)')
    
    args = parser.parse_args()
    
    if args.extract:
        extract_dataset(args.file, args.extract, args.output)
    else:
        viewer = H5Viewer(args.file, max_preview=args.num, show_data=args.data)
        viewer.view()


if __name__ == '__main__':
    main()
