import matplotlib.pyplot as plt
import numpy as np

class SensorVisualizer:
    """
    传感器数据可视化器 - 保持单窗口更新
    """
    def __init__(self, figsize=(12, 8)):
        self.fig = None
        self.axes = []
        self.imgs = []
        self.first_run = True
        self.figsize = figsize
        
    def _create_layout(self, n):
        """根据图像数量创建布局"""
        if n == 1:
            rows, cols = 1, 1
        elif n == 2:
            rows, cols = 1, 2
        elif n <= 4:
            rows, cols = 2, 2
        elif n <= 6:
            rows, cols = 2, 3
        else:
            rows, cols = 3, 3
        return rows, cols
    
    def visualize(self, sensor_data):
        """
        可视化sensor_data中的RGB图像 - 在固定窗口中更新
        sensor_data: dict, key为名称, value为(480,640,3) uint8数组
        """
        # 验证和收集有效图像
        valid_images = []
        for key, img_array_dict in sensor_data.items():
            img_array = img_array_dict["color"]
            if not isinstance(img_array, np.ndarray):
                print(f"✗ {key}: 不是numpy数组")
                continue
            if img_array.shape != (480, 640, 3):
                print(f"✗ {key}: 形状错误 {img_array.shape}, 应为 (480, 640, 3)")
                continue
            if img_array.dtype != np.uint8:
                print(f"✗ {key}: 数据类型错误 {img_array.dtype}, 应为 uint8")
                continue
            
            valid_images.append((key, img_array))
            # print(f"✓ {key}: 加载成功")
        
        if not valid_images:
            print("没有有效的图像数据")
            return
        
        n = len(valid_images)
        rows, cols = self._create_layout(n)
        
        # 第一次运行：创建图形和子图
        if self.first_run or len(self.axes) != n:
            if self.fig is not None:
                plt.close(self.fig)
            
            self.fig, self.axes = plt.subplots(rows, cols, figsize=self.figsize)
            if n == 1:
                self.axes = [self.axes]
            else:
                self.axes = self.axes.flatten() if n > 1 else [self.axes]
            
            # 初始化图像对象
            self.imgs = []
            for idx, (name, img_array) in enumerate(valid_images):
                ax = self.axes[idx]
                img = ax.imshow(img_array)
                ax.set_title(name, fontsize=12, color='white', pad=10)
                ax.axis('off')
                self.imgs.append(img)
            
            # 隐藏多余的子图
            for idx in range(n, len(self.axes)):
                self.axes[idx].axis('off')
            
            # 设置深色背景
            self.fig.patch.set_facecolor('#1a1a1a')
            for ax in self.axes:
                ax.set_facecolor('#1a1a1a')
            
            plt.tight_layout()
            plt.ion()  # 开启交互模式
            self.first_run = False
            
        else:
            # 更新现有图像数据
            for idx, (name, img_array) in enumerate(valid_images):
                self.imgs[idx].set_array(img_array)
                self.axes[idx].set_title(name, fontsize=12, color='white', pad=10)
        
        # 刷新显示
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)  # 小暂停确保更新
        
        return self.fig
    
    def close(self):
        """关闭窗口"""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.axes = []
            self.imgs = []
            self.first_run = True

def visualize_sensor_data(sensor_data):
    visualizer = SensorVisualizer(figsize=(12, 6))
    return visualizer.visualize(sensor_data)

# 模拟测试数据
def create_test_data():
    """创建测试用的模拟传感器数据"""
    data = {}
    # 创建左摄像头图像 - 红色渐变
    left = np.zeros((480, 640, 3), dtype=np.uint8)
    left[:, :, 0] = np.tile(np.linspace(0, 255, 640), (480, 1)).astype(np.uint8)
    data['left_camera'] = left
    
    # 创建右摄像头图像 - 蓝色渐变
    right = np.zeros((480, 640, 3), dtype=np.uint8)
    right[:, :, 2] = np.tile(np.linspace(0, 255, 640), (480, 1)).astype(np.uint8)
    data['right_camera'] = right
    
    return data

if __name__ == "__main__":
    # 测试：模拟多次调用，窗口保持不动
    print("=== 第一次调用 ===")
    test_data = create_test_data()
    visualize_sensor_data(test_data)

    print("\n=== 模拟2秒后更新数据 ===")
    import time
    time.sleep(2)

    # 修改数据模拟新帧
    test_data['left_camera'] = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    test_data['right_camera'] = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    visualize_sensor_data(test_data)

    print("\n窗口保持打开，数据已更新。调用 visualizer.close() 关闭窗口")
