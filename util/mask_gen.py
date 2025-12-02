from PIL import Image
import numpy as np

def process_image(input_path, output_path, target_number):
    """
    将图像处理为正方形，分割为3x3网格，对指定区块替换为类似二维码的块状噪声，调整透明度后保存
    
    参数:
        input_path: 输入图像路径
        output_path: 输出图像路径
        target_number: 要替换为噪声的区块编号(1-9)
    """
    # 验证目标数字是否有效
    if not (1 <= target_number <= 9):
        raise ValueError("目标数字必须在1-9之间")
    
    # 打开图像并转换为RGBA模式以支持透明度
    with Image.open(input_path).convert("RGBA") as img:
        # 获取原图尺寸
        width, height = img.size
        
        # 计算正方形尺寸（取较小边）
        square_size = min(width, height)
        
        # 计算裁剪区域（居中裁剪）
        left = (width - square_size) // 2
        top = (height - square_size) // 2
        right = left + square_size
        bottom = top + square_size
        
        # 裁剪为正方形
        square_img = img.crop((left, top, right, bottom))
        
        # 计算每个小区块的尺寸
        block_size = square_size // 3
        
        # 将图像转换为numpy数组以便处理
        img_array = np.array(square_img)
        
        # 计算目标区块的位置
        row = (target_number - 1) // 3
        col = (target_number - 1) % 3
        
        # 计算目标区块的边界
        start_row = row * block_size
        end_row = start_row + block_size
        start_col = col * block_size
        end_col = start_col + block_size
        
        # 生成类似二维码的块状噪声
        block_height = end_row - start_row
        block_width = end_col - start_col
        
        # 定义子块大小（控制块状效果的粗细）
        sub_block_size = max(4, min(block_height, block_width) // 30)  # 动态调整子块大小
        
        # 创建子块网格
        rows = block_height // sub_block_size
        cols = block_width // sub_block_size
        
        # 生成随机黑白子块分布
        sub_blocks = np.random.choice([0, 255], size=(rows, cols), p=[0.7, 0.3])
        
        # 放大子块以填充整个区域，形成块状噪声
        noise = np.repeat(np.repeat(sub_blocks, sub_block_size, axis=0), sub_block_size, axis=1)
        
        # 处理可能的尺寸不匹配
        if noise.shape[0] > block_height:
            noise = noise[:block_height, :]
        if noise.shape[1] > block_width:
            noise = noise[:, :block_width]
            
        # 扩展为RGB三通道
        noise_rgb = np.stack([noise, noise, noise], axis=-1).astype(np.uint8)
        
        # 完全替换目标区块
        img_array[start_row:end_row, start_col:end_col, :3] = noise_rgb
        
        # 调整整体透明度为50%
        img_array[:, :, 3] = 128  # Alpha通道值范围0-255，128表示50%透明度
        
        # 将numpy数组转回PIL图像并保存
        result_img = Image.fromarray(img_array)
        result_img.save(output_path, "PNG")

# 使用示例
if __name__ == "__main__":
    try:
        # 处理图像，将第5块（中心块）变为高斯噪声
        process_image("/home/data/dsy/data/SSROriDataset/NTIRE2020/NTIRE2020_Validation_Clean/ARAD_HS_0451_clean.png", "/home/data/dsy/data/Fusion/output2.png", 2)
        print("图像处理完成，已保存至output.png")
    except Exception as e:
        print(f"处理过程中出错: {e}")
