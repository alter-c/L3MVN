import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import os
import ffmpeg
import json
import argparse

plt.ioff()
def plot_3d_sequence_with_direction(points_with_theta, max_xy_lim=None, direct_show=False):
    """
    绘制三维序列在三个平面上的投影，颜色从前往后逐渐变浅，0坐标固定在图片中间
    在x-y平面上显示每个点的航向角θ方向
    
    参数:
    points_with_theta: 包含(x, y, z, theta)坐标和航向角的列表，例如 [(x1,y1,z1,theta1), (x2,y2,z2,theta2), ...]
    theta是弧度制的航向角，逆时针方向为正
    """
    if not points_with_theta:
        print("输入点序列为空！")
        return
    
    # 将点序列转换为numpy数组
    points_array = np.array(points_with_theta)
    x = points_array[:, 0]
    y = points_array[:, 1]
    z = points_array[:, 2]
    theta = points_array[:, 3]
    
    # 创建颜色渐变（从深到浅）
    n_points = len(points_with_theta)
    colors = plt.cm.Blues(np.linspace(0.7, 0.2, n_points))  # 蓝色渐变，从深到浅
    
    # 创建图形和子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    
    # 计算坐标轴范围，确保0点在中心
    def get_axis_range(data, margin_ratio=0.1):
        """计算坐标轴范围，确保0点在中心"""
        if max_xy_lim is None:
            max_abs = max(np.abs(data.min()), np.abs(data.max()))
            if max_abs == 0:  # 处理所有点都是0的情况
                max_abs = 1
            range_val = max_abs * (1 + margin_ratio)
            return [-range_val, range_val]
        else:
            return [-max_xy_lim, max_xy_lim]
    
    # 获取各坐标轴的范围
    x_range = get_axis_range(x)
    y_range = get_axis_range(y)
    z_range = get_axis_range(z)
    
    # 绘制x-y平面
    scatter1 = ax1.scatter(y, x, c=colors, s=50, alpha=0.8, edgecolors='black', linewidth=0.5)
    
    # 添加方向箭头
    arrow_length = 0.1 * min(x_range[1] - x_range[0], y_range[1] - y_range[0])  # 箭头长度自适应
    for i in range(n_points):
        dx = arrow_length * np.cos(theta[i])
        dy = - arrow_length * np.sin(theta[i])
        ax1.arrow(y[i], x[i], dy, dx, 
                 head_width=arrow_length*0.1, head_length=arrow_length*0.2,
                 fc=colors[i], ec='black', alpha=0.8, linewidth=0.5)
    
    ax1.set_xlabel('Y', fontsize=12)
    ax1.set_ylabel('X', fontsize=12)
    ax1.set_title('X-Y (θ)', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # 设置x-y平面的坐标轴范围，0点在中心
    ax1.set_xlim(y_range)
    ax1.set_ylim(x_range)
    
    # 添加坐标轴线和0点标记
    ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3, linewidth=0.8)
    ax1.axvline(x=0, color='k', linestyle='-', alpha=0.3, linewidth=0.8)
    ax1.plot(0, 0, 'k+', markersize=10, markeredgewidth=2, label='O')
    
    # 添加起始和结束点标记
    ax1.plot(y[0], x[0], 'bo', markersize=8, label='step 0')
    ax1.plot(y[-1], x[-1], 'ro', markersize=8, label=f'step {n_points}')
    ax1.legend()
    
    # 绘制y-z平面
    scatter2 = ax2.scatter(y, z, c=colors, s=50, alpha=0.8, edgecolors='black', linewidth=0.5)
    ax2.set_xlabel('Y', fontsize=12)
    ax2.set_ylabel('Z', fontsize=12)
    ax2.set_title('Y-Z', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    # 设置y-z平面的坐标轴范围，0点在中心
    ax2.set_xlim(y_range)
    ax2.set_ylim(z_range)
    
    # 添加坐标轴线和0点标记
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3, linewidth=0.8)
    ax2.axvline(x=0, color='k', linestyle='-', alpha=0.3, linewidth=0.8)
    ax2.plot(0, 0, 'k+', markersize=10, markeredgewidth=2, label='O')
    
    # 添加起始和结束点标记
    ax2.plot(y[0], z[0], 'bo', markersize=8, label='step 0')
    ax2.plot(y[-1], z[-1], 'ro', markersize=8, label=f'step {n_points}')
    ax2.legend()
    
    # 添加颜色条
    # cbar = plt.colorbar(scatter1, ax=[ax1, ax2], orientation='horizontal', 
    #                    pad=0.05, shrink=0.8)
    # cbar.set_label('时间顺序 (从深到浅)', fontsize=10)
    
    if direct_show:
        plt.tight_layout()
        plt.show()
    
    return fig


def combine_images_with_bottom_text(top_image_path, bottom_fig, text_list, output_path=None, 
                                  text_color=(0, 0, 0), text_size=20, direct_show=False):
    """
    将顶部图片与底部matplotlib图形拼接成一个新图片，并在底部添加白色留白和文字
    
    参数:
    top_image_path: 顶部图片的文件路径
    bottom_fig: 底部matplotlib图形对象
    text_list: 要添加的文本内容,每一条为一行
    output_path: 输出图片的保存路径，如果为None则不保存
    text_color: 文本颜色，默认为黑色(0, 0, 0)
    text_size: 文本大小，默认为20
    padding: 文本与边缘的间距，默认为20
    """
    # 将matplotlib图形转换为PIL图像
    buf = io.BytesIO()
    bottom_fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    bottom_img = Image.open(buf)
    
    # 打开顶部图片
    top_img = Image.open(top_image_path)
    
    # 调整顶部图片宽度与底部图形一致
    bottom_width, bottom_height = bottom_img.size
    top_img = top_img.resize((bottom_width, int(bottom_width * top_img.height / top_img.width)))
    
    # 计算最终图片尺寸（增加底部留白）
    top_width, top_height = top_img.size
    combined_width = bottom_width
    padding = text_size * 4
    bottom_margin = len(text_list) * text_size + 2 * padding  # 根据文本大小和间距计算底部留白高度
    combined_height = top_height + bottom_height + bottom_margin  # 增加底部留白
    
    # 创建新图片（白色背景）
    combined_image = Image.new('RGB', (combined_width, combined_height), (255, 255, 255))
    
    # 粘贴顶部图片
    combined_image.paste(top_img, (0, 0))
    
    # 粘贴底部图形
    combined_image.paste(bottom_img, (0, top_height))
    
    # 添加文本到底部留白区域
    draw = ImageDraw.Draw(combined_image)
    
    # 尝试加载字体，如果失败则使用默认字体
    try:
        font = ImageFont.truetype("arial.ttf", text_size)
    except IOError:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", text_size)
        except IOError:
            font = ImageFont.load_default()
    
    # 计算文本位置（在底部留白区域居中）
    text = "\n".join(text_list)
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    # 文本位置：水平居中，垂直方向在底部留白区域内居中
    # text_x = (combined_width - text_width) // 2
    # text_y = top_height + bottom_height + (bottom_margin - text_height) // 2
    text_x = 10
    text_y = top_height + bottom_height + (bottom_margin - text_height) // 2
    
    # 绘制文本
    draw.text((text_x, text_y), text, fill=text_color, font=font)
    
    if direct_show:
        # 显示结果
        plt.figure(figsize=(12, 12 * combined_height / combined_width))
        plt.imshow(combined_image)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    # 保存图片（如果指定了输出路径）
    if output_path:
        combined_image.save(output_path, dpi=(300, 300))
        print(f"拼接后的图片已保存到: {output_path}")
    
    return combined_image


def draw_anchors(step_sample, max_xy_lim=None, epsode_root_path=None):
    cur_flame_path = step_sample['visual_input']['front'][-1]
    if epsode_root_path is not None:
        cur_flame_path = os.path.join(epsode_root_path, cur_flame_path)

    label_anchors = [(x['x'], x['y'], x['z'], x['theta']) for x in step_sample['label']['trajectory']]
    fig = plot_3d_sequence_with_direction(label_anchors, max_xy_lim=max_xy_lim)

    informations = []
    informations.append(f"episode_id={step_sample['episode_id']}")
    informations.append(f"instruction_en={step_sample['text_input'].get('en', 'none')}")
    # informations.append(f"instruction_zh={step_sample['text_input'].get('zh', 'none')}")
    informations.append(f"Step {step_sample['step']}")
    for li in range(7, -1, -1):
        if li < len(step_sample['label']['trajectory']):
            x = step_sample['label']['trajectory'][li]
            informations.append(f"Label {li} : x={x['x']:.4f}, y = {x['y']:.4f}, z = {x['z']:.4f}, theta = {x['theta']:.4f}")
        else:
            informations.append(f"Label {li} : null")

    com_image = combine_images_with_bottom_text(cur_flame_path, fig, informations, text_size=50)

    return com_image


def save_gif(series_imgs, output_path="output.gif", duration=500, loop=0):
    if not series_imgs:
        return

    # 将第一帧作为GIF的基础，然后添加剩余帧
    first_img = series_imgs[0]
    other_imgs = series_imgs[1:]

    first_img.save(
        output_path,
        format='GIF',
        append_images=other_imgs,
        save_all=True,
        duration=duration,
        loop=loop
    )

    print(f"GIF已保存到: {output_path}")

def save_mp4(series_imgs, output_path="output.mp4", fps=2):
    first_img = series_imgs[0]
    process = (
        ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='bgr24', s=f'{first_img.size[0]}x{first_img.size[1]}', framerate=fps)
        .output(output_path, vcodec='libx264', pix_fmt='yuv420p', r=fps)
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )

    for img in series_imgs:
        process.stdin.write(img.tobytes())

    process.stdin.close()
    process.wait()

    print(f"MP4已保存到: {output_path}")


def calculate_percentile_simple(data, percentile=0.9):
    """
    简化版的90分位值计算（不使用插值）
    """
    if not data:
        raise ValueError("列表不能为空")
    
    sorted_data = sorted(data)
    n = len(sorted_data)
    
    # 直接计算位置索引
    index = int(percentile * n)
    
    # 确保索引不越界
    if index >= n:
        index = n - 1
    
    return sorted_data[index]

def draw_one_episode(episode_data, output_dir, epsode_root_path=None, width=1024, height=1024, fps=2, is_autoscale=False):
    output_frames_dir = os.path.join(output_dir, "frames")
    os.makedirs(output_frames_dir, exist_ok=True)
    output_video = os.path.join(output_dir, "output.mp4")
    output_git = os.path.join(output_dir, "output.gif")

    for step_sample in episode_data:
       for x in step_sample['label']['trajectory']:
           x['z'] = 0

    max_xy_lim = None
    if not is_autoscale:
        all_abs_vals = []
        for step_sample in episode_data:
            for x in step_sample['label']['trajectory']:
                all_abs_vals.append(abs(x['x']))
                all_abs_vals.append(abs(x['y']))
                all_abs_vals.append(abs(x['z']))

        max_xy_lim = calculate_percentile_simple(all_abs_vals, percentile=0.97)

    series_imgs = []
    for step_sample in episode_data:
        img = draw_anchors(step_sample, max_xy_lim=max_xy_lim, epsode_root_path=epsode_root_path)
        img = img.resize((width, height))
        series_imgs.append(img)
        img.save(os.path.join(output_frames_dir, f"step_{step_sample['step']:03d}.jpg"), dpi=(300, 300))

    save_gif(series_imgs, output_path=output_git, duration=(1.0 / fps) * 1000, loop=0)

    """
    如果要存mp4视频，请确保安装了ffmpeg
    # apt install ffmpeg libx264-dev libx265-dev
    # pip install ffmpeg-python
    """
    # save_mp4(series_imgs, output_path=output_video, fps=fps)

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', "--task_id", type=str, required=True,
                        help='Episode id for selecting the json file.')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    episode_root_path = "/home/yongao/repo/L3MVN/datasets"

    json_path = f"{episode_root_path}/objectnav/{args.task_id}.json"
    episode_data = load_json(json_path)
    print(f"Loaded {len(episode_data)} step from {json_path}")

    output_dir = f"{episode_root_path}/result"
    draw_one_episode(episode_data, output_dir, epsode_root_path=episode_root_path)

