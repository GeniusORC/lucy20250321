import os
# 设置环境变量解决OpenMP多重初始化问题
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# 基本配置类
class Config:
    def __init__(self):
        # 检测器配置
        self.det_config = 'projects/rtmpose/rtmdet/person/rtmdet_m_640-8xb32_coco-person.py'
        self.det_checkpoint = 'rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth'
        self.det_cat_id = 0  # 检测类别ID
        self.bbox_thr = 0.3  # 边界框阈值
        self.nms_thr = 0.4  # NMS阈值
        
        # 姿态估计配置
        self.pose_config = 'configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-x_8xb320-270e_cocktail14-384x288.py'
        self.pose_checkpoint = 'rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.pth'
        self.kpt_thr = 0.5  # 关键点阈值
        
        # 可视化配置
        self.draw_bbox = False  # 是否显示边界框
        self.draw_keypoints = True  # 是否显示关键点
        self.show_posture_analysis = True  # 是否显示姿态分析
        self.radius = 5  # 关键点半径
        self.thickness = 3  # 线条粗细
        self.alpha = 0.8  # 透明度
        self.draw_heatmap = False  # 是否显示热图
        self.show_kpt_idx = False  # 是否显示关键点索引
        self.skeleton_style = 'mmpose'  # 骨架风格
        
        # 自定义关键点配置
        self.draw_iliac_midpoint = True  # 是否显示髂骨中点
        self.draw_neck_midpoint = True  # 是否显示颈椎中点
        self.custom_keypoint_radius = 6  # 自定义关键点半径
        self.custom_keypoint_thickness = 4  # 自定义关键点连接线粗细
        
        # 性能配置
        self.fps = True  # 是否显示FPS
        self.device = 'cuda:0'  # 运行设备
        
        # 输出配置
        self.output_root = ''  # 输出目录
        self.save_predictions = False  # 是否保存预测结果

# 正常范围常量定义
NORMAL_RANGES = {
    '头前倾角': '0°～5°',
    '头侧倾角': '0°～2°',
    '头旋转角': '0°～5°',
    '肩倾斜角': '0°～2°',
    '圆肩角': '>65°',
    '背部角': '<39°',
    '腹部肥胖度': '0%～35%',
    '腰曲度': '0°～5°',
    '骨盆前倾角': '-7°～7°',
    '侧中位度': '175°～185°',
    '腿型-左腿': '177°～183°',
    '腿型-右腿': '177°～183°',
    '左膝评估角': '175°～185°',
    '右膝评估角': '175°～185°',
    '身体倾斜度': '0°～2°',
    '足八角': '-5°～11°'
}

# 步态分析的正常范围 (最小值, 最大值)
GAIT_NORMAL_RANGES = {
    '左腿抬起时间': (0.25, 0.6),    # 正常范围：250-600毫秒
    '右腿抬起时间': (0.25, 0.6),    # 正常范围：250-600毫秒
    '双支撑时间': (0.05, 0.25),     # 正常范围：50-250毫秒
    '步时': (0.8, 1.2),            # 正常范围：800-1200毫秒
    '摆动时间': (0.25, 0.55),       # 正常范围：250-550毫秒
    '支撑时间': (0.5, 0.8)         # 正常范围：500-800毫秒
}

# 关键点索引常量定义
LEFT_ANKLE_IDX = 15
RIGHT_ANKLE_IDX = 16
LEFT_KNEE_IDX = 13
RIGHT_KNEE_IDX = 14
LEFT_HIP_IDX = 11
RIGHT_HIP_IDX = 12
LEFT_SHOULDER_IDX = 5
RIGHT_SHOULDER_IDX = 6
NOSE_IDX = 0

# 姿态分析参数
NORMAL_HEAD_TILT_RANGE = (0, 2)  # 头部侧倾角度范围（度）
NORMAL_HEAD_ROTATION_RANGE = (0, 5)  # 头部旋转角度范围（度）
NORMAL_SHOULDER_TILT_RANGE = (0, 2)  # 肩部倾斜角度范围（度）
NORMAL_BACK_ANGLE_RANGE = (0, 39)  # 背部角度范围（度）
NORMAL_ABDOMINAL_OBESITY_RANGE = (0, 35)  # 腹部肥胖度范围（百分比）
NORMAL_SIDE_MEDIAN_RANGE = (175, 185)  # 侧中位角度范围（度）
NORMAL_LEG_ANGLE_RANGE = (177, 183)  # 腿型角度范围（度）
NORMAL_KNEE_ANGLE_RANGE = (175, 185)  # 膝关节角度范围（度）
NORMAL_FOOT_ANGLE_RANGE = (-5, 11)  # 足八字角度范围（度）

# 设置matplotlib支持中文显示
def setup_chinese_font():
    plt_config = {
        'font.sans-serif': ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei', 'SimSun', 'sans-serif'],
        'axes.unicode_minus': False,
        'font.family': 'sans-serif'
    }
    plt.rcParams.update(plt_config)
    
    try:
        import platform
        
        # 根据操作系统设置适当的中文字体
        if platform.system() == 'Windows':
            font_path = 'C:/Windows/Fonts/simhei.ttf'  # Windows下的黑体字体
        elif platform.system() == 'Darwin':  # macOS
            font_path = '/System/Library/Fonts/PingFang.ttc'
        else:  # Linux
            font_path = '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf'
        
        # 检查字体文件是否存在
        if os.path.exists(font_path):
            # 创建字体属性对象
            chinese_font = FontProperties(fname=font_path)
            plt.rcParams['font.sans-serif'] = [chinese_font.get_name()]
            print(f"已加载中文字体: {chinese_font.get_name()}")
            return chinese_font
    except Exception as e:
        print(f"加载系统中文字体失败: {str(e)}，使用默认字体设置")
    
    return None

# 自定义CSS样式
CUSTOM_CSS = """
<style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        padding: 0.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 0.5rem;
        padding: 0.5rem;
        margin: 0.3rem 0;
    }
    .metric-normal {
        color: #28a745;
    }
    .metric-warning {
        color: #dc3545;
    }
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 0.3rem;
    }
    .metric-value {
        margin: 0;
        font-size: 0.9rem;
    }
    .metric-title {
        margin: 0 0 0.2rem 0;
        font-size: 0.9rem;
        font-weight: bold;
    }
    .metrics-container {
        max-height: 70vh;
        overflow-y: auto;
        padding-right: 0.5rem;
    }
    .metrics-group-title {
        margin: 0.5rem 0 0.2rem 0;
        padding: 0.2rem 0.5rem;
        background-color: #e6f0ff;
        border-radius: 0.3rem;
        font-size: 0.9rem;
        font-weight: bold;
        grid-column: 1 / -1;
    }
    .metrics-summary {
        background-color: #f8f9fa;
        border-left: 3px solid #1f77b4;
        padding: 0.5rem;
        margin-bottom: 0.5rem;
        font-size: 0.85rem;
    }
    .value-badge {
        display: inline-block;
        padding: 0.1rem 0.3rem;
        border-radius: 0.2rem;
        margin-right: 0.2rem;
        font-weight: bold;
    }
    .normal-badge {
        background-color: rgba(40, 167, 69, 0.2);
    }
    .warning-badge {
        background-color: rgba(220, 53, 69, 0.2);
    }
    .gait-metrics-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .gait-metric-card {
        background-color: #f8f9fa;
        border-radius: 0.3rem;
        padding: 0.5rem;
        text-align: center;
        border-top: 3px solid #1f77b4;
        margin-bottom: 0.5rem;
    }
    .gait-metric-value {
        font-size: 1.2rem;
        font-weight: bold;
        color: #1f77b4;
        margin-top: 0.2rem;
    }
    .gait-metric-title {
        font-size: 0.8rem;
        color: #666;
        margin-bottom: 0.2rem;
    }
    .gait-summary {
        background-color: #f0f7ff;
        border-left: 3px solid #1f77b4;
        padding: 0.5rem;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
        border-radius: 0.3rem;
    }
    .symmetry-indicator {
        display: inline-block;
        width: 100%;
        height: 0.5rem;
        background: linear-gradient(to right, #ff7f0e, #1f77b4);
        border-radius: 1rem;
        margin: 0.3rem 0;
        position: relative;
    }
    .symmetry-marker {
        position: absolute;
        width: 0.6rem;
        height: 0.6rem;
        background-color: #333;
        border-radius: 50%;
        top: -0.05rem;
        transform: translateX(-50%);
    }
    .symmetry-label {
        font-size: 0.7rem;
        color: #666;
        display: flex;
        justify-content: space-between;
    }
    h1 {
        color: #1f77b4;
        text-align: center;
    }
    .stProgress .st-bo {
        background-color: #1f77b4;
    }
    .sidebar-section {
        background-color: #f8f9fa;
        padding: 1rem;
        margin-bottom: 1rem;
        border-radius: 0.5rem;
    }
</style>
"""

# 步态分析标准参考值

# 步宽范围 (厘米)
STEP_WIDTH_RANGES = {
    'normal': (7, 9),  # 正常范围
    'mild': (9, 11),   # 轻度异常
    'moderate': (11, 13),  # 中度异常
    'severe': (13, float('inf'))  # 重度异常
}

# 步长范围 (厘米)
STEP_LENGTH_RANGES = {
    'normal': (50, 80),  # 正常范围
    'mild': (42.5, 47.5),  # 轻度异常下限
    'mild_upper': (73.5, 80.5),  # 轻度异常上限
    'moderate': (35, 42.5),  # 中度异常下限
    'moderate_upper': (80.5, 91),  # 中度异常上限
    'severe': (0, 35),  # 重度异常下限
    'severe_upper': (91, float('inf'))  # 重度异常上限
}

# 跨步长范围 (厘米)
STRIDE_LENGTH_RANGES = {
    'normal': (120, 160),  # 正常范围
    'mild': (140, 150),  # 轻度异常
    'moderate': (150, 160),  # 中度异常
    'severe': (160, float('inf'))  # 重度异常
}

# 步频范围 (步/分)
CADENCE_RANGES = {
    'normal': (95, 125),  # 正常范围
    'mild': (70, 95),  # 轻度异常下限
    'mild_upper': (95, 100),  # 轻度异常上限
    'moderate': (60, 70),  # 中度异常下限
    'moderate_upper': (100, 110),  # 中度异常上限
    'severe': (0, 60),  # 重度异常下限
    'severe_upper': (110, float('inf'))  # 重度异常上限
}

# 左右步长对称性差异范围 (厘米)
STEP_LENGTH_SYMMETRY_RANGES = {
    'normal': (0, 2),  # 正常范围
    'mild': (2, 5),  # 轻度异常
    'moderate': (5, 10),  # 中度异常
    'severe': (10, float('inf'))  # 重度异常
}

# 支撑相时间差异范围 (与正常范围的百分比)
SUPPORT_TIME_DIFF_RANGES = {
    'normal': (0, 20),  # 正常范围
    'mild': (20, 40),  # 轻度异常
    'moderate': (40, 60),  # 中度异常
    'severe': (60, float('inf'))  # 重度异常
}

# 摆动相时间差异范围 (与正常范围的百分比)
SWING_TIME_DIFF_RANGES = {
    'normal': (0, 20),  # 正常范围
    'mild': (20, 40),  # 轻度异常
    'moderate': (40, 60),  # 中度异常
    'severe': (60, float('inf'))  # 重度异常
}

# 骨盆旋转角度差异范围 (度)
PELVIC_ROTATION_RANGES = {
    'normal': (0, 3),  # 正常范围
    'mild': (3, 5),  # 轻度异常
    'moderate': (5, 10),  # 中度异常
    'severe': (10, float('inf'))  # 重度异常
}

# 膝关节屈伸角度差异范围 (度)
KNEE_FLEXION_RANGES = {
    'normal': (0, 5),  # 正常范围
    'mild': (5, 10),  # 轻度异常
    'moderate': (10, 20),  # 中度异常
    'severe': (20, float('inf'))  # 重度异常
}

# 踝关节背屈/跖屈角度差异范围 (度)
ANKLE_FLEXION_RANGES = {
    'normal': (0, 10),  # 正常范围
    'mild': (10, 15),  # 轻度异常
    'moderate': (15, 30),  # 中度异常
    'severe': (30, float('inf'))  # 重度异常
}

# 身体重心转移范围 (占步长的百分比)
WEIGHT_SHIFT_RANGES = {
    'normal': (0, 5),  # 正常范围
    'mild': (5, 11),  # 轻度异常
    'moderate': (11, 15),  # 中度异常
    'severe': (15, float('inf'))  # 重度异常
}

def get_severity_level(value, ranges):
    """根据数值确定异常程度"""
    if value is None:
        return None
        
    if ranges['normal'][0] <= value <= ranges['normal'][1]:
        return "正常"
    elif ranges.get('mild') and ranges['mild'][0] <= value <= ranges['mild'][1]:
        return "轻度异常"
    elif ranges.get('moderate') and ranges['moderate'][0] <= value <= ranges['moderate'][1]:
        return "中度异常"
    else:
        return "重度异常" 