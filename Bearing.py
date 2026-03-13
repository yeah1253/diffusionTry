import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import hilbert
import matplotlib.pyplot as plt
import matplotlib as mpl

# 故障类型映射（含部位/深度mm）；NC 表示正常，不调用机理模型
FAULT_TYPE_MAP = {
    "NC":   {"component": "normal", "depth_mm": 0.0},
    "IF0.2": {"component": "inner",  "depth_mm": 0.2},
    "IF0.4": {"component": "inner",  "depth_mm": 0.4},
    "IF0.6": {"component": "inner",  "depth_mm": 0.6},
    "OF0.2": {"component": "outer",  "depth_mm": 0.2},
    "OF0.4": {"component": "outer",  "depth_mm": 0.4},
    "OF0.6": {"component": "outer",  "depth_mm": 0.6},
    "RF0.2": {"component": "ball",   "depth_mm": 0.2},
    "RF0.4": {"component": "ball",   "depth_mm": 0.4},
    "RF0.6": {"component": "ball",   "depth_mm": 0.6},
}

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
mpl.rcParams['figure.max_open_warning'] = 0  # 禁止最大图形警告

# 全局变量
global m1, m2, c1, c2, k1, k2, Fx, Fy, Wx, Wy, N, w, wc, wb, D, Dm, L, BPFO, BPFI, BPFB, delta_t, theta_dt, IO, fault_type

# 可选: 'ball' | 'outer' | 'inner'
fault_type = 'ball'

Fx_history = []
Fy_history = []

def set_fault_by_key(fault_key: str):
    """根据故障键设置全局 fault_type 与故障尺寸 L (m)。"""
    global fault_type, L
    if fault_key not in FAULT_TYPE_MAP:
        raise ValueError(f"Unknown fault key: {fault_key}")
    if fault_key == "NC":
        raise RuntimeError("NC (Normal) provided: Bearing model should be skipped.")
    info = FAULT_TYPE_MAP[fault_key]
    fault_type = info["component"]  # 'inner' | 'outer' | 'ball'
    L = info["depth_mm"] * 1e-3


# 定义常微分方程函数
def ode(t, y):
    global m1, m2, c1, c2, k1, k2, Fx, Fy, Wx, Wy, N, w, wc, wb, D, Dm, L, BPFO, BPFI, BPFB, delta_t, theta_dt, IO, fault_type

    dy = np.zeros(8)

    Fx = 0
    Fy = 0

    D = 7.938e-3  # 滚动体直径 (m) 6205 -> 7.938mm
    Dm = 38.5e-3  # 节圆直径 (m) 近似 (25mm+52mm)/2
    m1 = 0.11005
    c1 = 1376
    k1 = 4.241 * 10 ** 4
    m2 = 3.405461
    c2 = 2210.7
    k2 = 15.1056 * 10 ** 6
    N = 900  # 转速 (RPM)
    delta = 5 * 10 ** (-6)
    Ri = (Dm - D) / 2 - delta
    Ro = (Dm + D) / 2 + delta
    w = N / 60 * 2 * np.pi  # 轴转速 (rad/s)
    wc = 0.5 * (w * (1 - D / Dm))  # 保持架转速 (rad/s)
    wb = (0.5 * Dm * w / D) * (1 - (D / Dm) ** 2)  # 滚动体自转速度 (rad/s)
    Nb = 9  # 滚动体数量 (6205 型号)
    alpha_0 = 0
    gamma = (D * np.cos(alpha_0)) / Dm
    delta = 5 * 10 ** (-6) #游隙
    Wx = 0 # x方向外部载荷
    Wy = -0.9849 # y方向外部载荷
    L = L  # 使用全局故障尺寸 (由 set_fault_by_key 设置)



    # 计算接触刚度
    sigma = 0.26
    fi = 0.515
    fo = 0.525
    rho_11 = 2 / D
    rho_12 = 2 / D
    E0 = 19e10

    # 内外接触位移
    n_delta_i = 0.577
    n_delta_o = 0.68

    # 内滚道曲率和
    rho_22_i = -(1 / (fi * D))
    rho_21_i = (2 * gamma) / (D * (1 - gamma))
    sigma_rho_i = rho_11 + rho_12 + rho_21_i + rho_22_i
    ki = ((2 * np.sqrt(2) * E0) / (1 - sigma ** 2)) / ((9 * ((n_delta_i) ** 3) * sigma_rho_i) ** 0.5)

    # 外滚道曲率和
    rho_22_o = -(1 / (fo * D))
    rho_21_o = -(2 * gamma) / (D * (1 + gamma))
    sigma_rho_o = rho_11 + rho_12 + rho_21_o + rho_22_o
    ko = ((2 * np.sqrt(2) * E0) / (1 - sigma ** 2)) / ((9 * ((n_delta_o) ** 3) * sigma_rho_o) ** 0.5)

    # 等效接触刚度
    kb = ((ki ** (-2 / 3)) + (ko ** (-2 / 3))) ** (-1.5)

    # 故障特征频率
    BPFO = (Nb * N / 120) * (1 - D / Dm)  # 外圈故障特征频率
    BPFI = (Nb * N / 120) * (1 + D / Dm)  # 内圈故障特征频率
    BPFB = (Dm * N / (60 * D)) * (1 - (D / Dm) ** 2)  # 滚动体故障特征频率

    # 故障深度
    dr = D / 2 - ((D / 2) ** 2 - (L / 2) ** 2) ** 0.5
    d = 0.1e-3
    if dr >= d:
        Hmax = d
    else:
        Hmax = dr

    # 故障角度相关，内外圈
    theta_dt_o=np.arcsin(L/Ro)  #外圈故障宽度对应角度
    theta_dt_i = np.arcsin(L / Ri)  #内圈故障宽度对应角度
    theta_dt_b = np.arcsin(L / D) #滚动体故障宽度对应角度
    theta_dc_o = 0 #外圈故障所在角位置
    theta_dc_i = w * t #内圈故障所在角位置
    delta_t = theta_dt_o / (w - wc) #双冲击间隔

    # 滚动体
    theta_dt_b = np.arcsin(L / D)

    for j in range(1, Nb + 1):
        # 确定theta_j
        theta_j = ((2 * np.pi) / Nb) * (j - 1) + (wc) * t  # 第j个滚动体所在角位置

        # 根据故障类型确定H（故障轮廓）
        if fault_type == 'ball':
            if abs(np.mod(wb * t, 2 * np.pi)) < theta_dt_b:
                H = dr
            elif abs(np.mod(wb * t, 2 * np.pi) - np.pi) < theta_dt_b:
                H = dr
            else:
                H = 0
        elif fault_type == 'outer':
            theta_j_mod = np.mod(theta_j, 2 * np.pi)
            if theta_j_mod <= theta_dt_o / 2 or theta_j_mod >= 2 * np.pi - theta_dt_o / 2:
                H = dr
            else:
                H = 0
        elif fault_type == 'inner':
            rel = np.mod(theta_j - theta_dc_i + np.pi, 2 * np.pi) - np.pi
            if abs(rel) <= theta_dt_i / 2:
                H = dr
            else:
                H = 0
        else:
            H = 0

        # 确定gamma_j
        gamma_j = (y[0] - y[4]) * np.cos(theta_j) + (y[2] - y[6]) * np.sin(theta_j) - delta - H

        if gamma_j <= 0:
            lambda_j = 0
        else:
            lambda_j = 1

        # 修复标量幂运算错误：确保gamma_j为正并使用安全的幂运算
        if gamma_j > 0:
            # 使用np.power代替**操作符可以更安全
            Fx = Fx + kb * lambda_j * np.power(gamma_j, 1.5) * np.cos(theta_j)
            Fy = Fy + kb * lambda_j * np.power(gamma_j, 1.5) * np.sin(theta_j)
    Fx_history.append(Fx)
    Fy_history.append(Fy)

    # 微分方程
    dy[0] = y[1]  # x1：转轴径向水平方向绝对速度
    dy[1] = (1 / m1 * (Wx - Fx - c1 * y[1] - k1 * y[0]))  # x1：转轴径向水平方向绝对加速度
    dy[2] = y[3]  # x2：转轴径向垂直方向绝对速度
    dy[3] = (1 / m1 * (Wy - Fy - c1 * y[3] - k1 * y[2]))  # x2：转轴径向垂直方向绝对加速度
    dy[4] = y[5]  # x3：轴承座径向水平方向绝对速度
    dy[5] = (1 / m2 * (Fx - c2 * y[5] - k2 * y[4]))  # x3：轴承座径向水平方向绝对加速度
    dy[6] = y[7]  # y1：轴承座径向垂直方向绝对速度
    dy[7] = (1 / m2 * (Fy - c2 * y[7] - k2 * y[6]))  # y1：轴承座径向垂直方向绝对加速度

    return dy


# FFT函数
def FFT(t, y):
    tspan = t[-1] - t[0]
    fs = len(t) / tspan
    p = np.abs(np.fft.fft(y) / (fs * tspan))
    p = p[:int(np.floor((fs * tspan) / 2 + 1))]
    p[1:-1] = 2 * p[1:-1]
    p[0] = 0
    f = np.linspace(0, 1, int((fs * tspan) / 2 + 1)) * fs / 2
    return f, p


# 主仿真
def main(fault_key: str = None, no_plot: bool = False, target_len: int = None):
    """
    Run bearing simulation.
    - fault_key: one of FAULT_TYPE_MAP keys; if None, uses current global fault_type/L
    - no_plot: if True, skip plotting and return phys_signal (dy[:,7])
    - target_len: if set, resample phys_signal to this length
    """
    global L, w, wc, BPFO, BPFI, BPFB, delta_t, theta_dt, wb, fault_type, Fx_history, Fy_history

    # reset histories for fresh run
    Fx_history = []
    Fy_history = []

    if fault_key is not None:
        if fault_key == "NC":
            raise RuntimeError("FAULT_KEY=NC, Bearing simulation should be skipped.")
        set_fault_by_key(fault_key)

    if fault_type not in {'ball', 'outer', 'inner'}:
        raise ValueError("fault_type must be one of: 'ball', 'outer', 'inner'")

    tspan1 = 1e-5
    tspan = np.arange(0, 2 + tspan1, tspan1)

    y0 = [1e-6, 0, 1e-6, 0, 1e-6, 0, 1e-6, 0]

    try:
        # 求解ODE系统
        print("开始求解微分方程...")
        sol = solve_ivp(ode, [0, 2], y0, method='RK45', t_eval=tspan, rtol=1e-3, atol=1e-6)
        t = sol.t
        y = sol.y.T
        print("微分方程求解完成")

        # 切除初始瞬态响应
        cut1 = 0
        while t[cut1] < 0.5 and cut1 < len(t) - 1:
            cut1 += 1

        t = t[cut1:]
        y = y[cut1:, :]
        a=np.array(Fx_history)
        b=np.array(Fy_history)



        # 计算加速度
        print("计算加速度...")
        dy = np.zeros_like(y)
        dt = np.diff(t)
        for i in range(8):
            dy[:-1, i] = np.diff(y[:, i]) / dt
            dy[-1, i] = dy[-2, i]  # 设置最后一点为倒数第二点

        # 信息输出
        # 计算参数值
        D = 7.938e-3  # 滚动体直径 (m)
        Dm = 38.5e-3  # 节圆直径 (m)
        N = 900  # 转速 (RPM)
        delta = 5 * 10 ** (-6)
        Ri = (Dm - D) / 2 - delta
        Ro = (Dm + D) / 2 + delta
        w = N / 60 * 2 * np.pi  # 轴转速 (rad/s)
        wc = 0.5 * (w * (1 - D / Dm))  # 保持架转速 (rad/s)
        wb = (0.5 * Dm * w / D) * (1 - (D / Dm) ** 2)  # 滚动体自转速度 (rad/s)
        Nb = 9  # 滚动体数量
        L = L  # 使用全局故障尺寸


        BPFO = (Nb * N / 120) * (1 - D / Dm)  # 外圈故障特征频率
        BPFI = (Nb * N / 120) * (1 + D / Dm)  # 内圈故障特征频率
        BPFB = (Dm * N / (60 * D)) * (1 - (D / Dm) ** 2)  # 滚动体故障特征频率

        theta_dt_o = np.arcsin(L / Ro)  # 外圈故障宽度对应角度
        theta_dt_i = np.arcsin(L / Ri)  # 内圈故障宽度对应角度
        theta_dt_b = np.arcsin(L / D)  # 滚动体故障宽度对应角度
        theta_dc_o = 0  # 外圈故障所在角位置
        theta_dc_i = w * t  # 内圈故障所在角位置
        delta_t = theta_dt_o / (w - wc)  # 双冲击间隔

        dr = D / 2 - ((D / 2) ** 2 - (L / 2) ** 2) ** 0.5

        if no_plot:
            phys_signal = dy[:, 7]  # 轴承座垂直方向加速度
            if target_len is not None and len(phys_signal) != target_len:
                idx = np.linspace(0, len(phys_signal) - 1, target_len).astype(int)
                phys_signal = phys_signal[idx]
            # 归一化到 [-1,1]
            pmin, pmax = phys_signal.min(), phys_signal.max()
            prange = pmax - pmin + 1e-8
            phys_signal = 2.0 * (phys_signal - pmin) / prange - 1.0
            return phys_signal.astype(np.float32)

        print(f"转轴速度 = {w:.4f} rad/s")
        print(f"保持架转速 = {wc:.4f} rad/s")
        print(f"滚动体自转速度 = {wb:.4f} rad/s")
        print(f"外圈故障特征频率(BPFO) = {BPFO:.4f} Hz")
        print(f"内圈故障特征频率(BPFI) = {BPFI:.4f} Hz")
        print(f"滚动体故障特征频率(BPFB) = {BPFB:.4f} Hz")
        print(f"双冲击时间间隔(delta_t) = {delta_t:.4f} s")
        print(f"损伤角大小(theta_dt) = {theta_dt_i:.4f} rad")
        print(f"最大深度(dr) = {dr * 1000:.4f} mm")

        # 绘制时间-加速度曲线
        print("绘制时间-加速度曲线...")
        plt.figure(1, figsize=(12, 10))

        plt.subplot(2, 1, 1)
        plt.plot(t, dy[:, 1])
        plt.grid(True)
        plt.xlabel('时间 (s)')
        plt.ylabel('加速度 (m/s²)')
        plt.title('内圈水平方向：时间-加速度图')

        plt.subplot(2, 1, 2)
        plt.plot(t, dy[:, 3])
        plt.grid(True)
        plt.xlabel('时间 (s)')
        plt.ylabel('加速度 (m/s²)')
        plt.title('内圈垂直方向：时间-加速度图\n故障尺寸 = {} mm'.format(L * 10 ** 3))

        plt.tight_layout()
        plt.savefig('内圈加速度.png')  # 保存图片

        plt.figure(2, figsize=(12, 10))

        plt.subplot(2, 1, 1)
        plt.plot(t, dy[:, 5])
        plt.grid(True)
        plt.xlim(0.6, 1)
        # plt.ylim(-0.005, 0.005)
        plt.xlabel('时间 (s)')
        plt.ylabel('加速度 (m/s²)')
        plt.title('外圈水平方向：时间-加速度图\n故障尺寸 = {} mm'.format(L * 10 ** 3))

        plt.subplot(2, 1, 2)
        plt.plot(t, dy[:, 7])
        plt.grid(True)
        plt.xlim(0.6,1)
        # plt.ylim(-0.005, 0.005)
        plt.xlabel('时间 (s)')
        plt.ylabel('加速度 (m/s²)')
        plt.title('外圈垂直方向：时间-加速度图\n故障尺寸 = {} mm'.format(L * 10 ** 3))
        print(t, dy[:, 7])
        plt.tight_layout()
        plt.savefig('外圈加速度.png')  # 保存图片

        # 计算包络和FFT
        print("计算包络和频谱...")
        Hdy = np.zeros_like(dy)
        ep = np.zeros((0, 8))
        pp = np.zeros((0, 8))
        ffe = np.zeros(0)
        ffp = np.zeros(0)

        for yni in range(8):
            Hdy[:, yni] = np.abs(hilbert(dy[:, yni] - np.mean(dy[:, yni])))
            ffe_tmp, ep_tmp = FFT(t, Hdy[:, yni])
            ffp_tmp, pp_tmp = FFT(t, dy[:, yni])

            if yni == 0:
                ffe = ffe_tmp
                ffp = ffp_tmp
                ep = np.column_stack((ep, ep_tmp)) if ep.size > 0 else ep_tmp.reshape(-1, 1)
                pp = np.column_stack((pp, pp_tmp)) if pp.size > 0 else pp_tmp.reshape(-1, 1)
            else:
                ep = np.column_stack((ep, ep_tmp))
                pp = np.column_stack((pp, pp_tmp))

        # 绘制包络谱
        # 频率标注随转速(N)和故障类型自动变化
        marker_map = {
            'ball': ('ball', [BPFB / 2.0, BPFB, 1.5 * BPFB], 'red'),
            'outer': ('outer', [BPFO, 2 * BPFO, 3 * BPFO], 'green'),
            'inner': ('inner', [BPFI, 2 * BPFI, 3 * BPFI], 'purple')
        }
        marker_label, marker_freqs, marker_color = marker_map[fault_type]
        additional_freq = N / 60.0
        # 颜色定义（避免使用蓝色）
        colors = ['red', 'green', 'purple']


        # 标注额外频率

        print("绘制包络谱...")
        plt.figure(4, figsize=(10, 6))
        plt.plot(ffe, ep[:, 7])
        plt.xlim(0, 500)
        plt.grid(True)
        for f in marker_freqs:
            plt.axvline(x=f, color=marker_color, linestyle='--', linewidth=1.5, label=f'{marker_label} ({f:.2f}Hz)')

        # 标注额外频率
        plt.axvline(x=additional_freq, color='orange', linestyle='--', linewidth=1.5,
                    label=f'Rotation Freq ({additional_freq:.2f}Hz)')
        plt.legend(loc='upper right', bbox_to_anchor=(1, 1))

        plt.xlabel('频率 (Hz)')
        plt.ylabel('幅值')
        plt.title('包络谱')
        plt.savefig('包络谱.png')  # 保存图片

        # 绘制频谱图
        print("绘制频谱图...")
        plt.figure(5, figsize=(10, 6))
        plt.plot(ffp, pp[:, 7])
        plt.xlim(0, 6000)
        plt.grid(True)
        plt.xlabel('频率 (Hz)')
        plt.ylabel('幅值')
        plt.title('频谱图\n故障尺寸 = {} mm'.format(L * 10 ** 3))
        plt.savefig('频谱图.png')  # 保存图片

        print("显示图形...")
        plt.show()

    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 选择故障类型：IF0.2/IF0.4/IF0.6/OF0.2/OF0.4/OF0.6/RF0.2/RF0.4/RF0.6；NC 不调用本脚本
    FAULT_KEY = "IF0.2"
    if FAULT_KEY == "NC":
        print("FAULT_KEY=NC, 跳过 Bearing 机理仿真。")
    else:
        set_fault_by_key(FAULT_KEY)
        main()
