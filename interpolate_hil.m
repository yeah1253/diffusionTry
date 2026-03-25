function signal_out = interpolate_hil(target_load, target_rpm, t)
%INTERPOLATE_HIL  Method B 插值 + 两阶段数据增强（Simulink MATLAB Function Block）
%
% ★ 配套文件: load_hil_mat_data.m（需与 .slx 和 HIL_data.mat 放在同一目录）
% ★ Simulink 模型中需添加 Clock 模块并连接到第三个输入端口 t
%
% 输入:
%   target_load     - 目标负载值
%   target_rpm      - 目标转速值
%   t               - 仿真当前时间（来自 Clock 模块，单位：秒）
% 输出:
%   signal_out      - 1×SIGNAL_LEN 时域信号，每 REGEN_INTERVAL 秒自动刷新

% ══ 配置参数（按需修改）══════════════════════════════════════
SIGNAL_LEN      = 1024;   % 信号长度，须与 .mat 中实际一致
MAX_COND        = 500;    % 最大工况数预分配（已增大以消除警告）
K_NEIGHBORS     = 6;      % IDW 最近邻数量
IDW_POWER       = 2.0;    % IDW 距离衰减指数
REGEN_INTERVAL  = 0.5;    % 每隔多少秒生成新的增强样本（秒）

% ══ extrinsic：数据加载函数在 MATLAB 中执行，不参与代码生成 ══
coder.extrinsic('load_hil_mat_data');

% ══ Persistent 缓存 ══════════════════════════════════════════
persistent sig_matrix load_vec rpm_vec n_cond data_ready
persistent base_signal cur_signal last_interval last_load last_rpm

% 默认输出（防止未赋值报错）
signal_out = zeros(1, SIGNAL_LEN);
F = floor(SIGNAL_LEN / 2) + 1;   % 单边谱长度

% ══ 首次调用：加载数据、插值基础信号、生成第一个增强样本 ═════
if isempty(data_ready)
    % 先声明所有 persistent 变量的类型和大小
    data_ready    = false;
    sig_matrix    = zeros(MAX_COND, SIGNAL_LEN);
    load_vec      = zeros(MAX_COND, 1);
    rpm_vec       = zeros(MAX_COND, 1);
    n_cond        = int32(0);
    base_signal   = zeros(1, SIGNAL_LEN);
    cur_signal    = zeros(1, SIGNAL_LEN);
    last_interval = -1.0;
    last_load     = target_load;
    last_rpm      = target_rpm;

    % 通过 extrinsic 加载 .mat 文件（可使用完整 MATLAB 特性）
    [sig_matrix, load_vec, rpm_vec, n_cond] = ...
        load_hil_mat_data(MAX_COND, SIGNAL_LEN);

    if n_cond > 0
        data_ready  = true;
        base_signal = compute_interp(sig_matrix, load_vec, rpm_vec, ...
                          double(n_cond), target_load, target_rpm, ...
                          K_NEIGHBORS, IDW_POWER, SIGNAL_LEN, F, MAX_COND);
        cur_signal    = generate_aug(base_signal, SIGNAL_LEN, F);
        last_interval = floor(t / REGEN_INTERVAL);
    end
end

if ~data_ready
    return;
end

% ══ 每步检查：是否需要生成新样本 ═════════════════════════════
load_chg = abs(target_load - last_load) > 1e-6;
rpm_chg  = abs(target_rpm  - last_rpm ) > 1e-6;
cur_ivl  = floor(t / REGEN_INTERVAL);

if load_chg || rpm_chg
    % 工况改变 → 重新插值基础信号并立即生成新增强样本
    last_load   = target_load;
    last_rpm    = target_rpm;
    base_signal = compute_interp(sig_matrix, load_vec, rpm_vec, ...
                      double(n_cond), target_load, target_rpm, ...
                      K_NEIGHBORS, IDW_POWER, SIGNAL_LEN, F, MAX_COND);
    cur_signal    = generate_aug(base_signal, SIGNAL_LEN, F);
    last_interval = cur_ivl;

elseif cur_ivl ~= last_interval
    % 到达下一个 0.5s 间隔 → 保持基础信号，生成新增强样本
    cur_signal    = generate_aug(base_signal, SIGNAL_LEN, F);
    last_interval = cur_ivl;
end

signal_out = cur_signal;

end  % ══ 主函数结束 ════════════════════════════════════════════


%% ════════════════════════════════════════════════════════════
%  辅助函数 1：Method B 基础插值（IDW + 谱幅度-相位解耦）
%% ════════════════════════════════════════════════════════════
function sig_out = compute_interp(sig_mat, lv, rv, nc, tl, tr, K, P, N, F, MCOND)
%COMPUTE_INTERP  对给定工况 (tl, tr) 执行 IDW 加权频谱插值

% 归一化尺度（各轴最小正差值）
ls = scale_of(lv, nc);
rs = scale_of(rv, nc);

% 计算到所有工况的距离（多余槽位填大值避免被选中）
dists = ones(MCOND, 1) * 1e15;
for i = 1:nc
    dl       = (lv(i) - tl) / ls;
    dr       = (rv(i) - tr) / rs;
    dists(i) = sqrt(dl * dl + dr * dr);
end

% 取最近 K 个邻居
[sorted_d, sort_idx] = sort(dists);
sel_idx   = zeros(K, 1);
sel_dists = zeros(K, 1);
for ki = 1:K
    sel_idx(ki)   = sort_idx(ki);
    sel_dists(ki) = sorted_d(ki);
end

k_act = K;
if nc < K
    k_act = nc;
end

% IDW 权重
weights = zeros(K, 1);
if sel_dists(1) < 1e-12
    weights(1) = 1.0;
else
    w_sum = 0.0;
    for ki = 1:K
        if ki <= k_act
            weights(ki) = 1.0 / (sel_dists(ki) + 1e-12) ^ P;
            w_sum = w_sum + weights(ki);
        end
    end
    for ki = 1:K
        if ki <= k_act
            weights(ki) = weights(ki) / w_sum;
        end
    end
end

% 加权频谱（幅度 + 相量分别插值）
mag_w  = zeros(1, F);
phs_re = zeros(1, F);
phs_im = zeros(1, F);

for ki = 1:K
    if ki <= k_act
        sp_full = fft(sig_mat(sel_idx(ki), :));
        sp_h    = sp_full(1:F);
        m_i     = abs(sp_h);
        u_i     = sp_h ./ (m_i + 1e-12);
        mag_w   = mag_w  + weights(ki) * m_i;
        phs_re  = phs_re + weights(ki) * real(u_i);
        phs_im  = phs_im + weights(ki) * imag(u_i);
    end
end

mag_w(1) = 0.0;   % 去直流

% 合成插值频谱并 IFFT
spec_out  = mag_w .* exp(1j * atan2(phs_im, phs_re));
full_sp   = complex(zeros(1, N), zeros(1, N));
full_sp(1:F) = spec_out;
if mod(N, 2) == 0
    full_sp(F+1:N) = conj(spec_out(F-1:-1:2));
else
    full_sp(F+1:N) = conj(spec_out(F:-1:2));
end

sig_out = real(ifft(full_sp));
end


%% ════════════════════════════════════════════════════════════
%  辅助函数 2：两阶段随机数据增强（对应 Python interpolate_hil.py）
%% ════════════════════════════════════════════════════════════
function aug = generate_aug(base, N, F)
%GENERATE_AUG  对基础信号执行两阶段随机增强，每次调用产生不同样本
%
% Stage 1 – 频域增强（在复频谱上操作）:
%   Step A: 谱包络调制 — 5控制点随机增益曲线（线性插值）× 复频谱
%   Step B: 保功率相位抖动 — 各频点叠加 ±0.25 rad 随机噪声
%
% Stage 2 – 时域增强（在重建波形上操作）:
%   Step C: 随机循环移位 + 幅度缩放
%   Step D: SNR 控制高斯噪声注入（SNR 在 [15, 25] dB 随机选取）

% ── Stage 1A: 谱包络调制 ─────────────────────────────────────
N_CTRL    = 5;
GAIN_LOW  = 0.6;
GAIN_HIGH = 1.4;

spec_full = fft(base);
spec_h    = spec_full(1:F);

% 生成 N_CTRL 个随机增益控制点，线性插值为逐频点增益曲线
y_ctrl   = GAIN_LOW + (GAIN_HIGH - GAIN_LOW) * rand(1, N_CTRL);
step_sz  = (double(F) - 1.0) / double(N_CTRL - 1);

gain_curve = zeros(1, F);
for fi = 1:F
    xn  = (double(fi) - 1.0) / step_sz;
    seg = floor(xn);
    if seg >= N_CTRL - 1
        gain_curve(fi) = y_ctrl(N_CTRL);
    else
        alpha = xn - double(seg);
        gain_curve(fi) = y_ctrl(seg + 1) * (1.0 - alpha) + y_ctrl(seg + 2) * alpha;
    end
    if gain_curve(fi) < 0.0
        gain_curve(fi) = 0.0;
    end
end

spec_h    = spec_h .* gain_curve;
spec_h(1) = 0.0;   % 保持去直流

% ── Stage 1B: 保功率相位抖动 ─────────────────────────────────
JITTER_RAD = 0.25;
mag_h      = abs(spec_h);
phase_h    = atan2(imag(spec_h), real(spec_h));
pnoise     = JITTER_RAD * (2.0 * rand(1, F) - 1.0);
pnoise(1)  = 0.0;   % DC 相位不抖动
spec_h     = mag_h .* exp(1j * (phase_h + pnoise));

% IFFT → Stage 1 时域输出
full_s1 = complex(zeros(1, N), zeros(1, N));
full_s1(1:F) = spec_h;
if mod(N, 2) == 0
    full_s1(F+1:N) = conj(spec_h(F-1:-1:2));
else
    full_s1(F+1:N) = conj(spec_h(F:-1:2));
end
s1 = real(ifft(full_s1));

% ── Stage 2C: 随机循环移位 + 幅度缩放 ───────────────────────
SCALE_LOW  = 0.85;
SCALE_HIGH = 1.15;
shift      = floor(rand() * double(N));
sc         = SCALE_LOW + (SCALE_HIGH - SCALE_LOW) * rand();

s2 = zeros(1, N);
for ii = 1:N
    src = mod(ii - 1 + shift, N) + 1;
    s2(ii) = s1(src) * sc;
end

% ── Stage 2D: SNR 控制高斯噪声注入 ──────────────────────────
SNR_LOW  = 15.0;
SNR_HIGH = 25.0;

s2_ac    = s2 - mean(s2);
ac_power = mean(s2_ac .^ 2);

snr_db    = SNR_LOW + (SNR_HIGH - SNR_LOW) * rand();
snr_lin   = 10.0 ^ (snr_db / 10.0);
nse_power = ac_power / snr_lin;
if nse_power < 0.0
    nse_power = 0.0;
end

aug = s2 + sqrt(nse_power) * randn(1, N);
end


%% ════════════════════════════════════════════════════════════
%  辅助函数 3：归一化尺度（最小正差值，兼容 Simulink Coder）
%% ════════════════════════════════════════════════════════════
function s = scale_of(arr, n)
%SCALE_OF  在 arr(1:n) 中找最小正差值，用于 IDW 归一化
    min_d = 1e15;
    for i = 1:n
        for j = i+1:n
            d = arr(i) - arr(j);
            if d < 0; d = -d; end
            if d > 1e-10 && d < min_d
                min_d = d;
            end
        end
    end
    if min_d > 1e14
        s = 1.0;
    else
        s = min_d;
    end
end
