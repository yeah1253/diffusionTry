function signal_out = interpolate_hil(target_load, target_rpm, fault_sel, t)
%INTERPOLATE_HIL  Method B 插值 + 两阶段数据增强（Simulink MATLAB Function Block）
%
% ★ 配套文件（需与 .slx 同目录或在 MATLAB 路径上）:
%     HIL_packed_for_codegen.mat — pack 生成，仅含变量 hil_packed（double）；coder.const 嵌入（必）
%     hil_get_const_data.m        — coder.load 读取上述 .mat（仅 Coder 可解析代码）
%     pack_hil_for_coder.m        — 主机运行；默认打包 IF0_2 / IF0_4 / IF0_6
%     load_hil_mat_data.m         — 仅 pack 脚本使用（含 try/which，不进 Coder）
%     HIL_data.mat                — 原始数据；打包时需要，目标机运行不需要
% ★ Simulink：四个输入依次为 target_load, target_rpm, fault_sel, t（Clock 接 t）
%
% 部署说明:
%   数据通过 coder.const 在 Build 时嵌入目标机，Speedgoat 运行时无需访问文件。
%   【分布对齐】离线训练数据多为「原始/滤波」波形；若 USE_RANDOM_AUGMENT_FOR_UDP=true，每步随机增强
%   会使 UDP 分布与训练集严重偏离，PC 端准确率会显著低于 val/test。部署诊断建议置为 false，仅发插值基信号。
%   UDP 发送：将 signal_out 连接到 Simulink UDP Send 块；每包为 129×double（与 PC 端 receive_udp_hil.py 一致）:
%            第 1 个 double = 当前真实类别标签（以 double 发送，PC 端转 int；此处为 0..N_FAULT-1）；
%            后 128 个 double = 振动数据。UDP Send / Byte Pack 宽度须设为 129。
%
% 输入:
%   target_load     - 目标负载值
%   target_rpm      - 目标转速值
%   fault_sel       - 故障类型：1=IF0_2，2=IF0_4，3=IF0_6（与 HIL_data.mat 变量名一致）
%   t               - 仿真当前时间（来自 Clock 模块，单位：秒）
% 输出:
%   signal_out      - 1×(1+128) 向量：标签 + 振动分片；1024 点仍分 8 包发出，每包带同一 GT 标签

% ══ 配置参数（按需修改）══════════════════════════════════════
SIGNAL_LEN          = 1024;   % 信号长度，须与 .mat 中实际一致
SAMPLES_PER_PACKET  = 128;    % 每包中振动采样点数
PACKET_LEN          = 1 + SAMPLES_PER_PACKET;  % 1 标签 + 128 数据 = 129（与 receive_udp_hil 默认 DOUBLES_PER_PACKET 一致）
NUM_PACKETS         = int32(SIGNAL_LEN / SAMPLES_PER_PACKET); % 1024/128 = 8
MAX_COND        = 1500;   % 须 >= 实际工况数；与 pack_hil_for_coder / HIL_packed 行数一致
N_FAULT         = 10;      % 打包的故障种类数，须与 pack_hil_for_coder 中 fault_list 一致
PACKED_COLS     = SIGNAL_LEN;
if (MAX_COND + 1) > PACKED_COLS
    PACKED_COLS = MAX_COND + 1;   % 行1需列 1..(MAX_COND+1) 放 n_cond 与 load_vec
end
K_NEIGHBORS     = 6;      % IDW 最近邻数量
IDW_POWER       = 2.0;    % IDW 距离衰减指数
REGEN_INTERVAL  = 0.5;    % 每隔多少秒刷新 cur_signal（仅当 USE_RANDOM_AUGMENT_FOR_UDP=true 时有随机性）
% false=UDP 发送 IDW 插值后的 base_signal（推荐，贴近 HIL_train/.npy 与训练分布）；true=原 generate_aug 强随机增强
USE_RANDOM_AUGMENT_FOR_UDP = false;

% ══ Persistent 缓存 ══════════════════════════════════════════
% 注意：已移除占用极大的 sig_matrix 缓存
persistent sig_tensor load_vec rpm_vec n_cond data_ready
persistent base_signal cur_signal last_interval last_load last_rpm pkt_idx last_fault_idx

% 默认输出（防止未赋值报错；长度与 UDP 包一致）
signal_out = zeros(1, PACKET_LEN);
F = floor(SIGNAL_LEN / 2) + 1;   % 单边谱长度

fault_idx = int32(round(fault_sel));
if fault_idx < int32(1)
    fault_idx = int32(1);
end
if fault_idx > int32(N_FAULT)
    fault_idx = int32(N_FAULT);
end

% ══ 首次调用：通过 coder.const 加载数据（Speedgoat 兼容）══════
if isempty(data_ready)
    data_ready    = false;
    sig_tensor    = zeros(N_FAULT, MAX_COND, SIGNAL_LEN);
    load_vec      = zeros(MAX_COND, 1);
    rpm_vec       = zeros(MAX_COND, 1);
    n_cond        = int32(0);
    base_signal   = zeros(1, SIGNAL_LEN);
    cur_signal    = zeros(1, SIGNAL_LEN);
    last_interval = -1.0;
    last_load     = target_load;
    last_rpm      = target_rpm;
    pkt_idx       = int32(0);
    last_fault_idx = int32(0);

    % ★ coder.const：多故障纵向堆叠
    BR = MAX_COND + 2;
    packed     = coder.const(hil_get_const_data(MAX_COND, SIGNAL_LEN, PACKED_COLS, N_FAULT));
    n_cond     = int32(packed(1, 1));
    load_vec   = packed(1, 2:MAX_COND+1)';
    rpm_vec    = packed(2, 2:MAX_COND+1)';
    for kf = 1:N_FAULT
        b = (kf - 1) * BR;
        sig_tensor(kf, 1:MAX_COND, 1:SIGNAL_LEN) = packed(b+3:b+BR, 1:SIGNAL_LEN);
    end

    last_fault_idx = fault_idx;

    if n_cond > 0
        data_ready  = true;
        % 直接将 sig_tensor 和 fault_idx 传入，按需提取
        base_signal = compute_interp(sig_tensor, fault_idx, load_vec, rpm_vec, ...
                          double(n_cond), target_load, target_rpm, ...
                          K_NEIGHBORS, IDW_POWER, SIGNAL_LEN, F, MAX_COND);
        if USE_RANDOM_AUGMENT_FOR_UDP
            cur_signal = generate_aug(base_signal, SIGNAL_LEN, F);
        else
            cur_signal = base_signal;
        end
        last_interval = floor(t / REGEN_INTERVAL);
    end
end

if ~data_ready
    return;
end

% ══ 每步检查：是否需要生成新样本 ═════════════════════════════
load_chg = abs(target_load - last_load) > 1e-6;
rpm_chg  = abs(target_rpm  - last_rpm ) > 1e-6;
fault_changed = (fault_idx ~= last_fault_idx);
cur_ivl  = floor(t / REGEN_INTERVAL);

% 故障切换时，仅更新索引标志，不执行全局内存拷贝
if fault_changed
    last_fault_idx = fault_idx;
end

if load_chg || rpm_chg || fault_changed
    % 负载 / 转速 / 故障类型 改变 → 重新插值基础信号并立即生成新增强样本
    last_load   = target_load;
    last_rpm    = target_rpm;
    % 传入 3D 张量和当前选定的故障层索引
    base_signal = compute_interp(sig_tensor, fault_idx, load_vec, rpm_vec, ...
                      double(n_cond), target_load, target_rpm, ...
                      K_NEIGHBORS, IDW_POWER, SIGNAL_LEN, F, MAX_COND);
    if USE_RANDOM_AUGMENT_FOR_UDP
        cur_signal = generate_aug(base_signal, SIGNAL_LEN, F);
    else
        cur_signal = base_signal;
    end
    last_interval = cur_ivl;
    pkt_idx       = int32(0);  % 新样本从第 1 段开始发送

elseif cur_ivl ~= last_interval
    % 到达下一个时间间隔：有增强则重新随机；无增强则重复发送当前 base（由 load/rpm 未变时 base 未重算）
    if USE_RANDOM_AUGMENT_FOR_UDP
        cur_signal = generate_aug(base_signal, SIGNAL_LEN, F);
    else
        cur_signal = base_signal;
    end
    last_interval = cur_ivl;
    pkt_idx       = int32(0);
end

% 每次调用输出一包 UDP：[GT 标签 | 128 点振动]
% 标签：fault_idx 为 1..N_FAULT（与 fault_sel 一致），发送 double(fault_idx-1) 供 PC 端作 0..num_classes-1
signal_out = zeros(1, PACKET_LEN);
signal_out(1) = double(fault_idx - 1);
idx_start = double(pkt_idx) * double(SAMPLES_PER_PACKET) + 1;
for i = 1:SAMPLES_PER_PACKET
    signal_out(1 + i) = cur_signal(idx_start + i - 1);
end
% 下次调用输出下一段（8 包凑满 1024 点振动，每包均带同一标签）
pkt_idx = pkt_idx + 1;
if pkt_idx >= NUM_PACKETS
    pkt_idx = int32(0);
end

end  % ══ 主函数结束 ════════════════════════════════════════════


%% ════════════════════════════════════════════════════════════
%  辅助函数 1：Method B 基础插值（IDW + 谱幅度-相位解耦）
%% ════════════════════════════════════════════════════════════
function sig_out = compute_interp(sig_tensor, fault_idx, lv, rv, nc, tl, tr, K, P, N, F, MCOND)
%COMPUTE_INTERP  对给定工况 (tl, tr) 执行 IDW 加权频谱插值，按需提取 3D 数据

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
        % =========================================================
        % 【核心修改点】仅在计算 FFT 前，动态提取需要用到的 1024 个点
        % =========================================================
        r_idx = sel_idx(ki);
        sig_1d = zeros(1, N);
        for c = 1:N
            sig_1d(c) = sig_tensor(fault_idx, r_idx, c);
        end

        sp_full = fft(sig_1d);
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
%  辅助函数 2：两阶段随机数据增强
%% ════════════════════════════════════════════════════════════
function aug = generate_aug(base, N, F)
%GENERATE_AUG  对基础信号执行两阶段随机增强，每次调用产生不同样本

% ── Stage 1A: 谱包络调制 ─────────────────────────────────────
N_CTRL    = 5;
GAIN_LOW  = 0.6;
GAIN_HIGH = 1.4;

spec_full = fft(base);
spec_h    = spec_full(1:F);

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
pnoise(1)  = 0.0;
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