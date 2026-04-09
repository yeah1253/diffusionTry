function signal_out = interpolate_hil(target_load, target_rpm, fault_sel, t)
%INTERPOLATE_HIL  最近 (load,rpm) 网格格 + 格内均匀随机选一条 1024 样本（无插值、无增强）
%
% ★ 配套文件（需与 .slx 同目录或在 MATLAB 路径上）:
%     HIL_packed_for_codegen.mat — pack_hil_for_coder 生成；hil_packed（double）；coder.const 嵌入（必）
%     hil_get_const_data.m        — coder.load 读取上述 .mat
%     pack_hil_for_coder.m        — 从 bearing_nested.mat 打包；每 filtered* 一行
%     load_hil_mat_data.m         — 仅打包用，不进 Coder
% ★ Simulink：四个输入依次为 target_load, target_rpm, fault_sel, t（Clock 接 t）
%
% 逻辑说明（与 bearing_nested 四级结构对应）:
%   - 打包后：同一 (load,rpm) 对应多行（如 filtered0..filtered14），load_vec/rpm_vec 相同；
%   - 用归一化欧氏距离找离 (target_load, target_rpm) 最近的点；最小距离上的所有行并列；
%   - 在并列行中 uniform 随机选一行 → 即在该工况格内随机选一条 filtered 样本；
%   - 每条 1024 分 8 包发完后，在工况与故障不变时再次在同格并列行中随机选一条（不重复循环同一条）；
%   - UDP 输出格式不变：1 标签 + 128 点/包，共 8 包凑满 1024。
%
% UDP 发送：将 signal_out 连接到 Simulink UDP Send 块；每包为 129×double（与 PC 端 receive_udp_hil.py 一致）:
%            第 1 个 double = 当前真实类别标签（0..N_FAULT-1）；
%            后 128 个 double = 振动数据。
%
% 输入:
%   target_load     - 目标负载值
%   target_rpm      - 目标转速值
%   fault_sel       - 故障类型 1..N_FAULT，顺序与 pack_hil_for_coder 的 fault_list 一致（默认 10 类）
%   t               - 仿真当前时间（来自 Clock 模块，单位：秒）
% 输出:
%   signal_out      - 1×(1+128) 向量：标签 + 振动分片；1024 点分 8 包发出，每包带同一 GT 标签；
%                   发满 8 包后换同工况下另一条随机样本，负载/转速/fault_sel 变化时亦重新选样并重置包序

% ══ 配置参数（按需修改）══════════════════════════════════════
SIGNAL_LEN          = 1024;
SAMPLES_PER_PACKET  = 128;
PACKET_LEN          = 1 + SAMPLES_PER_PACKET;
NUM_PACKETS         = int32(SIGNAL_LEN / SAMPLES_PER_PACKET);
MAX_COND        = 1500;
N_FAULT         = 10;
PACKED_COLS     = SIGNAL_LEN;
if (MAX_COND + 1) > PACKED_COLS
    PACKED_COLS = MAX_COND + 1;
end

% ══ Persistent 缓存 ══════════════════════════════════════════
persistent sig_tensor load_vec rpm_vec n_cond data_ready
persistent cur_signal last_load last_rpm pkt_idx last_fault_idx

signal_out = zeros(1, PACKET_LEN);

fault_idx = int32(round(fault_sel));
if fault_idx < int32(1)
    fault_idx = int32(1);
end
if fault_idx > int32(N_FAULT)
    fault_idx = int32(N_FAULT);
end

if isempty(data_ready)
    data_ready    = false;
    sig_tensor    = zeros(N_FAULT, MAX_COND, SIGNAL_LEN);
    load_vec      = zeros(MAX_COND, 1);
    rpm_vec       = zeros(MAX_COND, 1);
    n_cond        = int32(0);
    cur_signal    = zeros(1, SIGNAL_LEN);
    last_load     = target_load;
    last_rpm      = target_rpm;
    pkt_idx       = int32(0);
    last_fault_idx = int32(0);

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
        cur_signal = pick_random_nearest_signal(sig_tensor, fault_idx, load_vec, rpm_vec, ...
            double(n_cond), target_load, target_rpm, SIGNAL_LEN, MAX_COND);
    end
end

if ~data_ready
    return;
end

load_chg = abs(target_load - last_load) > 1e-6;
rpm_chg  = abs(target_rpm  - last_rpm ) > 1e-6;
fault_changed = (fault_idx ~= last_fault_idx);

if fault_changed
    last_fault_idx = fault_idx;
end

if load_chg || rpm_chg || fault_changed
    last_load   = target_load;
    last_rpm    = target_rpm;
    cur_signal = pick_random_nearest_signal(sig_tensor, fault_idx, load_vec, rpm_vec, ...
        double(n_cond), target_load, target_rpm, SIGNAL_LEN, MAX_COND);
    pkt_idx     = int32(0);
end

signal_out = zeros(1, PACKET_LEN);
signal_out(1) = double(fault_idx - 1);
idx_start = double(pkt_idx) * double(SAMPLES_PER_PACKET) + 1;
for i = 1:SAMPLES_PER_PACKET
    signal_out(1 + i) = cur_signal(idx_start + i - 1);
end
pkt_idx = pkt_idx + 1;
if pkt_idx >= NUM_PACKETS
    pkt_idx = int32(0);
    % 发满一条 1024 后，工况与故障未变时也在最近格内再随机选一条，避免无限重复同一样本
    cur_signal = pick_random_nearest_signal(sig_tensor, fault_idx, load_vec, rpm_vec, ...
        double(n_cond), target_load, target_rpm, SIGNAL_LEN, MAX_COND);
end

end


%% ════════════════════════════════════════════════════════════
%  最近工况 + 并列最近时随机选一条（无插值）
%% ════════════════════════════════════════════════════════════
function sig_out = pick_random_nearest_signal(sig_tensor, fault_idx, lv, rv, nc, tl, tr, N, MCOND)
%PICK_RANDOM_NEAREST_SIGNAL  最近 (load,rpm)；并列含同格全部 filtered 行 → 均匀随机一条

ls = scale_of(lv, nc);
rs = scale_of(rv, nc);

dists = ones(MCOND, 1) * 1e15;
for i = 1:nc
    dl       = (lv(i) - tl) / ls;
    dr       = (rv(i) - tr) / rs;
    dists(i) = sqrt(dl * dl + dr * dr);
end

dmin = 1e15;
for i = 1:nc
    if dists(i) < dmin
        dmin = dists(i);
    end
end

n_tie = int32(0);
tie_idx = zeros(MCOND, 1);
for i = 1:nc
    if dists(i) - dmin < 1e-9
        n_tie = n_tie + int32(1);
        tie_idx(n_tie) = i;
    end
end

sig_out = zeros(1, N);
if n_tie < int32(1)
    return;
end

r = rand();
pick = int32(floor(r * double(n_tie))) + int32(1);
if pick > n_tie
    pick = n_tie;
end
if pick < int32(1)
    pick = int32(1);
end
idx = tie_idx(pick);

for c = 1:N
    sig_out(c) = sig_tensor(fault_idx, idx, c);
end
end


%% ════════════════════════════════════════════════════════════
%  归一化尺度（与旧 IDW 一致）
%% ════════════════════════════════════════════════════════════
function s = scale_of(arr, n)
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
