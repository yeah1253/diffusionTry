function signal_out = interpolate_hil(target_load, target_rpm)
% interpolate_hil — Method B 谱幅度-相位解耦插值 (Simulink MATLAB Function Block)
%
% 从 HIL_data.mat 中读取 HIL 实测信号网格，对给定目标工况进行插值，
% 返回重建后的时域信号（1×1024）。
%
% ┌─ 输入 ────────────────────────────────────────────────────┐
% │  target_load  : 目标负载值（对应 .matStruct 中 loadX 的 X）│
% │  target_rpm   : 目标转速值（对应 .matStruct 中 rpmY  的 Y）│
% └───────────────────────────────────────────────────────────┘
% ┌─ 输出 ────────────────────────────────────────────────────┐
% │  signal_out   : 插值重建的时域信号 [1 × SIGNAL_LEN]       │
% └───────────────────────────────────────────────────────────┘
%
% HIL_data.mat 数据结构（与图示一致）:
%   IF0_2, IF0_4, IF0_6  (1×1 struct)
%     .load0, .load25, ...  (struct，字段名末尾数字为实际负载值)
%       .rpm1000, .rpm2000, ... (struct，字段名末尾数字为实际转速值)
%         .filtered0, .filtered1, ... (1×1024 single，多个滤波样本)
%
% 使用方法:
%   1. 将本文件内容粘贴至 Simulink MATLAB Function Block
%   2. 确保 HIL_data.mat 与 .slx 文件在同一目录
%   3. 修改下方 FAULT_VAR 选择所需故障类型
%   4. 连接 target_load、target_rpm 输入（Constant Block 或信号源）

% ═══════════════════════════════════════════════════════════════
%  ★ 可配置参数（按需修改）
% ═══════════════════════════════════════════════════════════════
SIGNAL_LEN  = 1024;      % 信号长度（与 .mat 中实际长度一致）
K_NEIGHBORS = 6;         % IDW 最近邻数量
IDW_POWER   = 2.0;       % IDW 距离衰减指数
FAULT_VAR   = 'IF0_2';   % 选用的顶层故障变量名: 'IF0_2' / 'IF0_4' / 'IF0_6'

% ═══════════════════════════════════════════════════════════════
%  persistent 缓存：仅在仿真首次调用时加载一次数据
% ═══════════════════════════════════════════════════════════════
persistent sig_matrix load_vec rpm_vec data_ready

% 初始化输出，防止未赋值报错
signal_out = zeros(1, SIGNAL_LEN);

% ─── 第一次调用：加载并解析 .mat 文件 ─────────────────────────
if isempty(data_ready)
    data_ready = false;

    % 获取 Simulink 模型所在目录
    try
        mdl_dir = fileparts(which(bdroot));
    catch
        mdl_dir = '';
    end
    if isempty(mdl_dir)
        mdl_dir = pwd;
    end

    mat_path = fullfile(mdl_dir, 'HIL_data.mat');

    % 加载 .mat 并解析层级结构
    try
        raw          = load(mat_path, FAULT_VAR);
        fault_struct = raw.(FAULT_VAR);

        % ── 遍历 loadX 层 ──────────────────────────────────────
        load_fields = fieldnames(fault_struct);
        all_sigs  = [];   % 累积: 每行一条信号
        all_loads = [];   % 对应负载值
        all_rpms  = [];   % 对应转速值

        for li = 1:numel(load_fields)
            lf = load_fields{li};
            % 字段名须以 'load' 开头，后缀为实际负载数值
            if numel(lf) <= 4 || ~strcmp(lf(1:4), 'load')
                continue;
            end
            load_val = str2double(lf(5:end));
            if isnan(load_val)
                continue;
            end

            rpm_struct = fault_struct.(lf);
            rpm_fields = fieldnames(rpm_struct);

            % ── 遍历 rpmY 层 ───────────────────────────────────
            for ri = 1:numel(rpm_fields)
                rf = rpm_fields{ri};
                if numel(rf) <= 3 || ~strcmp(rf(1:3), 'rpm')
                    continue;
                end
                rpm_val = str2double(rf(4:end));
                if isnan(rpm_val)
                    continue;
                end

                filt_struct = rpm_struct.(rf);
                filt_fields = fieldnames(filt_struct);

                % ── 遍历 filteredZ 层：取均值作为代表信号 ────────
                acc_sig = zeros(1, SIGNAL_LEN);
                n_filt  = 0;
                for fi = 1:numel(filt_fields)
                    ff = filt_fields{fi};
                    if numel(ff) <= 8 || ~strcmp(ff(1:8), 'filtered')
                        continue;
                    end
                    raw_sig = double(filt_struct.(ff));
                    raw_sig = raw_sig(:)';   % 强制行向量
                    L = min(numel(raw_sig), SIGNAL_LEN);
                    acc_sig(1:L) = acc_sig(1:L) + raw_sig(1:L);
                    n_filt = n_filt + 1;
                end

                if n_filt == 0
                    continue;
                end
                mean_sig = acc_sig / n_filt;

                all_sigs  = [all_sigs;  mean_sig];   %#ok<AGROW>
                all_loads = [all_loads; load_val];   %#ok<AGROW>
                all_rpms  = [all_rpms;  rpm_val ];   %#ok<AGROW>
            end
        end

        if isempty(all_sigs)
            return;
        end

        sig_matrix = all_sigs;
        load_vec   = all_loads;
        rpm_vec    = all_rpms;
        data_ready = true;

    catch
        data_ready = false;
        return;
    end
end

if ~data_ready
    return;
end

% ═══════════════════════════════════════════════════════════════
%  IDW 权重计算
% ═══════════════════════════════════════════════════════════════
ls = compute_scale(load_vec);   % 负载轴归一化尺度
rs = compute_scale(rpm_vec);    % 转速轴归一化尺度

dists = sqrt( ((load_vec - target_load) ./ ls).^2 + ...
              ((rpm_vec  - target_rpm ) ./ rs).^2 );

n_cond   = length(dists);
k_actual = min(K_NEIGHBORS, n_cond);

[~, idx_sorted] = sort(dists, 'ascend');
sel_idx   = idx_sorted(1:k_actual);
sel_dists = dists(sel_idx);

if sel_dists(1) < 1e-12
    % 目标点与某网格点完全重合
    weights      = zeros(k_actual, 1);
    weights(1)   = 1.0;
else
    raw_w  = 1.0 ./ (sel_dists + 1e-12) .^ IDW_POWER;
    weights = raw_w ./ sum(raw_w);
end

% ═══════════════════════════════════════════════════════════════
%  Method B: 谱幅度-相位解耦插值
% ═══════════════════════════════════════════════════════════════
N = SIGNAL_LEN;
F = floor(N / 2) + 1;   % 单边频谱长度（对应 scipy.rfft 的输出长度）

mag_interp   = zeros(1, F);   % 加权幅度谱
phase_vec_re = zeros(1, F);   % 单位相量实部加权和
phase_vec_im = zeros(1, F);   % 单位相量虚部加权和

for ki = 1:k_actual
    sig_i  = sig_matrix(sel_idx(ki), :);   % 取第 ki 个邻居信号
    spec_i = fft(sig_i);                    % 全复数频谱
    spec_h = spec_i(1:F);                   % 单边（等价 rfft）

    mag_i  = abs(spec_h);
    unit_i = spec_h ./ (mag_i + 1e-12);    % 单位相量

    mag_interp   = mag_interp   + weights(ki) * mag_i;
    phase_vec_re = phase_vec_re + weights(ki) * real(unit_i);
    phase_vec_im = phase_vec_im + weights(ki) * imag(unit_i);
end

% 去直流（DC 频点置零，消除高直流偏置）
mag_interp(1) = 0.0;

% 合成插值复频谱
phase_interp = atan2(phase_vec_im, phase_vec_re);
spec_interp  = mag_interp .* exp(1j * phase_interp);

% 构造共轭对称全频谱 → IFFT 重建时域信号
full_spec = zeros(1, N);
full_spec(1:F) = spec_interp;
if mod(N, 2) == 0
    % N 为偶数：镜像区间 [F+1 .. N] 对应 [F-1 .. 2] 的共轭
    full_spec(F+1:N) = conj(spec_interp(F-1:-1:2));
else
    % N 为奇数
    full_spec(F+1:N) = conj(spec_interp(F:-1:2));
end

signal_out = real(ifft(full_spec));

end % ── 主函数结束 ──────────────────────────────────────────────


% ═══════════════════════════════════════════════════════════════
%  辅助函数：计算归一化尺度（中位数间距）
%  与 Python 版本 _scale() 逻辑完全一致
% ═══════════════════════════════════════════════════════════════
function s = compute_scale(arr)
    u = unique(arr);
    if numel(u) < 2
        s = 1.0;
        return;
    end
    d     = diff(sort(u(:)));
    d_pos = d(d > 0);
    if isempty(d_pos)
        s = 1.0;
    else
        s = median(d_pos);
    end
end
