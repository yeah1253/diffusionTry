function [sig_matrix, load_vec, rpm_vec, n_cond] = load_hil_mat_data(max_cond, signal_len, hil_data_mat)
%LOAD_HIL_MAT_DATA  从 HIL_data.mat 加载信号数据（主机 / 打包脚本用）
%
% 不再由 hil_get_const_data 调用（避免 Simulink Coder 分析本文件）。
% 构建模型前请运行 pack_hil_for_coder.m 生成 HIL_packed_for_codegen.mat。
%
% 第三参数 hil_data_mat（可选）: HIL_data.mat 的完整路径，或包含该文件的文件夹。
% 省略时仍尝试 bdroot 目录，否则当前工作目录。
%
% HIL_data.mat 数据结构:
%   IF0_2 / IF0_4 / IF0_6  (1×1 struct)
%     .load0, .load25, ...  (字段名末尾数字 = 实际负载值)
%       .rpm1000, .rpm2000, ... (字段名末尾数字 = 实际转速值)
%         .filtered0, .filtered1, ... (1×1024 single，多滤波样本)
%
% ★ 修改 FAULT_VAR 选择故障类型: 'IF0_2' / 'IF0_4' / 'IF0_6'

FAULT_VAR = 'IF0_2';

% 初始化输出（固定大小，由调用方的 max_cond 和 signal_len 决定）
sig_matrix = zeros(max_cond, signal_len);
load_vec   = zeros(max_cond, 1);
rpm_vec    = zeros(max_cond, 1);
n_cond     = int32(0);

% ── 定位 HIL_data.mat ────────────────────────────────────────
if nargin >= 3 && ~isempty(hil_data_mat)
    mat_path = char(hil_data_mat);
    if isfolder(mat_path)
        mat_path = fullfile(mat_path, 'HIL_data.mat');
    end
else
    try
        mdl_dir = fileparts(which(bdroot));
    catch
        mdl_dir = '';
    end
    if isempty(mdl_dir)
        mdl_dir = pwd;
    end
    mat_path = fullfile(mdl_dir, 'HIL_data.mat');
end

if ~isfile(mat_path)
    warning('[load_hil_mat_data] File not found: %s', mat_path);
    return;
end

% ── 加载并解析层级结构 ────────────────────────────────────────
try
    raw = load(mat_path, FAULT_VAR);
catch ME
    warning('[load_hil_mat_data] load failed: %s', ME.message);
    return;
end

if ~isfield(raw, FAULT_VAR)
    warning('[load_hil_mat_data] Variable "%s" not in .mat', FAULT_VAR);
    return;
end

fault_struct = raw.(FAULT_VAR);
load_fields  = fieldnames(fault_struct);
cnt = int32(0);

% ── 遍历 loadX 层 ─────────────────────────────────────────────
for li = 1:numel(load_fields)
    lf = load_fields{li};
    if numel(lf) <= 4 || ~strcmp(lf(1:4), 'load')
        continue;
    end
    load_val = str2double(lf(5:end));
    if isnan(load_val)
        continue;
    end

    rpm_struct = fault_struct.(lf);
    rpm_fields = fieldnames(rpm_struct);

    % ── 遍历 rpmY 层 ─────────────────────────────────────────
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

        % ── 遍历 filteredZ 层：多样本取均值 ─────────────────
        acc_sig = zeros(1, signal_len);
        n_filt  = 0;
        for fi = 1:numel(filt_fields)
            ff = filt_fields{fi};
            if numel(ff) <= 8 || ~strcmp(ff(1:8), 'filtered')
                continue;
            end
            raw_sig = double(filt_struct.(ff));
            raw_sig = raw_sig(:)';   % 强制行向量
            L = min(numel(raw_sig), signal_len);
            acc_sig(1:L) = acc_sig(1:L) + raw_sig(1:L);
            n_filt = n_filt + 1;
        end

        if n_filt == 0
            continue;
        end

        cnt = cnt + int32(1);
        if cnt > int32(max_cond)
            warning('[load_hil_mat_data] Condition count exceeds MAX_COND=%d; increase limit', max_cond);
            n_cond = cnt - int32(1);
            return;
        end

        sig_matrix(cnt, :) = acc_sig / n_filt;
        load_vec(cnt) = load_val;
        rpm_vec(cnt)  = rpm_val;
    end
end

n_cond = cnt;
fprintf('[load_hil_mat_data] Loaded %s, %d conditions\n', FAULT_VAR, n_cond);
end
