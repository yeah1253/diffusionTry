function [sig_matrix, load_vec, rpm_vec, n_cond] = load_hil_mat_data(max_cond, signal_len, hil_data_mat, fault_var)
%LOAD_HIL_MAT_DATA  从 HIL_data.mat 加载信号数据（主机 / 打包脚本用）
%
% 仅由 pack_hil_for_coder / 交互调用。hil_get_const_data 必须用 coder.load，不得调用本文件，
% 否则 Simulink Coder 会分析 try/which/load 并报错。
% 构建模型前请运行 pack_hil_for_coder.m 生成 HIL_packed_for_codegen.mat。
%
% 第三参数 hil_data_mat（可选）: HIL_data.mat 的完整路径，或包含该文件的文件夹。
% 省略时仍尝试 bdroot 目录，否则当前工作目录。
%
% bearing_nested.mat / HIL_data.mat 数据结构（四级）:
%   IF0_2 / NC / OF0_2 / …  (1×1 struct，10 类工况)
%     .load0, .load25, …  (字段名 load 后数字 = 负载)
%       .rpm1000, .rpm1500, … (字段名 rpm 后数字 = 转速)
%         .filtered0 … .filtered14 (各 1×1024 single，每格多条样本)
%
% 打包：每个 filtered* 单独一行，同一 (load,rpm) 重复多行；interpolate_hil 在最近
% 工况格内对所有并列行均匀随机，等价于在该格 15 条样本中随机选一条。
%
% 第四参数 fault_var（可选）: .mat 顶层变量名，须与 HIL_data.mat 中一致，如 'IF0_2' / 'IF0_4' / 'IF0_6'

if nargin < 4 || isempty(fault_var)
    fault_var = 'IF0_2';
else
    fault_var = char(fault_var);
end

% 初始化输出（固定大小，由调用方的 max_cond 和 signal_len 决定）
sig_matrix = zeros(max_cond, signal_len);
load_vec   = zeros(max_cond, 1);
rpm_vec    = zeros(max_cond, 1);
n_cond     = int32(0);

% ── 定位 HIL_data.mat ────────────────────────────────────────
if nargin >= 3 && ~isempty(hil_data_mat)
    mat_path = char(hil_data_mat);
    if isfolder(mat_path)
        cand = fullfile(mat_path, 'bearing_nested.mat');
        if isfile(cand)
            mat_path = cand;
        else
            mat_path = fullfile(mat_path, 'HIL_data.mat');
        end
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
    raw = load(mat_path, fault_var);
catch ME
    warning('[load_hil_mat_data] load failed: %s', ME.message);
    return;
end

if ~isfield(raw, fault_var)
    warning('[load_hil_mat_data] Variable "%s" not in .mat', fault_var);
    return;
end

fault_struct = raw.(fault_var);
load_fields  = fieldnames(fault_struct);
load_order   = sort_load_fieldnames(load_fields);
cnt = int32(0);

% ── 遍历 loadX 层（按负载数值排序，保证各故障打包行序一致）──────
for li = 1:numel(load_order)
    lf = load_order{li};
    if isempty(lf)
        continue;
    end
    tokL = regexp(lf, '^load_?(\d+(\.\d+)?)$', 'tokens', 'once');
    if isempty(tokL)
        continue;
    end
    load_val = str2double(tokL{1});
    if isnan(load_val)
        continue;
    end

    rpm_struct = fault_struct.(lf);
    rpm_fields = fieldnames(rpm_struct);
    rpm_order  = sort_rpm_fieldnames(rpm_fields);

    % ── 遍历 rpmY 层 ─────────────────────────────────────────
    for ri = 1:numel(rpm_order)
        rf = rpm_order{ri};
        if isempty(rf)
            continue;
        end
        tokR = regexp(rf, '^rpm_?(\d+(\.\d+)?)$', 'tokens', 'once');
        if isempty(tokR)
            continue;
        end
        rpm_val = str2double(tokR{1});
        if isnan(rpm_val)
            continue;
        end

        filt_struct = rpm_struct.(rf);
        [sig_matrix, load_vec, rpm_vec, cnt, ok_cell] = ...
            append_samples_for_rpm_cell(filt_struct, load_val, rpm_val, ...
            sig_matrix, load_vec, rpm_vec, cnt, max_cond, signal_len);

        if cnt < 0
            n_cond = int32(max_cond);
            return;
        end
        if ~ok_cell
            warning('[load_hil_mat_data] 工况 (load=%g,rpm=%g) 下未识别到任何样本字段', load_val, rpm_val);
        end
    end
end

n_cond = cnt;
fprintf('[load_hil_mat_data] Loaded %s, %d rows (含每格多条 filtered 样本)\n', fault_var, n_cond);
end


function out = sort_load_fieldnames(fnames)
% load0 / load_0 / load25 等，按负载数值排序
    out = {};
    nums = [];
    for i = 1:numel(fnames)
        f = fnames{i};
        tok = regexp(f, '^load_?(\d+(\.\d+)?)$', 'tokens', 'once');
        if isempty(tok)
            continue;
        end
        v = str2double(tok{1});
        if isnan(v)
            continue;
        end
        out{end+1} = f; %#ok<AGROW>
        nums(end+1) = v; %#ok<AGROW>
    end
    if isempty(out)
        return;
    end
    [~, ord] = sort(nums);
    out = out(ord);
end


function out = sort_rpm_fieldnames(fnames)
% rpm1000 / rpm_1000 等，按转速数值排序
    out = {};
    nums = [];
    for i = 1:numel(fnames)
        f = fnames{i};
        tok = regexp(f, '^rpm_?(\d+(\.\d+)?)$', 'tokens', 'once');
        if isempty(tok)
            continue;
        end
        v = str2double(tok{1});
        if isnan(v)
            continue;
        end
        out{end+1} = f; %#ok<AGROW>
        nums(end+1) = v; %#ok<AGROW>
    end
    if isempty(out)
        return;
    end
    [~, ord] = sort(nums);
    out = out(ord);
end


function out = sort_filtered_fieldnames(fnames)
% filtered0 / filtered_0 / Filtered12（不区分大小写）按序号排序
    out = {};
    nums = [];
    for i = 1:numel(fnames)
        f = fnames{i};
        fl = lower(f);
        tok = regexp(fl, '^filtered_?(\d+)$', 'tokens', 'once');
        if isempty(tok)
            continue;
        end
        v = str2double(tok{1});
        if isnan(v)
            continue;
        end
        out{end+1} = fnames{i}; %#ok<AGROW>  % 保留原始字段名用于取值
        nums(end+1) = v; %#ok<AGROW>
    end
    if isempty(out)
        return;
    end
    [~, ord] = sort(nums);
    out = out(ord);
end


function [sig_matrix, load_vec, rpm_vec, cnt, ok_cell] = append_samples_for_rpm_cell( ...
    filt_struct, load_val, rpm_val, sig_matrix, load_vec, rpm_vec, cnt, max_cond, signal_len)
% 从一个 (load,rpm) struct 抽出多条样本，每条占一行（目标：每格 15 条 → 每故障约 24*15=360 行）
    ok_cell = false;
    fn = fieldnames(filt_struct);
    filt_order = sort_filtered_fieldnames(fn);

    % ── 方式 A：多个 filtered* 字段 ─────────────────────────
    for fii = 1:numel(filt_order)
        ff = filt_order{fii};
        if isempty(ff)
            continue;
        end
        raw_sig = double(filt_struct.(ff));
        raw_sig = raw_sig(:)';
        [sig_matrix, load_vec, rpm_vec, cnt, overflow] = append_one_row(sig_matrix, load_vec, rpm_vec, cnt, max_cond, signal_len, ...
            raw_sig, load_val, rpm_val);
        if overflow
            cnt = int32(-1);
            return;
        end
        ok_cell = true;
    end
    if ok_cell
        return;
    end

    % ── 方式 B：单字段二维数组 samples / filtered_stack / all_samples（N×L 或 L×N）──
    bundle_names = {'samples', 'filtered_stack', 'all_samples', 'sample_matrix', 'X'};
    for bi = 1:numel(bundle_names)
        bname = bundle_names{bi};
        if ~isfield(filt_struct, bname)
            continue;
        end
        S = double(filt_struct.(bname));
        if ndims(S) > 2 %#ok<ISMAT>
            continue;
        end
        [nr, nc] = size(S);
        if nr == 0 || nc == 0
            continue;
        end
        if nc == signal_len
            n_samp = nr;
            rows = S;
        elseif nr == signal_len
            n_samp = nc;
            rows = S';
        else
            continue;
        end
        for r = 1:n_samp
            raw_sig = rows(r, :);
            [sig_matrix, load_vec, rpm_vec, cnt, overflow] = append_one_row(sig_matrix, load_vec, rpm_vec, cnt, max_cond, signal_len, ...
                raw_sig, load_val, rpm_val);
            if overflow
                cnt = int32(-1);
                return;
            end
            ok_cell = true;
        end
        if ok_cell
            return;
        end
    end

    % ── 方式 C：cell 数组 filtered_cell / samples_cell ───────
    for ci = 1:numel(fn)
        f = fn{ci};
        fl = lower(f);
        if ~ismember(fl, {'filtered_cell', 'samples_cell', 'signals_cell'})
            continue;
        end
        C = filt_struct.(f);
        if ~iscell(C)
            continue;
        end
        for k = 1:numel(C)
            raw_sig = double(C{k});
            raw_sig = raw_sig(:)';
            [sig_matrix, load_vec, rpm_vec, cnt, overflow] = append_one_row(sig_matrix, load_vec, rpm_vec, cnt, max_cond, signal_len, ...
                raw_sig, load_val, rpm_val);
            if overflow
                cnt = int32(-1);
                return;
            end
            ok_cell = true;
        end
        if ok_cell
            return;
        end
    end
end


function [sig_matrix, load_vec, rpm_vec, cnt, overflow] = append_one_row(sig_matrix, load_vec, rpm_vec, cnt, max_cond, signal_len, ...
    raw_sig, load_val, rpm_val)
    overflow = false;
    nxt = cnt + int32(1);
    if nxt > int32(max_cond)
        warning('[load_hil_mat_data] Condition count exceeds MAX_COND=%d; increase pack_hil_for_coder max_cond', max_cond);
        overflow = true;
        return;
    end
    cnt = nxt;
    L = min(numel(raw_sig), signal_len);
    sig_matrix(cnt, 1:L) = raw_sig(1:L);
    load_vec(cnt) = load_val;
    rpm_vec(cnt)  = rpm_val;
end
