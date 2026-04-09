function pack_hil_for_coder(max_cond, signal_len, hil_data_mat, out_dir, fault_list)
%PACK_HIL_FOR_CODER  Build HIL_packed_for_codegen.mat for coder.const / Simulink Coder
%
% bearing_nested.mat 四级结构：工况 -> load* -> rpm* -> filtered0..filteredN（各 1×1024）。
% load_hil_mat_data 将每个 filtered* 打成一行；同一 (load,rpm) 下多行共享相同 load/rpm，
% interpolate_hil 在最近工况格内均匀随机选一行发送（UDP 包格式不变）。
%
% 要求：fault_list 中各故障在 .mat 内行数 n_cond 一致（网格与 filtered 数相同）。
%
% Optional:
%   max_cond — 须 >= 总行数（约 负载数×转速数×每格样本数）；须与 interpolate_hil.m 一致
%   signal_len — 默认 1024
%   hil_data_mat — bearing_nested.mat 或 HIL_data.mat 路径；或文件夹（优先 bearing_nested.mat）
%   out_dir — 输出目录
%   fault_list — 与 .mat 顶层变量名一致，顺序即 fault_sel 1..N

if nargin < 1 || isempty(max_cond)
    % 须 >= 每故障行数：约 (负载档数×转速档数×每格样本数)，如 24×15=360
    max_cond = 1500;
end
if nargin < 2 || isempty(signal_len)
    signal_len = 1024;
end
here = fileparts(mfilename('fullpath'));
if nargin < 3 || isempty(hil_data_mat)
    hil_data_mat = fullfile(here, 'bearing_nested.mat');
end
if nargin < 4 || isempty(out_dir)
    out_dir = here;
end
if nargin < 5 || isempty(fault_list)
    fault_list = {'IF0_2', 'IF0_4', 'IF0_6','NC','OF0_2', 'OF0_4', 'OF0_6','RF0_2', 'RF0_4', 'RF0_6'};
end

nf = numel(fault_list);
sig_tensor = zeros(nf, max_cond, signal_len);
load_vec   = zeros(max_cond, 1);
rpm_vec    = zeros(max_cond, 1);
n_cond     = int32(0);
n_ref      = int32(0);

for k = 1:nf
    fv = fault_list{k};
    [sig_matrix, load_vec, rpm_vec, n_cond] = load_hil_mat_data(max_cond, signal_len, hil_data_mat, fv);
    sig_tensor(k, 1:max_cond, 1:signal_len) = sig_matrix;
    if k == 1
        n_ref = n_cond;
    elseif n_cond ~= n_ref
        error('pack_hil_for_coder: 故障 %s 行数 n_cond=%d 与首个故障 n_cond=%d 不一致，请检查 .mat', ...
            fv, n_cond, n_ref);
    end
end

packed_cols = signal_len;
if (max_cond + 1) > packed_cols
    packed_cols = max_cond + 1;
end

hil_packed = hil_build_packed_matrix(sig_tensor, load_vec, rpm_vec, double(n_cond), ...
    max_cond, signal_len, packed_cols, nf);

out_mat = fullfile(out_dir, 'HIL_packed_for_codegen.mat');
save(out_mat, 'hil_packed', '-v7');

fprintf('[pack_hil_for_coder] Wrote %s (hil_packed size [%d x %d])\n', ...
    out_mat, size(hil_packed, 1), size(hil_packed, 2));
fprintf('[pack_hil_for_coder] n_cond=%d rows/fault (expect 工况数×每工况样本数，如 24×15=360)\n', n_ref);
fprintf('[pack_hil_for_coder] interpolate_hil.m 中 MAX_COND 须 >= %d\n', double(n_ref));
end


function P = hil_build_packed_matrix(sig_tensor, load_vec, rpm_vec, n_cond, max_cond, signal_len, packed_cols, n_fault)
% Same layout as interpolate_hil unpack (N_FAULT blocks of MAX_COND+2 rows).
block_rows = max_cond + 2;
P = zeros(n_fault * block_rows, packed_cols);
nc = n_cond;
for fi = 1:n_fault
    offset = (fi - 1) * block_rows;
    P(offset + 1, 1)            = nc;
    P(offset + 1, 2:max_cond+1) = load_vec(:, 1)';
    P(offset + 2, 2:max_cond+1) = rpm_vec(:, 1)';
    sm = sig_tensor(fi, 1:max_cond, 1:signal_len);
    P(offset + 3:offset + block_rows, 1:signal_len) = sm;
end
end
