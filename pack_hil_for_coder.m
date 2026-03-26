function pack_hil_for_coder(max_cond, signal_len, hil_data_mat, out_dir, fault_list)
%PACK_HIL_FOR_CODER  Build HIL_packed_for_codegen.mat for coder.const / Simulink Coder
%
% Saves a single variable hil_packed (double) so hil_get_const_data can coder.load it
% without multi-field struct inference errors.
%
% Optional:
%   max_cond, signal_len — must match interpolate_hil.m
%   hil_data_mat — path to HIL_data.mat, or folder containing it
%   out_dir — output folder for HIL_packed_for_codegen.mat
%   fault_list — cellstr of fault variable names, e.g. {'IF0_2','IF0_4','IF0_6'}

if nargin < 1 || isempty(max_cond)
    max_cond = 1500;
end
if nargin < 2 || isempty(signal_len)
    signal_len = 1024;
end
here = fileparts(mfilename('fullpath'));
if nargin < 3 || isempty(hil_data_mat)
    hil_data_mat = fullfile(here, 'HIL_data.mat');
end
if nargin < 4 || isempty(out_dir)
    out_dir = here;
end
if nargin < 5 || isempty(fault_list)
    fault_list = {'IF0_2', 'IF0_4', 'IF0_6'};
end

nf = numel(fault_list);
sig_tensor = zeros(nf, max_cond, signal_len);
load_vec   = zeros(max_cond, 1);
rpm_vec    = zeros(max_cond, 1);
n_cond     = int32(0);

for k = 1:nf
    fv = fault_list{k};
    [sig_matrix, load_vec, rpm_vec, n_cond] = load_hil_mat_data(max_cond, signal_len, hil_data_mat, fv);
    sig_tensor(k, 1:max_cond, 1:signal_len) = sig_matrix;
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
