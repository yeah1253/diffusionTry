function pack_hil_for_coder(max_cond, signal_len, hil_data_mat, out_dir)
%PACK_HIL_FOR_CODER  Build HIL_packed_for_codegen.mat for coder.const / Simulink Coder
%
% Run from MATLAB on the host before slbuild / code generation.
% Default: writes HIL_packed_for_codegen.mat next to this file.
%
% Optional:
%   max_cond, signal_len — must match interpolate_hil.m MAX_COND / SIGNAL_LEN（默认 1500, 1024）
%   hil_data_mat — path to HIL_data.mat, or folder containing it (default: this folder)
%   out_dir — folder for HIL_packed_for_codegen.mat (default: this folder)

if nargin < 1 || isempty(max_cond)
    max_cond = 1500;   % >= 1271 for full load×rpm grid; must match interpolate_hil MAX_COND
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

[sig_matrix, load_vec, rpm_vec, n_cond] = load_hil_mat_data(max_cond, signal_len, hil_data_mat);

n_cond = double(n_cond);
out_mat = fullfile(out_dir, 'HIL_packed_for_codegen.mat');
save(out_mat, 'sig_matrix', 'load_vec', 'rpm_vec', 'n_cond', '-v7');

fprintf('[pack_hil_for_coder] Wrote %s (n_cond=%g)\n', out_mat, n_cond);
end
