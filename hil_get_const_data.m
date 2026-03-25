function data = hil_get_const_data(max_cond, signal_len)
%HIL_GET_CONST_DATA  Pack struct for coder.const (Simulink Coder / Speedgoat)
%
% This file must contain ONLY constructs supported by MATLAB Coder, because it
% is analyzed when used inside coder.const(...) from a MATLAB Function block.
% Raw HIL_data.mat parsing stays in load_hil_mat_data.m (host / pack step only).
%
% Before building the model, run pack_hil_for_coder.m once (when HIL_data changes)
% to generate HIL_packed_for_codegen.mat next to this file / on the MATLAB path.
%
%   hil = coder.const(hil_get_const_data(MAX_COND, SIGNAL_LEN));

%#codegen

S = coder.load('HIL_packed_for_codegen.mat');

% Rows must equal MAX_COND from interpolate_hil; cols must equal SIGNAL_LEN.
% Repack with pack_hil_for_coder(MAX_COND, SIGNAL_LEN, ...) if this fails.
if size(S.sig_matrix, 2) ~= signal_len
    assert(false);
end
if size(S.sig_matrix, 1) ~= max_cond
    assert(false);
end
nc = double(S.n_cond);
if nc < 0.0 || nc > double(max_cond) || nc > double(size(S.sig_matrix, 1))
    assert(false);
end

data.sig_matrix = S.sig_matrix;
data.load_vec   = S.load_vec;
data.rpm_vec    = S.rpm_vec;
data.n_cond     = double(S.n_cond);

end
