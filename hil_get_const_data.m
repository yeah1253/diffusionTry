function packed = hil_get_const_data(max_cond, signal_len, packed_cols, n_fault)
%HIL_GET_CONST_DATA  Load pre-built packed matrix for coder.const (Simulink Coder)
%
% HIL_packed_for_codegen.mat must contain only one variable: hil_packed (double matrix).
% Run pack_hil_for_coder.m to generate it — avoids coder.load multi-field struct issues.
%
% 尺寸须与 interpolate_hil.m 中 MAX_COND、PACKED_COLS、N_FAULT 一致。
% 有效样本行数 n_cond 存于 packed(1,1)（每故障块第一行首列），可大于 24（含每工况 15 条等）。
%
%   packed = coder.const(hil_get_const_data(MAX_COND, SIGNAL_LEN, PACKED_COLS, N_FAULT));

%#codegen

S = coder.load('HIL_packed_for_codegen.mat');
packed = S.hil_packed;

% assert message must be ASCII-only for Simulink Coder (GBK codegen issue with CJK)
if size(packed, 1) ~= n_fault * (max_cond + 2)
    assert(false);
end
if size(packed, 2) ~= packed_cols
    assert(false);
end

end
