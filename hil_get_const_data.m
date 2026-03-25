function packed = hil_get_const_data(max_cond, signal_len, packed_cols)
%HIL_GET_CONST_DATA  Build packed double matrix for coder.const (Simulink Coder)
%
% Returns packed [(max_cond+2) x packed_cols] double:
%   Row 1  : col1=n_cond, cols 2:max_cond+1 = load_vec (needs packed_cols >= max_cond+1)
%   Row 2  : col1=0,      cols 2:max_cond+1 = rpm_vec
%   Row 3+ : cols 1:signal_len = sig_matrix (needs packed_cols >= signal_len)
%
% Third arg packed_cols must be max(signal_len, max_cond+1) — use literal in caller
% (e.g. PACKED_COLS = 1501) so Simulink Coder can fold sizes.
%
%   packed = coder.const(hil_get_const_data(MAX_COND, SIGNAL_LEN, PACKED_COLS));

%#codegen

S = coder.load('HIL_packed_for_codegen.mat');

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
if packed_cols < signal_len || packed_cols < max_cond + 1
    assert(false);
end

packed = zeros(max_cond + 2, packed_cols);
packed(1, 1)            = nc;
packed(1, 2:max_cond+1) = S.load_vec(:, 1)';
packed(2, 2:max_cond+1) = S.rpm_vec(:, 1)';
packed(3:max_cond+2, 1:signal_len) = S.sig_matrix;

end
