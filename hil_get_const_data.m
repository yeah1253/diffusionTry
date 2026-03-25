function packed = hil_get_const_data(max_cond, signal_len)
%HIL_GET_CONST_DATA  将 HIL 数据打包为单一 double 矩阵，供 coder.const 使用
%
% 返回: packed [(max_cond+2) × signal_len] double 矩阵
%   Row 1  : [n_cond,  load_vec(1..max_cond),  zeros...]
%   Row 2  : [0,       rpm_vec(1..max_cond),   zeros...]
%   Row 3+ : sig_matrix（每行一个工况信号）
%
% 修改说明:
%   原版返回 struct，coder.const 在 Simulink Real-Time 代码生成时不支持 struct，
%   导致"断言失败"。改为返回 double 矩阵可彻底解决该问题。

[sig_matrix, load_vec, rpm_vec, n_cond] = load_hil_mat_data(max_cond, signal_len);

packed = zeros(max_cond + 2, signal_len);

packed(1, 1)            = double(n_cond);
packed(1, 2:max_cond+1) = load_vec(:)';    % 负载值存第1行的第2..max_cond+1列
packed(2, 2:max_cond+1) = rpm_vec(:)';     % 转速值存第2行的第2..max_cond+1列
packed(3:max_cond+2, :) = sig_matrix;      % 信号矩阵存第3行起

fprintf('[hil_get_const_data] 打包完成: %d 个工况, 矩阵尺寸 [%d x %d]\n', ...
        n_cond, size(packed,1), size(packed,2));
end
