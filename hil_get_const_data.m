function data = hil_get_const_data(max_cond, signal_len)
%HIL_GET_CONST_DATA  将 HIL 数据打包为结构体，供 coder.const 在构建时调用
%
% 本函数仅在 Simulink 构建（Build）阶段由主机 PC 执行，
% 结果作为编译期常量嵌入 Speedgoat 二进制文件，运行时不访问任何文件。
%
% 调用方式（在 interpolate_hil.m 中）:
%   hil = coder.const(hil_get_const_data(MAX_COND, SIGNAL_LEN));

[sig_matrix, load_vec, rpm_vec, n_cond] = load_hil_mat_data(max_cond, signal_len);

data.sig_matrix = sig_matrix;          % [max_cond × signal_len] double
data.load_vec   = load_vec;            % [max_cond × 1] double
data.rpm_vec    = rpm_vec;             % [max_cond × 1] double
data.n_cond     = double(n_cond);      % scalar double（coder.const 不支持 int32 顶层字段）

fprintf('[hil_get_const_data] 数据打包完成，共 %d 个工况，将嵌入目标机二进制文件\n', n_cond);
end
