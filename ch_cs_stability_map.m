% ch_cs_stability_map.m
% =========================================================
% CH Convex Splitting STABILITY MAP
% Sweeps over (a, dt) parameter space
% Runs ALL simulations in parallel on GPU
%
% Yellow = stable (energy decreases)
% Purple = unstable (energy increases or blows up)
%
% Domain: [0, 2pi] x [0, 2pi], N=64
% =========================================================

clear all;

% --- Fixed parameters ---
N   = 64;        % grid size (change to 256 for higher accuracy)
eps = 0.1;
T   = 10.0;       % short time, enough to detect instability

% --- Parameter sweep ---
na  = 100;
ndt = 100;
a_values  = linspace(0, 2, na);
dt_values = linspace(0.01, 5, ndt);

% Total simulations
M = na * ndt;

fprintf('\n==============================================\n');
fprintf('  CH CS Stability Map\n');
fprintf('  Grid: %dx%d, eps=%.2f, T=%.1f\n', N, N, eps, T);
fprintf('  a:  %d values from %.2f to %.2f\n', na, a_values(1), a_values(end));
fprintf('  dt: %d values from %.3f to %.3f\n', ndt, dt_values(1), dt_values(end));
fprintf('  Total simulations: %d\n', M);
fprintf('==============================================\n\n');

% --- Wavenumbers for [0, 2pi] ---
L   = 2*pi;
dx  = L/N;
k1d = [0:N/2-1, -N/2:-1];
[KX, KY] = meshgrid(k1d, k1d);
K2  = KX.^2 + KY.^2;
K4  = K2.^2;

% --- Build all (a, dt) combinations ---
% Each simulation j corresponds to (a_values(ia), dt_values(idt))
% We flatten the 2D grid into 1D of length M = na*ndt
[A_grid, DT_grid] = meshgrid(a_values, dt_values);  % ndt x na
a_flat  = A_grid(:)';    % 1 x M
dt_flat = DT_grid(:)';   % 1 x M

% --- Same random IC for all simulations ---
rng(42);
u0 = 0.01*randn(N,N);

% --- Build 3D arrays: N x N x M ---
fprintf('Building 3D arrays...\n');
u3d    = repmat(u0, [1,1,M]);

% broadcast a and dt across domain
a3d  = reshape(a_flat,  [1,1,M]);
a3d  = repmat(a3d,  [N,N,1]);
dt3d = reshape(dt_flat, [1,1,M]);
dt3d = repmat(dt3d, [N,N,1]);

K2_3d = repmat(K2, [1,1,M]);
K4_3d = repmat(K4, [1,1,M]);

% denominator: different for each (a, dt) combination
denom3d = 1 + dt3d.*eps^2.*K4_3d + dt3d.*a3d.*K2_3d;

% --- Send to GPU ---
fprintf('Sending to GPU...\n');
u3d_gpu     = gpuArray(u3d);
a3d_gpu     = gpuArray(a3d);
dt3d_gpu    = gpuArray(dt3d);
K2_3d_gpu   = gpuArray(K2_3d);
denom3d_gpu = gpuArray(denom3d);

% --- Compute initial energy ---
u_hat3d = fft2(u3d_gpu);
E_initial = eps^2/2 * sum(sum(K2_3d_gpu.*abs(u_hat3d).^2,1),2)/N^2*dx^2 ...
          + sum(sum(u3d_gpu.^4/4 - u3d_gpu.^2/2, 1), 2)*dx^2;
E_initial = gather(E_initial(:));

% --- Time stepping ---
fprintf('Running %d simulations on GPU...\n', M);
tic;

% each simulation has different dt so steps vary!
% use max steps and just run all to T=5
max_steps = round(T / min(dt_values));
fprintf('Max steps: %d\n\n', max_steps);

% track current time for each simulation
t_current = zeros(1,1,M,'gpuArray');
t_current = gpuArray(zeros(1,1,M));

for step = 1:max_steps
    % only update simulations that haven't reached T yet
    NL      = u3d_gpu.^3 - (1 + a3d_gpu).*u3d_gpu;
    NL_hat  = fft2(NL);
    u_hat3d = (u_hat3d - dt3d_gpu.*K2_3d_gpu.*NL_hat) ./ denom3d_gpu;
    u3d_gpu = real(ifft2(u_hat3d));
    t_current = t_current + dt3d_gpu(1,1,:);
end
wait(gpuDevice);
t_gpu = toc;
fprintf('GPU time: %.2f seconds\n\n', t_gpu);

% --- Compute final energy ---
u_hat3d_final = fft2(u3d_gpu);
E_final = eps^2/2 * sum(sum(K2_3d_gpu.*abs(u_hat3d_final).^2,1),2)/N^2*dx^2 ...
        + sum(sum(u3d_gpu.^4/4 - u3d_gpu.^2/2, 1), 2)*dx^2;
E_final = gather(E_final(:));

% --- Stability: 1=stable, 0=unstable ---
stable = double(E_final < E_initial);

% --- Reshape to 2D map: ndt x na ---
stable_map = reshape(stable, ndt, na);

fprintf('Stable:   %d / %d (%.1f%%)\n', sum(stable), M, 100*sum(stable)/M);
fprintf('Unstable: %d / %d (%.1f%%)\n', M-sum(stable), M, 100*(M-sum(stable))/M);

% --- Plot stability map ---
figure('Position', [100 100 800 600]);

% custom colormap: purple=unstable, yellow=stable
cmap = [0.4 0.0 0.6;   % purple = unstable
        1.0 0.9 0.0];  % yellow = stable
colormap(cmap);

pcolor(dt_values, a_values, stable_map');
shading flat;
colormap(cmap);
colorbar('Ticks', [0.25 0.75], 'TickLabels', {'Unstable', 'Stable'}, 'FontSize', 13);
caxis([0 1]);
xlabel('\Deltat (timestep)', 'FontSize', 14);
ylabel('a (splitting parameter)', 'FontSize', 14);
title(sprintf('CS Scheme Stability Map — %d simulations on GPU (%.1fs)', M, t_gpu), ...
    'FontSize', 14);
hold on;
yline(2.0, 'w--', 'LineWidth', 2, 'Label', 'a=2 (theory)', ...
    'LabelOrientation', 'horizontal', 'FontSize', 12);
hold off;
grid off;
box on;
set(gca, 'FontSize', 12);

saveas(gcf, 'ch_cs_stability_map.png');
fprintf('\nPlot saved to ch_cs_stability_map.png\n\n');
