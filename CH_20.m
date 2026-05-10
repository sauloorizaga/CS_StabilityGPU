% ch_cs_parallel.m
% =========================================================
% CH Convex Splitting parallel GPU sweep
% Runs M simulations simultaneously, each with different a
%
% Domain: [0, 2pi] x [0, 2pi], N=64
% =========================================================

clear all;

% --- Fixed parameters ---
N   = 256;
dt  = 0.01;
eps = 0.1;
T   = 10.0;
steps = round(T/dt);

% --- a values to sweep ---
a_values = linspace(1, 2, 20);
M = length(a_values);

fprintf('\n==============================================\n');
fprintf('  CH CS Parallel Sweep over a values\n');
fprintf('  Grid: %dx%d, dt=%.4f, eps=%.2f\n', N, N, dt, eps);
fprintf('  T=%.1f, M=%d simulations\n', T, M);
fprintf('  a values: '); fprintf('%.2f ', a_values); fprintf('\n');
fprintf('==============================================\n\n');

% --- Wavenumbers ---
L   = 2*pi;
dx  = L/N;
k1d = [0:N/2-1, -N/2:-1];
[KX, KY] = meshgrid(k1d, k1d);
K2 = KX.^2 + KY.^2;
K4 = K2.^2;

% --- Same random initial condition for all ---
rng(42);
u0 = 0.01*randn(N,N);

%% --- Sequential CPU ---
fprintf('Running SEQUENTIAL on CPU...\n');
energy_initial = zeros(1,M);
energy_final   = zeros(1,M);

tic;
for j = 1:M
    a = a_values(j);
    u = u0;
    u_hat = fft2(u);
    denom = 1 + dt*eps^2*K4 + dt*a*K2;
    energy_initial(j) = eps^2/2 * sum(sum(K2.*abs(u_hat).^2))/N^2*dx^2 ...
                      + sum(sum(u.^4/4 - u.^2/2))*dx^2;
    for step = 1:steps
        NL     = u.^3 - (1+a)*u;
        NL_hat = fft2(NL);
        u_hat  = (u_hat - dt*K2.*NL_hat) ./ denom;
        u      = real(ifft2(u_hat));
    end
    energy_final(j) = eps^2/2 * sum(sum(K2.*abs(u_hat).^2))/N^2*dx^2 ...
                    + sum(sum(u.^4/4 - u.^2/2))*dx^2;
end
t_seq = toc;
fprintf('Sequential time: %.3f seconds\n\n', t_seq);

%% --- Parallel GPU ---
fprintf('Running PARALLEL on GPU...\n');

% stack into 3D: N x N x M
u3d = repmat(u0, [1,1,M]);

% broadcast a values: N x N x M
a3d = reshape(a_values, [1,1,M]);
a3d = repmat(a3d, [N,N,1]);

% 3D wavenumbers
K2_3d = repmat(K2, [1,1,M]);
K4_3d = repmat(K4, [1,1,M]);

% denominator per simulation
denom3d = 1 + dt*eps^2*K4_3d + dt*a3d.*K2_3d;

% send to GPU
u3d_gpu     = gpuArray(u3d);
a3d_gpu     = gpuArray(a3d);
K2_3d_gpu   = gpuArray(K2_3d);
denom3d_gpu = gpuArray(denom3d);

u_hat3d = fft2(u3d_gpu);

tic;
for step = 1:steps
    NL      = u3d_gpu.^3 - (1 + a3d_gpu).*u3d_gpu;
    NL_hat  = fft2(NL);
    u_hat3d = (u_hat3d - dt*K2_3d_gpu.*NL_hat) ./ denom3d_gpu;
    u3d_gpu = real(ifft2(u_hat3d));
end
wait(gpuDevice);
t_par = toc;
fprintf('Parallel time:   %.3f seconds\n', t_par);
fprintf('Speedup:         %.2fx\n\n', t_seq/t_par);

% gather
u3d_final    = gather(u3d_gpu);
uhat3d_final = gather(u_hat3d);

% final energies
energy_final_gpu = zeros(1,M);
for j = 1:M
    uj  = u3d_final(:,:,j);
    uhj = uhat3d_final(:,:,j);
    energy_final_gpu(j) = eps^2/2 * sum(sum(K2.*abs(uhj).^2))/N^2*dx^2 ...
                        + sum(sum(uj.^4/4 - uj.^2/2))*dx^2;
end

%% --- Results table ---
fprintf('%-8s  %-15s  %-15s  %-10s\n', 'a value', 'E_initial', 'E_final(GPU)', 'Decreased?');
fprintf('%s\n', repmat('-', 1, 55));
for j = 1:M
    decreased = energy_final_gpu(j) < energy_initial(j);
    fprintf('%-8.2f  %-15.6f  %-15.6f  %-10s\n', ...
        a_values(j), energy_initial(j), energy_final_gpu(j), mat2str(decreased));
end

%% --- Plot 20 panels: 4 rows x 5 cols ---
figure('Position', [100 100 1400 800]);
for j = 1:M
    subplot(4, 5, j);
    pcolor(u3d_final(:,:,j));
    shading interp;
    colormap(gca, redblue(256));
    caxis([-1 1]);
    axis square off;
    title(sprintf('a=%.2f\nE=%.2f', a_values(j), energy_final_gpu(j)), 'FontSize', 9);
end
sgtitle(sprintf('CH CS: 20 simulations in parallel, T=%.1f, dt=%.4f, Speedup=%.2fx', ...
    T, dt, t_seq/t_par), 'FontSize', 14);
saveas(gcf, 'ch_cs_parallel.png');
fprintf('\nPlot saved to ch_cs_parallel.png\n\n');

% --- Red-blue colormap ---
function c = redblue(m)
    if nargin < 1, m = 256; end
    top    = [1 0 0];
    middle = [1 1 1];
    bottom = [0 0 1];
    c = [interp1([1 m/2 m], [bottom(1) middle(1) top(1)], 1:m)', ...
         interp1([1 m/2 m], [bottom(2) middle(2) top(2)], 1:m)', ...
         interp1([1 m/2 m], [bottom(3) middle(3) top(3)], 1:m)'];
end