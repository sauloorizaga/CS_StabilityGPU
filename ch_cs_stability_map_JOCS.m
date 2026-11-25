% ch_cs_stability_map_JOCS.m
% =========================================================
% CH Convex Splitting STABILITY MAP - JOCS paper
% Sweeps over (a, dt); all simulations advanced together on the GPU.
%
% Stable   = E_h(U^{n+1}) <= E_h(U^n) at EVERY step (with tolerance)
%            and all values finite.
% Unstable = some stepwise energy increase above tolerance, or NaN/Inf.
%
% mode = 'fixedsteps' : every pair runs Tsteps steps (map of the paper)
% mode = 'commonT'    : every pair runs to the same physical time T
%
% Domain: [0, 2pi] x [0, 2pi]
% =========================================================
clc;
clear all;

%% ---------------- Settings ----------------
mode   = 'fixedsteps';  % 'fixedsteps' or 'commonT'
N      = 128;            % grid size (paper map: 64; sensitivity: 128, 256)
eps    = 0.1;           % paper: 0.1; sensitivity: 0.05
Tsteps = 1000;          % steps per run      (mode = 'fixedsteps')
T      = 20;            % common final time  (mode = 'commonT')
seed   = 42;            % sensitivity: repeat with several seeds
m      = 0;             % mean composition (paper map: 0; check: 0.25)
tolRel = 1e-12;         % relative tolerance in the stepwise energy test

% Sensitivity subset near the boundary
%na  = 41;  ndt = 6;
%a_values  = linspace(0.6, 1.0, na);
%dt_values = [0.05 0.1 0.5 1 2 5];

% Full paper map: use instead
na = 200; ndt = 200;
a_values  = linspace(0, 2, na);
dt_values = linspace(0.01, 5, ndt);

M = na*ndt;
fprintf('\n==============================================\n');
fprintf('  CH CS Stability Map (%s)\n', mode);
fprintf('  N=%d, eps=%.3f, seed=%d, mean=%.2f\n', N, eps, seed, m);
fprintf('  a:  %d values in [%.3f, %.3f]\n', na, a_values(1), a_values(end));
fprintf('  dt: %d values in [%.3f, %.3f]\n', ndt, dt_values(1), dt_values(end));
fprintf('  Total simulations: %d\n', M);
fprintf('==============================================\n');

%% ---------------- Grid and operators ----------------
L   = 2*pi;  dx = L/N;
k1d = [0:N/2, -N/2+1:-1];
[KX, KY] = meshgrid(k1d, k1d);
K2  = gpuArray(KX.^2 + KY.^2);
K4  = K2.^2;

[A_grid, DT_grid] = meshgrid(a_values, dt_values);   % ndt x na
a3  = gpuArray(reshape(A_grid(:),  1,1,M));          % 1 x 1 x M (implicit expansion)
dt3 = gpuArray(reshape(DT_grid(:), 1,1,M));
denom = 1 + dt3.*eps^2.*K4 + dt3.*a3.*K2;            % N x N x M

%% ---------------- Initial condition (same for all pairs) ----------------
rng(seed);
u0 = m + 0.01*(2*rand(N,N)-1);       % uniform on [-1,1], as in the paper
u0 = u0 - mean(u0(:)) + m;           % exact mean m
u  = repmat(gpuArray(u0), 1,1,M);
uh = fft2(u);

Efun = @(u,uh) eps^2/2*sum(sum(K2.*abs(uh).^2,1),2)/N^2*dx^2 ...
             + sum(sum(u.^4/4 - u.^2/2,1),2)*dx^2;            % 1 x 1 x M
E      = Efun(u,uh);
mass0  = sum(sum(u,1),2)*dx^2;

%% ---------------- Number of steps per pair ----------------
switch mode
    case 'fixedsteps', nst = Tsteps*ones(1,1,M,'gpuArray');
    case 'commonT',    nst = round(T./dt3);
end
maxSteps = gather(max(nst(:)));
fprintf('Max steps: %d\n', maxSteps);

stable   = true(1,1,M,'gpuArray');
dEmax    = -inf(1,1,M,'gpuArray');    % largest relative stepwise change
maxU     = max(max(abs(u),[],1),[],2);

%% ---------------- Time stepping ----------------
tic;
for step = 1:maxSteps
    active = (step <= nst);                              % pairs still running
    NLh    = fft2(u.^3 - (1 + a3).*u);
    uh_new = (uh - dt3.*K2.*NLh) ./ denom;
    uh     = active.*uh_new + (~active).*uh;
    u      = real(ifft2(uh));

    Enew  = Efun(u,uh);
    dErel = (Enew - E)./max(1,abs(E));
    dErel(~active) = -inf;
    dEmax  = max(dEmax, dErel);
    bad    = (dErel > tolRel) | ~isfinite(Enew);
    stable = stable & ~bad;
    maxU   = max(maxU, max(max(abs(u),[],1),[],2));
    E      = Enew;
end
wait(gpuDevice);
t_gpu = toc;

%% ---------------- Results ----------------
stable  = gather(stable(:));
dEmax   = gather(dEmax(:));
maxU    = gather(maxU(:));
massErr = gather(abs(sum(sum(u,1),2)*dx^2 - mass0));
fin     = isfinite(maxU);

fprintf('\nGPU time: %.2f s\n', t_gpu);
fprintf('Stable:   %d / %d (%.1f%%)\n', sum(stable), M, 100*mean(stable));
fprintf('Nonfinite runs (counted unstable): %d\n', sum(~fin));
fprintf('Max |U| over stable runs: %.4f\n', max(maxU(stable)));
fprintf('Largest relative stepwise dE among stable runs: %.3e\n', max(dEmax(stable)));
fprintf('Max mass drift over finite runs: %.3e\n', max(massErr(fin)));

% Practical boundary: smallest stable a for each dt (above which all a are stable)
stable_map = reshape(stable, ndt, na);
fprintf('\n   dt      a_c (smallest a with all larger a stable)\n');
for j = 1:ndt
    row = stable_map(j,:);
    ic  = find(~row, 1, 'last');
    if isempty(ic), ac = a_values(1); elseif ic == na, ac = NaN; else, ac = a_values(ic+1); end
    if ndt <= 20 || mod(j, 20) == 1
        fprintf('  %6.3f   %.4f\n', dt_values(j), ac);
    end
end

%% ---------------- Size of the largest stepwise energy increase ----------------
dEmap = reshape(dEmax, ndt, na);
for dtq = [0.5 1]
    j = find(abs(dt_values - dtq) < 1e-12);
    if isempty(j), continue; end
    fprintf('\n dt = %.2f\n   a       max relative dE\n', dtq);
    for i = 1:5:na
        fprintf('  %.3f   % .3e\n', a_values(i), dEmap(j,i));
    end
end

%% ---------------- Plot ----------------
figure('Position', [100 100 800 600]);
cmap = [0.4 0.0 0.6; 1.0 0.9 0.0];
pcolor(dt_values, a_values, double(stable_map')); shading flat;
colormap(cmap); caxis([0 1]);
colorbar('Ticks', [0.25 0.75], 'TickLabels', {'Unstable', 'Stable'}, 'FontSize', 13);
xlabel('\Deltat', 'FontSize', 14); ylabel('a', 'FontSize', 14);
title(sprintf('CS stability map (%s), %d runs, N=%d, \\epsilon=%.2f', mode, M, N, eps), 'FontSize', 14);
hold on;
dtc = linspace(dt_values(1), dt_values(end), 400);
plot(dtc, max(0, 1 - eps*sqrt(2./dtc)), 'k-', 'LineWidth', 2);   % Theorem 2 curve
yline(1, 'k--', 'LineWidth', 1.5);                                % Theorem 1
hold off; box on; set(gca, 'FontSize', 12);
saveas(gcf, sprintf('ch_cs_map_%s_N%d_eps%g_seed%d_m%g.png', mode, N, eps, seed, m));
