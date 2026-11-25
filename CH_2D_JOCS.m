%% 2D Cahn-Hilliard CS Scheme - JOCS Paper Simulation
%  Validation run: a = 0.05, dt = 0.01, eps = 0.1
%  Gap run:        a = 0.7,  dt = 0.5,  eps = 0.1
clc; clear all; close all;
%% Parameters
N       = 256;
epsilon = 0.1;
eps2    = epsilon^2;
dt      = 0.5;       % gap run: dt = 0.5
a_cs    = 0.7;       % gap run: a_cs = 0.7
Tfinal  = 20;
nsteps  = round(Tfinal/dt);
%% Domain [0, 2*pi]
L  = 2*pi;
h  = L/N;
x  = h*(1:N);
[X,Y] = meshgrid(x,x);
%% Wavenumbers
kx = [0:N/2, -N/2+1:-1];
[KX, KY] = meshgrid(kx, kx);
k2 = KX.^2 + KY.^2;
k4 = k2.^2;
%% Initial condition - reproducible random
rng(42);
U = 0.01*(2*rand(N,N)-1)+.25;
%% Move to GPU
U  = gpuArray(U);
k2 = gpuArray(k2);
k4 = gpuArray(k4);
%% Implicit LHS in Fourier space
lhs = 1 + dt*(eps2*k4 + a_cs*k2);
%% Snapshot times
snap_times = [1, 5, 10, 20];
snap_idx   = round(snap_times/dt);
snapshots  = cell(length(snap_times),1);
snap_count = 1;
%% Energy storage
energy_vec = zeros(1, nsteps+1);                                % NEW (includes t = 0)
time_vec   = (0:nsteps)*dt;                                     % NEW
hat_U = fft2(U);
%% Discrete energy E_h (spectral gradient, consistent with the scheme)   % NEW
energy = @(U, hat_U) gather( h^2*eps2/2/N^2*sum(k2(:).*abs(hat_U(:)).^2) ...
    + h^2*sum((1/4)*U(:).^4 - (1/2)*U(:).^2) );   % NEW
energy_vec(1) = energy(U, hat_U);                               % NEW
mass0 = gather(sum(U(:)))*h^2;                                  % NEW
maxU  = gather(max(abs(U(:))));                                 % NEW
%% Main time loop
for it = 1:nsteps
    % Nonlinear term
    fU = U.*U.*U - (1 + a_cs)*U;
    % CS update in Fourier space
    hat_rhs = hat_U + dt*(-k2.*fft2(fU));
    hat_U   = hat_rhs./lhs;
    U       = real(ifft2(hat_U));
    % Energy calculation
    energy_vec(it+1) = energy(U, hat_U);                        % NEW
    maxU = max(maxU, gather(max(abs(U(:)))));                   % NEW
    % Save snapshots
    if snap_count <= length(snap_times) && it == snap_idx(snap_count)
        snapshots{snap_count} = gather(U);
        snap_count = snap_count + 1;
    end
end
%% Figure 1 - Phase field snapshots
figure('Position', [100 100 1200 300]);
titles = {'t = 1', 't = 5', 't = 10', 't = 20'};
for i = 1:4
    subplot(1,4,i)
    pcolor(snapshots{i}), shading interp
    colormap(jet)        %
    axis off, axis equal
    title(titles{i}, 'FontSize', 14, 'FontWeight', 'bold')
end
sgtitle(['CH CS Scheme: a = ' num2str(a_cs) ...
    ',  \Deltat = ' num2str(dt) ...
    ',  \epsilon = ' num2str(epsilon)], 'FontSize', 14)
%% Figure 2 - Energy decay
figure('Position', [100 500 700 400]);
plot(time_vec, energy_vec, 'b-', 'LineWidth', 2)
xlabel('Time', 'FontSize', 14)
ylabel('Free Energy  E(t)', 'FontSize', 14)
title('Energy dissipation — CS scheme', 'FontSize', 14)
set(gca, 'FontSize', 13)
grid on
%% Diagnostics                                                  % NEW
dE      = diff(energy_vec);                                     % NEW
massErr = abs(gather(sum(U(:)))*h^2 - mass0);                   % NEW
Lc      = 3*maxU^2 - 1;                                         % NEW
fprintf('\n=========== DIAGNOSTICS (2D) ===========\n');          % NEW
fprintf('N = %d, a = %g, dt = %g, eps = %g, Tf = %g\n', N, a_cs, dt, epsilon, Tfinal);   % NEW
fprintf('Final energy: %.4f\n', energy_vec(end))
fprintf('Energy dissipated: %.4f\n', energy_vec(1) - energy_vec(end))
fprintf('Energy monotonically decreasing: %d\n', all(dE <= 0))
fprintf('Max |U| over run         = %.4f\n', maxU);                          % NEW
fprintf('L/2 = (3B^2-1)/2         = %.4f\n', Lc/2);                          % NEW
fprintf('Thm 2 requires a >=      %.4f\n', Lc/2 - epsilon*sqrt(2/dt));       % NEW
fprintf('Max positive dE per step = %.3e\n', max(dE));                       % NEW
fprintf('Mass drift               = %.3e\n', massErr);                       % NEW
fprintf('========================================\n');                        % NEW