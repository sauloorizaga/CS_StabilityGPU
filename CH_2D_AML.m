%% 2D Cahn-Hilliard CS Scheme - AML Paper Simulation
%  Parameters chosen from analytical neutral curve (eq. 8)
%  a = 0.05, dt = 0.01, eps = 0.1 -> a_neutral(0.01) ~ 0.038
clc; clear all; close all;

%% Parameters
N       = 256;
epsilon = 0.1;
eps2    = epsilon^2;
dt      = 0.01;
a_cs    = 0.05;       % from green formula: a_neutral(0.01) ~ 0.038
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
U = 0.01*(2*rand(N,N)-1);

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
energy_vec = zeros(1, nsteps);
time_vec   = zeros(1, nsteps);

hat_U = fft2(U);

%% Main time loop
for it = 1:nsteps
    % Nonlinear term
    fU = U.*U.*U - (1 + a_cs)*U;

    % CS update in Fourier space
    hat_rhs = hat_U + dt*(-k2.*fft2(fU));
    hat_U   = hat_rhs./lhs;
    U       = real(ifft2(hat_U));

    t = it*dt;

    % Energy calculation
    [Ux, Uy]       = gradient(gather(U), h, h);
    energy_vec(it) = h^2*sum(sum(eps2/2*(Ux.^2+Uy.^2) + ...
                     (1/4)*gather(U).^4 - (1/2)*gather(U).^2));
    time_vec(it)   = t;

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
axis([time_vec(1) time_vec(end) energy_vec(end)*1.05 energy_vec(1)*0.95])
set(gca, 'FontSize', 13)
grid on

fprintf('Final energy: %.4f\n', energy_vec(end))
fprintf('Energy dissipated: %.4f\n', energy_vec(1) - energy_vec(end))
fprintf('Energy monotonically decreasing: %d\n', all(diff(energy_vec) <= 0))
