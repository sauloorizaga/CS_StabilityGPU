%% 3D Cahn-Hilliard CS Scheme - AML Paper
%  Parameters from analytical neutral curve (eq. 8):
%  a = 0.05, dt = 0.01, eps = 0.1
clc; clear all; close all;

%% Parameters
N      = 128;
dt     = 0.01;
a_cs   = 0.05;       % from green formula: a_neutral(0.01) ~ 0.038
eps    = 0.1;
eps2   = eps^2;
a_dom  = 0;
b_dom  = 2*pi;
h      = (b_dom - a_dom)/N;

%% Grid and wavenumbers
x  = a_dom : h : b_dom - h;
kx = [0:N/2, -N/2+1:-1];
[KX, KY, KZ] = meshgrid(kx, kx, kx);
k2 = KX.^2 + KY.^2 + KZ.^2;
k4 = k2.^2;

%% Move to GPU
k2 = gpuArray(k2);
k4 = gpuArray(k4);

%% Implicit LHS
lhs = 1 + dt*(eps2*k4 + a_cs*k2);

%% Initial condition
rng(42);
U = gpuArray(0.01*(2*rand(N,N,N,'gpuArray')-1));

%% Snapshot times
snap_times = [1, 50];
snap_steps = round(snap_times/dt);
snapshots  = cell(length(snap_times),1);
snap_count = 1;
nsteps     = max(snap_steps);

hat_U = fftn(U);

%% Main loop
tic;
for it = 1:nsteps
    fU    = U.*U.*U - (1 + a_cs)*U;
    hat_U = (hat_U + dt*(-k2.*fftn(fU))) ./ lhs;
    U     = real(ifftn(hat_U));

    if snap_count <= length(snap_times) && it == snap_steps(snap_count)
        snapshots{snap_count} = gather(U);
        fprintf('Saved snapshot at t = %g\n', snap_times(snap_count));
        snap_count = snap_count + 1;
    end
end
fprintf('Done in %.1f seconds\n', toc);

%% Plotting
[X,Y,Z] = meshgrid(x, x, x);
titles_t = {'t = 1', 't = 50'};

for s = 1:2
    Uplot = snapshots{s};

    %% Isosurface figure
    figure(500 + s);
    isosurface(X, Y, Z, Uplot, -.15);
    isosurface(X, Y, Z, Uplot, -.05);
    isosurface(X, Y, Z, Uplot,  .05);
    isosurface(X, Y, Z, Uplot,  .15);
    colormap(jet)        %
    ax = gca;
    ax.FontSize = 14;
    camlight; lighting phong
    axis([a_dom b_dom a_dom b_dom a_dom b_dom])
    title(['CS Scheme, N=', num2str(N), ', ', titles_t{s}, ...
           ',  a=', num2str(a_cs)], 'FontSize', 14)

    %% Slice/box figure
    figure(600 + s);
    xslice = [0];
    yslice = [0];
    zslice = [2*pi-h];
    slice(X, Y, Z, Uplot, xslice, yslice, zslice);
    shading interp;
    colormap(jet)        %
    ax = gca;
    ax.FontSize = 14;
    camlight; lighting phong
    axis([a_dom b_dom a_dom b_dom a_dom b_dom])
    title(['CS Scheme, N=', num2str(N), ', ', titles_t{s}, ...
           ',  a=', num2str(a_cs)], 'FontSize', 14)

    %% Save figures
    print(figure(500+s), ['CH3D_iso_t' num2str(snap_times(s))], '-dpng', '-r300');
    print(figure(600+s), ['CH3D_box_t' num2str(snap_times(s))], '-dpng', '-r300');
end

fprintf('Max U = %.4f\n', max(snapshots{end}(:)))
fprintf('Min U = %.4f\n', min(snapshots{end}(:)))
