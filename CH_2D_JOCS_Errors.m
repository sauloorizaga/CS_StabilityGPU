%% CH_2D_JOCS_Errors.m
%  Accuracy study of the convex splitting (CS) scheme for the 2D Cahn-Hilliard
%  equation: effect of the splitting parameter a on the time-discretization error.
%
%  PART 0  (optional) warm-up: fixed random data evolved to t = 0.5 -> smooth U0
%  PART 1  builds a reference solution at tf = 1 with dt_ref = 1e-5 (GPU)
%          and saves it to CH2D_reference.mat
%  PART 2  runs the CS scheme for several a and dt, computes the L2 error
%          against the reference, and saves CH2D_errors.mat
%  PART 2b (NEW) checks for the JOCS revision: seed, max|U_ref|, reference with
%          dt_ref/2, and residual of the exact discrete energy identity
%  PART 3  makes the figures (initial condition, reference solution, errors)
%
%  Switch parts on/off with the flags below. If a part is off, its results
%  are loaded from the .mat file saved in a previous run.
clc; clear; close all;

RUN_REFERENCE = false;    % false -> load CH2D_reference.mat
RUN_SWEEP     = true;    % false -> load CH2D_errors.mat
RUN_CHECKS    = true;    % NEW: PART 2b
MAKE_PLOTS    = true;

%% Parameters
N       = 256;
epsilon = 0.1;
eps2    = epsilon^2;
Tfinal  = 1;

dt_ref  = 1e-5;          % reference time step (100,000 steps)
a_ref   = 0;             % reference splitting parameter; stable by Theorem 2,
                         % since dt_ref <= 2*epsilon^2 = 0.02
a_list  = [4 3 2 1 0.5 0.25 .1 0];   % NEW: a = 0 added
dt_list = [4e-3 2e-3 1e-3 5e-4 2.5e-4];

%% Domain [0, 2*pi]^2 and wavenumbers
L  = 2*pi;
h  = L/N;
x  = h*(0:N-1);
[X,Y] = meshgrid(x,x);
kx = [0:N/2, -N/2+1:-1];
[KX, KY] = meshgrid(kx, kx);
k2 = gpuArray(KX.^2 + KY.^2);
k4 = k2.^2;

%% Initial condition
% USE_WARMUP = true : start from fixed random data, evolve to t = T_warmup, and use
%                     that (already smooth) state as the initial condition U0.
% USE_WARMUP = false: use a smooth analytic initial condition instead.
USE_WARMUP = true;
T_warmup   = 0.5;
dt_warmup  = 1e-3;       % only generates U0; its accuracy does not affect the study
if USE_WARMUP
    rng(42);
    Urand = 0.01*(2*rand(N,N)-1) + 0.25;
    U0 = cs_solve(Urand, 2, dt_warmup, T_warmup, eps2, k2, k4, h, 0);
    fprintf('Warm-up done: U0 = state at t = %g from random data\n', T_warmup);
else
    U0 = 0.25 + 0.4*cos(3*X).*cos(2*Y) + 0.2*sin(X+Y);
end

%% PART 1: reference solution
if RUN_REFERENCE
    fprintf('Reference: a = %g, dt = %g, %d steps ...\n', ...
            a_ref, dt_ref, round(Tfinal/dt_ref));
    tic;
    [Uref, Eref, tE] = cs_solve(U0, a_ref, dt_ref, Tfinal, eps2, k2, k4, h, 1000);
    fprintf('  done in %.1f s\n', toc);
    fprintf('  energy nonincreasing (every 1000 steps): %d\n', all(diff(Eref) <= 1e-12));
    save('CH2D_reference.mat', 'U0', 'Uref', 'Eref', 'tE', 'N', 'epsilon', ...
         'Tfinal', 'dt_ref', 'a_ref');
else
    load('CH2D_reference.mat', 'U0', 'Uref', 'Eref', 'tE');   % overrides U0 above
    fprintf('Loaded CH2D_reference.mat\n');
end

%% PART 2: error sweep over a and dt
if RUN_SWEEP
    err = zeros(numel(a_list), numel(dt_list));
    for i = 1:numel(a_list)
        for j = 1:numel(dt_list)
            U = cs_solve(U0, a_list(i), dt_list(j), Tfinal, eps2, k2, k4, h, 0);
            err(i,j) = sqrt(h^2*sum((U - Uref).^2, 'all'));    % discrete L2 error
        end
        rates = log2(err(i,1:end-1)./err(i,2:end));
        fprintf('a = %5.2f  errors: %s  rates: %s\n', a_list(i), ...
                sprintf('%9.2e', err(i,:)), sprintf('%6.2f', rates));
    end
    save('CH2D_errors.mat', 'a_list', 'dt_list', 'err', 'Tfinal', 'epsilon');
else
    load('CH2D_errors.mat', 'a_list', 'dt_list', 'err');
    fprintf('Loaded CH2D_errors.mat\n');
end

%% NEW: PART 2b: checks for the JOCS revision
if RUN_CHECKS
    fprintf('\n--- Checks (JOCS) ---\n');                                          % NEW
    fprintf('Random data: rng(42), MATLAB default generator (Mersenne Twister)\n'); % NEW
    fprintf('max|U_ref| = %.4f\n', max(abs(Uref(:))));                             % NEW

    % NEW: reference recomputed with dt_ref/2
    tic;                                                                            % NEW
    Uref2 = cs_solve(U0, a_ref, dt_ref/2, Tfinal, eps2, k2, k4, h, 0);              % NEW
    fprintf('||U_ref(dt_ref) - U_ref(dt_ref/2)||_L2 = %.3e   (%.1f s)\n', ...      % NEW
            sqrt(h^2*sum((Uref - Uref2).^2, 'all')), toc);                          % NEW

    % NEW: residual of the exact discrete energy identity (Lemma 1), every step
    a_c = 0.1;  dt_c = 1e-3;  nst = round(Tfinal/dt_c);                             % NEW
    U     = gpuArray(U0);  hat_U = fft2(U);                                         % NEW
    lhs   = 1 + dt_c*(eps2*k4 + a_c*k2);                                            % NEW
    k2inv = 1./k2;  k2inv(1,1) = 0;                                                 % NEW
    F     = @(u) 0.25*(u.^2 - 1).^2;                                                % NEW
    Fp    = @(u) u.^3 - u;                                                          % NEW
    Eh    = @(U, hU) h^2*(0.5*eps2*sum(k2.*abs(hU).^2, 'all')/N^2 + sum(F(U), 'all'));  % NEW
    res_max = 0;  dE_max = 0;                                                       % NEW
    for it = 1:nst                                                                  % NEW
        hat_new = (hat_U - dt_c*k2.*fft2(U.^3 - (1 + a_c)*U))./lhs;                 % NEW
        Unew    = real(ifft2(hat_new));                                             % NEW
        dU      = Unew - U;   hat_dU = hat_new - hat_U;                             % NEW
        dE      = Eh(Unew, hat_new) - Eh(U, hat_U);                                 % NEW
        rhs     = - h^2/N^2*sum(k2inv.*abs(hat_dU).^2, 'all')/dt_c ...              % NEW
                  - 0.5*eps2*h^2/N^2*sum(k2.*abs(hat_dU).^2, 'all') ...             % NEW
                  - a_c*h^2*sum(dU.^2, 'all') ...                                   % NEW
                  + h^2*sum(F(Unew) - F(U) - Fp(U).*dU, 'all');                     % NEW
        res_max = max(res_max, gather(abs(dE - rhs)));                              % NEW
        dE_max  = max(dE_max,  gather(abs(dE)));                                    % NEW
        U = Unew;  hat_U = hat_new;                                                 % NEW
    end                                                                             % NEW
    fprintf('Energy identity (a = %g, dt = %g, %d steps):\n', a_c, dt_c, nst);      % NEW
    fprintf('   max |residual| = %.3e,   max |E^{n+1}-E^n| = %.3e\n', res_max, dE_max);  % NEW
end

%% PART 3: figures
if MAKE_PLOTS
    % Figure 1: initial condition
    figure(1); clf
    pcolor(X, Y, U0); shading interp; colormap(jet); colorbar
    axis equal tight; set(gca, 'FontSize', 13)
    title('Initial condition, $t=0$', 'Interpreter', 'latex', 'FontSize', 15)
    exportgraphics(gcf, 'CH2D_IC.png', 'Resolution', 300)

    % Figure 2: reference solution at tf
    figure(2); clf
    pcolor(X, Y, Uref); shading interp; colormap(jet); colorbar
    axis equal tight; set(gca, 'FontSize', 13)
    title(sprintf('Reference solution, $t=%g$', Tfinal), ...
          'Interpreter', 'latex', 'FontSize', 15)
    exportgraphics(gcf, 'CH2D_reference.png', 'Resolution', 300)

    % Figure 3: error versus dt for each a
    figure(3); clf
    cols = lines(numel(a_list));
    mk   = {'o','s','d','^','v','>','<','p'};    % NEW: marker for a = 0
    for i = 1:numel(a_list)
        loglog(dt_list, err(i,:), ['-' mk{i}], 'Color', cols(i,:), ...
               'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', cols(i,:)); hold on
    end
    ref1 = err(end,end)*(dt_list/dt_list(end));               % slope-1 guide
    loglog(dt_list, ref1/2, 'k--', 'LineWidth', 1.5); hold off
    grid on; set(gca, 'FontSize', 13)
    xlabel('$\Delta t$', 'Interpreter', 'latex', 'FontSize', 16)
    ylabel('$\|U^n-u_{\mathrm{ref}}\|_{L^2}$ at $t=1$', 'Interpreter', 'latex', 'FontSize', 16)
    leg = [arrayfun(@(a) sprintf('$a=%g$', a), a_list, 'UniformOutput', false), {'slope 1'}];
    legend(leg, 'Interpreter', 'latex', 'Location', 'northwest', 'FontSize', 12)
    exportgraphics(gcf, 'CH2D_errors_dt.png', 'Resolution', 300)

    % Figure 4: error versus a at a fixed dt
    jfix = find(dt_list == 1e-3, 1);
    figure(4); clf
    plot(a_list, err(:,jfix), 'bo-', 'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', 'b')
    grid on; set(gca, 'FontSize', 13)
    xlabel('$a$', 'Interpreter', 'latex', 'FontSize', 16)
    ylabel(sprintf('$L^2$ error at $t=1$, $\\Delta t=%g$', dt_list(jfix)), ...
           'Interpreter', 'latex', 'FontSize', 16)
    exportgraphics(gcf, 'CH2D_errors_a.png', 'Resolution', 300)

    % Optional: reference energy
    figure(5); clf
    plot(tE, Eref, 'b-', 'LineWidth', 2); grid on; set(gca, 'FontSize', 13)
    xlabel('$t$', 'Interpreter', 'latex', 'FontSize', 16)
    ylabel('$E(U)$', 'Interpreter', 'latex', 'FontSize', 16)
    title('Reference run energy', 'Interpreter', 'latex', 'FontSize', 15)
end

%% ------------------------------------------------------------------------
function [U, E, tE] = cs_solve(U0, a, dt, Tf, eps2, k2, k4, h, every)
% CS scheme for 2D CH on the GPU. If every > 0, the free energy is recorded
% every 'every' steps (computed spectrally, without extra FFTs).
    nsteps = round(Tf/dt);
    U      = gpuArray(U0);
    hat_U  = fft2(U);
    lhs    = 1 + dt*(eps2*k4 + a*k2);
    N      = size(U0,1);
    energy = @(U, hat_U) gather(h^2*(0.5*eps2*sum(k2.*abs(hat_U).^2, 'all')/N^2 ...
                                + sum(0.25*(U.^2 - 1).^2, 'all')));   % Parseval
    E = []; tE = [];
    if every > 0
        nrec = floor(nsteps/every) + 1;
        E = zeros(1, nrec); tE = zeros(1, nrec);
        E(1) = energy(U, hat_U); tE(1) = 0; r = 1;
    end
    for it = 1:nsteps
        hat_U = (hat_U - dt*k2.*fft2(U.^3 - (1 + a)*U))./lhs;
        U     = real(ifft2(hat_U));
        if every > 0 && mod(it, every) == 0
            r = r + 1; E(r) = energy(U, hat_U); tE(r) = it*dt;
        end
    end
    U = gather(U);
end