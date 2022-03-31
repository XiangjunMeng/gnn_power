clear;
close all;
clc;

addpath(genpath('/home/hugo/source/matpower7.1'));

%casefile = 'case12da';
casefile = 'case38si';

[PQ, PV, REF, NONE, BUS_I, BUS_TYPE, PD, QD, GS, BS, BUS_AREA, VM, ...
    VA, BASE_KV, ZONE, VMAX, VMIN, LAM_P, LAM_Q, MU_VMAX, MU_VMIN] = idx_bus;
[F_BUS, T_BUS, BR_R, BR_X, BR_B, RATE_A, RATE_B, RATE_C, ...
    TAP, SHIFT, BR_STATUS, PF, QF, PT, QT, MU_SF, MU_ST, ...
    ANGMIN, ANGMAX, MU_ANGMIN, MU_ANGMAX] = idx_brch;
[GEN_BUS, PG, QG, QMAX, QMIN, VG, MBASE, GEN_STATUS, PMAX, PMIN, ...
    MU_PMAX, MU_PMIN, MU_QMAX, MU_QMIN, PC1, PC2, QC1MIN, QC1MAX, ...
    QC2MIN, QC2MAX, RAMP_AGC, RAMP_10, RAMP_30, RAMP_Q, APF] = idx_gen;
[CT_LABEL, CT_PROB, CT_TABLE, CT_TBUS, CT_TGEN, CT_TBRCH, CT_TAREABUS, ...
   CT_TAREAGEN, CT_TAREABRCH, CT_ROW, CT_COL, CT_CHGTYPE, CT_REP, ...
    CT_REL, CT_ADD, CT_NEWVAL, CT_TLOAD, CT_TAREALOAD, CT_LOAD_ALL_PQ, ...
    CT_LOAD_FIX_PQ, CT_LOAD_DIS_PQ, CT_LOAD_ALL_P, CT_LOAD_FIX_P, ...
    CT_LOAD_DIS_P, CT_TGENCOST, CT_TAREAGENCOST, CT_MODCOST_F, ...
    CT_MODCOST_X] = idx_ct;


BUS_DEBUG = [BUS_I, BUS_TYPE, PD, QD];
BRCH_DEBUG = [F_BUS, T_BUS, PF, QF, PT, QT];
GEN_DEBUG = [GEN_BUS, PG, QG];


SAVERESULT = 1;
DEBUG = 0;

seed = 1;

rng(seed, 'simdTwister');

%sample = 2;
sample = 5000;
delta_low = -10;
delta_high = 5;
%delta_low = 0; % pn 10%
%delta_high = 50;
accuracy = 3; % 0.01

mpccase = loadcase(casefile);

pqi = find(mpccase.bus(:, BUS_TYPE) == PQ);
npq = length(mpccase.bus((mpccase.bus(:, BUS_TYPE) == PQ), BUS_I));
default_row = [1, 1, CT_TBUS, 1, PD, CT_REL, 1];
chgtab = repmat(default_row, npq * 2 * sample, 1);

for loop0 = 1 : sample
    % fprintf('%d  \n', loop0);

    starti = (loop0 - 1) * 2* npq + 1;
    mpccase = loadcase(casefile);
    nbus = length( mpccase.bus(:, 1) );
    nline = length( mpccase.branch(:,1) );
    ngen = length(mpccase.gen(:,1));
    ndev = nline + nbus;

    load_change1 = randi([delta_low * 10^accuracy, delta_high * 10^accuracy], npq, 1) / (100 * 10^accuracy);
    load_change2 = randi([delta_low * 10^accuracy, delta_high * 10^accuracy], npq, 1) / (100 * 10^accuracy);
    chgtab([starti : starti + 2*npq - 1], 1) = loop0;
    chgtab([starti : starti + npq - 1], 4) = pqi;
    chgtab([starti : starti + npq - 1], 7) = load_change1 + 1;
    chgtab([starti  + npq : starti + 2*npq - 1], 4) = pqi;
    chgtab([starti  + npq : starti + 2*npq - 1], 5) = QD;
    chgtab([starti + npq : starti + 2*npq - 1], 7) = load_change2 + 1;


    
end


if (SAVERESULT == 1)
    %chgtab
    filename = sprintf('./result/%s_scenario.mat', casefile);
    %save(filename, 'chgtab');
    save(filename);
end
