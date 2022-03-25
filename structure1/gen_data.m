clear;
close all;
clc;

addpath(genpath('/home/hugo/source/matpower7.1'));

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

casefile = 'case12da';

SAVERESULT = 1;
DEBUG = 0;

seed = 1;

rng(seed, 'simdTwister');

filename = sprintf('../acc/result/%s_scenario.mat', casefile);
load(filename);  % load chgtab

%sample = 2;
%delta_low = -10
%delta_high = 5;
%delta_low = 0; % pn 10%
%delta_high = 50;
%accuracy = 3; % 0.01

mpccase = loadcase(casefile);

pqi = find(mpccase.bus(:, BUS_TYPE) == PQ);
npq = length(mpccase.bus((mpccase.bus(:, BUS_TYPE) == PQ), BUS_I));
nbus = length( mpccase.bus(:, 1) );
nline = length( mpccase.branch(:,1) );
ngen = length(mpccase.gen(:,1));
geni = mpccase.gen(:, GEN_BUS );
ndev = nline + nbus;
%opfopt = mpoption('opf.ac.solver', 'MIPS', 'out.all', 0, 'verbose', 0);
pfopt = mpoption('pf.alg', 'NR', 'out.all', 0, 'verbose', 0);


% make sure that bus index are numbered in order
assert( sum(transpose([1:nbus]) == mpccase.bus(:, BUS_I)) == nbus );

assert( length(chgtab(:,1)) == npq * 2 * sample );

% graph parameters;
num_features = 2;
out_features = 2;
data.x = zeros(nbus, num_features);
data.y = zeros(nbus, out_features);
% formulate edge_index first
data.edge_index = transpose(mpccase.branch(:, [F_BUS, T_BUS]) - 1);
data.edge_attr = mpccase.branch(:, [BR_R BR_X]);

dataset = cell(sample, 1);
suc_i = 1;

for loop0 = 1 : sample
    fprintf('%d  \n', loop0);

    starti = (loop0 - 1) * 2* npq + 1;
    mpc_modify = apply_changes(loop0, mpccase, chgtab);

    result = runpf(mpc_modify, pfopt);
    %result = runopf(mpc_modify, opfopt);
    %if result.success == 0
    %    result = runpf(mpc_modify, pfopt);
    %end
    if result.success == 1
        pinject = result.bus(:, PD);
        %pinject(geni) = pinject(geni) + result.gen(:, PG);
        pinject = pinject / result.baseMVA;
        qinject = result.bus(:, QD);
        %qinject(geni) = qinject(geni) + result.gen(:, QG);
        qinject = qinject / result.baseMVA;
        slack_bus = (result.bus(:, BUS_TYPE) == REF);
        data.x = [pinject, qinject];
        %data.x(slack_bus, :) = 0;
        data.y(geni, :) = result.gen(:, [PG,QG]);
        %data.y(slack_bus, :) = [pinject(slack_bus), qinject(slack_bus)];
        dataset{suc_i} = data;
        suc_i = suc_i + 1;
    end
end

if suc_i < sample
    dataset = dataset{1:suc_i};
end


if (SAVERESULT == 1)
    %dataset
    filename = sprintf('./result/%s_data.mat', casefile);
    save(filename);
end
