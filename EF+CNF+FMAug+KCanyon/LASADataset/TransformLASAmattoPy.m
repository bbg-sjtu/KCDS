clear;

% filename = "GShape";
% filename = "Angle";
% filename = "CShape";
% filename = "DoubleBendedLine";
% filename = "BendedLine";
% filename = "JShape";
% filename = "Khamesh";
% filename = "LShape";
% filename = "Leaf_1";
% filename = "Leaf_2";
% filename = "Line";
% filename = "Multi_Models_1";
% filename = "Multi_Models_2";
% filename = "Multi_Models_3";
filename = "Multi_Models_4";
% filename = "NShape";
% filename = "PShape";
% filename = "RShape";
% filename = "Saeghe";
% filename = "Sharpc";
% filename = "Sine";
% filename = "Snake";
% filename = "Spoon";
% filename = "Sshape";
% filename = "Trapezoid";
% filename = "WShape";
% filename = "Worm";
% filename = "Zshape";
% filename = "heee";
load(filename + ".mat");

dt = zeros(size(demos, 2), 1);
t = zeros(1000, size(demos, 2));
pos = zeros(1000, 2, size(demos, 2));
vel = zeros(1000, 2, size(demos, 2));
acc = zeros(1000, 2, size(demos, 2));
for i = 1: size(demos, 2)
    dt(i) = demos{i}.dt;
    t(:, i) = demos{i}.t';
    pos(:, :, i) = demos{i}.pos';
    vel(:, :, i) = demos{i}.vel';
    acc(:, :, i) = demos{i}.acc';
end

save(filename + "Py.mat", 'dt', 't', 'pos', 'vel', 'acc');