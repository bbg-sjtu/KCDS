filename='3D_Cshape_bottom.mat';
load(filename);

figure;
hold on;
for i=1:6
    traj=data{i};
    plot3(traj(1,:),traj(2,:),traj(3,:));
end