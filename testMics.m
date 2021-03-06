clc; clear; close all; 


Room.room = [10, 6, 3];


a = 1; d = 2; t = atand(Room.room(2)/Room.room(1)); b = 0.02;


Room.source(1,:) = [a*cosd(t), a*sind(t), 0.5];

Room.mic_pos(1,:)  = [(a+d)*cosd(t)+b*sind(t), ((a+d)*sind(t)-b*cosd(t)), 0.5];
Room.mic_pos(2,:)  = [(a+d)*cosd(t), (a+d)*sind(t), 0.5];
Room.mic_pos(3,:)  = [(a+d)*cosd(t)-b*sind(t), ((a+d)*sind(t)+b*cosd(t)), 0.5];
Room.mic_pos(4,:)  = [(a+d-b)*cosd(t), (a+d-b)*sind(t), 0.5];
Room.mic_pos(5,:)  = [(a+d+b)*cosd(t), (a+d+b)*sind(t), 0.5];


plotRoom(Room)


function plotRoom(Room)

l = Room.room(1); b = Room.room(2);
figure;
plot([0,l,l,0,0],[0,0,b,b,0], 'Linewidth',2); hold on;
xlim([-0.5,l+0.5]); ylim([-0.5,b+0.5]);
for k = 1:size(Room.mic_pos,1)
    plot(Room.mic_pos(k,1), Room.mic_pos(k,2), 'kx','Linewidth',4);
end

for k = 1:size(Room.source,1)
    plot(Room.source(k,1), Room.source(k,2), 'bo','Linewidth',2);
end

plot([0, Room.room(1)], [0,Room.room(2)], 'r--','Linewidth',1);
plot([0, Room.room(2)], [Room.room(1),0], 'r--','Linewidth',1);
for i = 1:5
    for j = 1:5
        d(i,j) = norm(Room.mic_pos(i,:) - Room.mic_pos(j,:));
    end
end
disp(d);
    

end
