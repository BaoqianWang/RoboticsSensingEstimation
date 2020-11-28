clc;
clear;

for i=1:200
imageName=sprintf("./trainset/%d.jpg",i);
if exist(imageName, 'file') == 2
    imageName=sprintf('./trainset/%d.jpg',i);
else
    imageName=sprintf('./trainset/%d.png',i);
end
if ~ (exist(imageName, 'file') == 2)
   continue; 
end
    
imshow(imageName);
e=impoly; %Use impoly to create polygon shape
BW=createMask(e); %Create the mask for the polygon shape
figure;
imshow(BW);
maskName=sprintf('./maskset/mask%d.jpg',i);
imwrite(BW, maskName);
close all;
end