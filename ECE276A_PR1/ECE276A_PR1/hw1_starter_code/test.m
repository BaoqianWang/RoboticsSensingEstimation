clc;
clear;

rawImage=imread('./trainset/2.jpg');
maskImage=imread('./maskset/mask2.jpg');

j=0;
for i=1:2316*3460
    if maskImage(i) >=240 
    j=j+1;
    end
end