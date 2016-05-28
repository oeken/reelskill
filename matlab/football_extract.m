clc;
clear;
close all;

filename = '../data/basketball/nba_2014_2015.csv';
A = importdata(filename);
% A = A.textdata;
% teams = A(2:end,[3,4]);
% results = A(2:end,[5,6]);
teams = A.textdata;
results = A.data;
% results = str2double(results);

output = (results(:,1) > results(:,2)) *2 - 1;
output(find(results(:,1) == results(:,2))) = 0;
output = num2cell(output);
whole = [teams, output];

cell2csv('nba.csv', whole)