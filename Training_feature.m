clear all; 
close all; 
clc;
feature_train=[];

for imc=1:35
 
%% General settings
fontSize = 14;
%%%%%%%%   stage 1 Image acquisition   %%%%%%%%%%%%%%

%% Pick the image and Load the image
%  [filename, pathname] = uigetfile( ...
%        {'*.jpg;*.tif;*.tiff;*.png;*.bmp', 'All image Files (*.jpg, *.tif, *.tiff, *.png, *.bmp)'}, ...
%         'Pick a file');
% f = fullfile(pathname, filename);
% disp('Reading image')
cd all
f=[num2str(imc) '.jpg'];

rgbImage = imread(f);
cd ..
rgbImage = imresize(rgbImage,[256 256]);
rgrayge=rgb2gray(rgbImage);
[rows, columns numberOfColorBands] = size(rgbImage);

%%%%%%%%%%%%%%%  checking this image is as gray or rgb range

% % % % %        image conversion process

if strcmpi(class(rgbImage), 'uint8')
		% Flag for 256 gray levels.
		eightBit = true;
	else
		eightBit = false;
    end
    
	if numberOfColorBands == 1
		if isempty(storedColorMap)
			rgbImage = cat(3, rgbImage, rgbImage, rgbImage);
		else
			% It's an indexed image.
			rgbImage = ind2rgb(rgbImage, storedColorMap);
			if eightBit
				rgbImage = uint8(255 * rgbImage);
			end
		end
    end         	
    
%% Display the color image 
% disp('Displaying color original image')
% % F1 = figure(1);
% %    subplot(3,4,1);
% %     imshow(rgbImage);
%     
%     if numberOfColorBands > 1 
% % 		title('Original Color Image', 'FontSize', fontSize); 
% 	else 
% 		caption = sprintf('Original Indexed Image\n(converted to true color with its stored colormap)');
% 		title(caption, 'FontSize', fontSize);
%     end
  
%% Size of the picture - to occupy the whole screen
%  scnsize = get(0,'ScreenSize'); % - - width height
%  position = get(F1,'Position'); % x-pos y-pos widht height
%  outerpos = get(F1,'OuterPosition');
%  borders = outerpos - position;
%  edge = abs(borders(1))/2;
%  pos1 = [edge,...
%         1/20*scnsize(4), ...
%         9/10*scnsize(3),...
%         9/10*scnsize(4)];
%  set(F1,'OuterPosition',pos1) 

%% Explore RGB    
% Extract out the color bands from the original image
% into 3 separate 2D arrays, one for each color component.
	redBand = rgbImage(:, :, 1); 
	greenBand = rgbImage(:, :, 2); 
	blueBand = rgbImage(:, :, 3); 
%    % Display them.
% 	subplot(3, 4, 2);
%         imshow(redBand);
%         title('Red Band', 'FontSize', fontSize);
% 	subplot(3, 4, 3);
%         imshow(greenBand);
%         title('Green Band', 'FontSize', fontSize);
% 	subplot(3, 4, 4);
%         imshow(blueBand);
%         title('Blue Band', 'FontSize', fontSize);

%% Compute and plot the red histogram. 
% 	hR = subplot(3, 4, 6); 
	[countsR, grayLevelsR] = imhist(redBand); 
	maxGLValueR = find(countsR > 0, 1, 'last'); 
	maxCountR = max(countsR); 
% 	bar(countsR, 'r'); 
% 	grid on; 
% 	xlabel('Gray Levels'); 
% 	ylabel('Pixel Count'); 
% 	title('Histogram of Red Band', 'FontSize', fontSize);

%% Compute and plot the green histogram. 
% 	hG = subplot(3, 4, 7); 
	[countsG, grayLevelsG] = imhist(greenBand); 
	maxGLValueG = find(countsG > 0, 1, 'last'); 
	maxCountG = max(countsG); 
% 	bar(countsG, 'g', 'BarWidth', 0.95); 
% 	grid on; 
% 	xlabel('Gray Levels'); 
% 	ylabel('Pixel Count'); 
% 	title('Histogram of Green Band', 'FontSize', fontSize);

%% Compute and plot the blue histogram. 
% 	hB = subplot(3, 4, 8); 
	[countsB, grayLevelsB] = imhist(blueBand); 
	maxGLValueB = find(countsB > 0, 1, 'last'); 
	maxCountB = max(countsB); 
% 	bar(countsB, 'b'); 
% 	grid on; 
% 	xlabel('Gray Levels'); 
% 	ylabel('Pixel Count'); 
% 	title('Histogram of Blue Band', 'FontSize', fontSize);

%% Set all axes to be the same width and height.
% This makes it easier to compare them.
	maxGL = max([maxGLValueR,  maxGLValueG, maxGLValueB]); 
	if eightBit 
			maxGL = 255; 
	end 
	maxCount = max([maxCountR,  maxCountG, maxCountB]); 
% 	axis([hR hG hB], [0 maxGL 0 maxCount]); 

%% Plot all 3 histograms in one plot.
% 	subplot(3, 4, 5); 
% 	plot(grayLevelsR, countsR, 'r', 'LineWidth', 2); 
% 	grid on; 
% 	xlabel('Gray Levels'); 
% 	ylabel('Pixel Count'); 
% 	hold on; 
% 	plot(grayLevelsG, countsG, 'g', 'LineWidth', 2); 
% 	plot(grayLevelsB, countsB, 'b', 'LineWidth', 2); 
% 	title('Histogram of All Bands', 'FontSize', fontSize); 
	maxGrayLevel = max([maxGLValueR, maxGLValueG, maxGLValueB]); 
	% Trim x-axis to just the max gray level on the bright end. 
	if eightBit 
		xlim([0 255]); 
	else 
		xlim([0 maxGrayLevel]); 
    end
    
redMask=im2bw(redBand,0.4);
greenMask=im2bw(greenBand,0.9);
blueMask=im2bw(blueBand,0.7);
    
%% Display the thresholded binary images.
% 	subplot(3, 4, 10);
%         imshow(redMask, []);
%         title('Red Mask', 'FontSize', fontSize);
% 	subplot(3, 4, 11);
%         imshow(greenMask, []);
%         title('Green Mask', 'FontSize', fontSize);
% 	subplot(3, 4, 12);
%         imshow(blueMask, []);
%         title('Blue Mask', 'FontSize', fontSize);
% 	    
%% Combine the masks to find where all 3 are "true."
% Then we will have the mask of only the chosen color parts of the image.
% 
ObjectsMask=uint8(redMask | greenMask | blueMask);
% 	subplot(3, 4, 9);
% 	imshow(ObjectsMask, []);
% 	caption = sprintf('Mask of the objects with chosen color');
% 	title(caption, 'FontSize', fontSize);
    
  % read an image
im = im2double(rgbImage);
  gray=rgb2gray(im);
reima=imresize(gray,[256,256]);


%%%%%%%%%% Kmeans segmenation %%%%%%%%%%%%%
[seg_im,vec_mean] = kmeans_fast_Color(reima,3);
figure();imshow(seg_im,[]);

stats = graycoprops(seg_im,'Contrast Correlation Energy Homogeneity');
Contrast = stats.Contrast;
Correlation = stats.Correlation;
Energy = stats.Energy;
Homogeneity = stats.Homogeneity;
Mean = mean2(seg_im);
Standard_Deviation = std2(seg_im);
Entropy = entropy(seg_im);
RMS = mean2(rms(seg_im));
%Skewness = skewness(img)
Variance = mean2(var(double(seg_im)));
a = sum(double(seg_im(:)));
Smoothness = 1-(1/(1+a));
Kurtosis = kurtosis(double(seg_im(:)));
Skewness = skewness(double(seg_im(:)));
% Inverse Difference Movement
m = size(seg_im,1);
n = size(seg_im,2);
in_diff = 0;
for i = 1:m
    for j = 1:n
        temp = seg_im(i,j)./(1+(i-j).^2);
        in_diff = in_diff+temp;
    end
end
IDM = double(in_diff);
    
%     feat = [Contrast,Correlation,Energy,Homogeneity, Entropy, Kurtosis, Skewness ];

 featq(imc,:) = [Contrast,Correlation,Energy,Homogeneity, Entropy, RMS, Variance,  Kurtosis, Skewness,max(countsR), max(countsG) ,max(countsB)];
end
truelabel=[ones(1,5),ones(1,5)*2,ones(1,5)*4,ones(1,5)*8,ones(1,5)*16,ones(1,5)*32,ones(1,5)*64];
truelabelN=(de2bi(truelabel,7))';
trainfea=abs(featq)';
net = newff(trainfea,truelabelN,10);
net.trainParam.epochs = 10;
[net,tr] = train(net,trainfea,truelabelN);
Y = sim(net,trainfea);
Y1=round(Y);
uy=7;
if nnz(Y1(:,uy))~=1 
    if nnz(Y1(:,uy))==0
        Y1(:,uy)=[0;0;1;0;0;0;0];
    else
        Dew=find(abs(Y1(:,uy))==1);
        Y1(:,uy)=abs(Y1(:,uy));
        if length(Dew)==2
        Y1(Dew(1),uy)=0;
        elseif length(Dew)==3
            Y1(Dew(1),uy)=0;
            Y1(Dew(2),uy)=0;
        elseif length(Dew)==4
            Y1(Dew(1),uy)=0;
            Y1(Dew(2),uy)=0;
            Y1(Dew(3),uy)=0;
              elseif length(Dew)==5
            Y1(Dew(1),uy)=0;
            Y1(Dew(2),uy)=0;
            Y1(Dew(3),uy)=0;
   elseif length(Dew)==6
            Y1(Dew(1),uy)=0;
            Y1(Dew(2),uy)=0;
            Y1(Dew(3),uy)=0;
            elseif length(Dew)==7
            Y1(Dew(1),uy)=0;
            Y1(Dew(2),uy)=0;
            Y1(Dew(3),uy)=0;
        end
    end
end
save FianlizedData
