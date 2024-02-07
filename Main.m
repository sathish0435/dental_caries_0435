clear all; 
close all; 
clc;

%% General settings
fontSize = 14;
%%%%%%%%   stage 1 Image acquisition   %%%%%%%%%%%%%%

%% Pick the image and Load the image
 [filename, pathname] = uigetfile( ...
       {'*.jpg;*.tif;*.tiff;*.png;*.bmp', 'All image Files (*.jpg, *.tif, *.tiff, *.png, *.bmp)'}, ...
        'Pick a file');
f = fullfile(pathname, filename);
disp('Reading image')
rgbImage = imread(f);
rgbImage = imresize(rgbImage,[256 256]);
[rows columns numberOfColorBands] = size(rgbImage);

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
disp('Displaying color original image')
F1 = figure(1);
   subplot(3,4,1);
    imshow(rgbImage);
    
    if numberOfColorBands > 1 
		title('Original Color Image', 'FontSize', fontSize); 
	else 
		caption = sprintf('Original Indexed Image\n(converted to true color with its stored colormap)');
		title(caption, 'FontSize', fontSize);
    end
  
%% Size of the picture - to occupy the whole screen
 scnsize = get(0,'ScreenSize'); % - - width height
 position = get(F1,'Position'); % x-pos y-pos widht height
 outerpos = get(F1,'OuterPosition');
 borders = outerpos - position;
 edge = abs(borders(1))/2;
 pos1 = [edge,...
        1/20*scnsize(4), ...
        9/10*scnsize(3),...
        9/10*scnsize(4)];
 set(F1,'OuterPosition',pos1) 

%% Explore RGB    
% Extract out the color bands from the original image
% into 3 separate 2D arrays, one for each color component.
	redBand = rgbImage(:, :, 1); 
	greenBand = rgbImage(:, :, 2); 
	blueBand = rgbImage(:, :, 3); 
   % Display them.
	subplot(3, 4, 2);
        imshow(redBand);
        title('Red Band', 'FontSize', fontSize);
	subplot(3, 4, 3);
        imshow(greenBand);
        title('Green Band', 'FontSize', fontSize);
	subplot(3, 4, 4);
        imshow(blueBand);
        title('Blue Band', 'FontSize', fontSize);

%% Compute and plot the red histogram. 
	hR = subplot(3, 4, 6); 
	[countsR, grayLevelsR] = imhist(redBand); 
	maxGLValueR = find(countsR > 0, 1, 'last'); 
	maxCountR = max(countsR); 
	bar(countsR, 'r'); 
	grid on; 
	xlabel('Gray Levels'); 
	ylabel('Pixel Count'); 
	title('Histogram of Red Band', 'FontSize', fontSize);

%% Compute and plot the green histogram. 
	hG = subplot(3, 4, 7); 
	[countsG, grayLevelsG] = imhist(greenBand); 
	maxGLValueG = find(countsG > 0, 1, 'last'); 
	maxCountG = max(countsG); 
	bar(countsG, 'g', 'BarWidth', 0.95); 
	grid on; 
	xlabel('Gray Levels'); 
	ylabel('Pixel Count'); 
	title('Histogram of Green Band', 'FontSize', fontSize);

%% Compute and plot the blue histogram. 
	hB = subplot(3, 4, 8); 
	[countsB, grayLevelsB] = imhist(blueBand); 
	maxGLValueB = find(countsB > 0, 1, 'last'); 
	maxCountB = max(countsB); 
	bar(countsB, 'b'); 
	grid on; 
	xlabel('Gray Levels'); 
	ylabel('Pixel Count'); 
	title('Histogram of Blue Band', 'FontSize', fontSize);

%% Set all axes to be the same width and height.
% This makes it easier to compare them.
	maxGL = max([maxGLValueR,  maxGLValueG, maxGLValueB]); 
	if eightBit 
			maxGL = 255; 
	end 
	maxCount = max([maxCountR,  maxCountG, maxCountB]); 
	axis([hR hG hB], [0 maxGL 0 maxCount]); 

%% Plot all 3 histograms in one plot.
	subplot(3, 4, 5); 
	plot(grayLevelsR, countsR, 'r', 'LineWidth', 2); 
	grid on; 
	xlabel('Gray Levels'); 
	ylabel('Pixel Count'); 
	hold on; 
	plot(grayLevelsG, countsG, 'g', 'LineWidth', 2); 
	plot(grayLevelsB, countsB, 'b', 'LineWidth', 2); 
	title('Histogram of All Bands', 'FontSize', fontSize); 
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
	subplot(3, 4, 10);
        imshow(redMask, []);
        title('Red Mask', 'FontSize', fontSize);
	subplot(3, 4, 11);
        imshow(greenMask, []);
        title('Green Mask', 'FontSize', fontSize);
	subplot(3, 4, 12);
        imshow(blueMask, []);
        title('Blue Mask', 'FontSize', fontSize);
	    
%% Combine the masks to find where all 3 are "true."
% Then we will have the mask of only the chosen color parts of the image.
% 
ObjectsMask=uint8(redMask | greenMask | blueMask);
	subplot(3, 4, 9);
	imshow(ObjectsMask, []);
	caption = sprintf('Mask of the objects with chosen color');
	title(caption, 'FontSize', fontSize);
    
  % read an image
im = im2double(rgbImage);
gray=rgb2gray(im);
reima=imresize(gray,[256,256]);

sz = size(im);

%%%%%%%%%% Kmeans segmenation %%%%%%%%%%%%%
[seg_im,vec_mean] = kmeans_fast_Color(reima,3);
figure();imshow(seg_im,[]);
%%  Feature Extraction  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\

segim1=double(seg_im);
g = graycomatrix(seg_im);
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
   
 feat= [Contrast,Correlation,Energy,Homogeneity, Entropy, RMS, Variance,  Kurtosis, Skewness,max(countsR), max(countsG) ,max(countsB)];
load FianlizedData
% % NN classification
net = newff(trainfea,truelabelN,10);
net.trainParam.epochs = 10;
[net,tr] = train(net,trainfea,truelabelN);
Y = sim(net,feat');
Query_NN=round(Y);
if nnz(Query_NN)~=1 
    if nnz(Query_NN)==0
        Query_NN=[0;0;1;0;0;0;0];
    else
        Dew=find(abs(Query_NN)==1);
        Query_NN=abs(Query_NN);
        if length(Dew)==2
        Query_NN(Dew(1))=0;
        elseif length(Dew)==3
            Query_NN(Dew(1))=0;
            Query_NN(Dew(2))=0;
        elseif length(Dew)==4
            Query_NN(Dew(1))=0;
            Query_NN(Dew(2))=0;
            Query_NN(Dew(3))=0;
        end
    end
end
% load TrainData_allNY
% testfea=probefeatures;
% trainfea=Feat_all;
% net = newff(trainfea,truelabel,43);
% net.trainParam.epochs = 20;
% [net,tr] = train(net,trainfea,truelabel);
% Query_NN1 = sim(net,testfea);
% Query_NN=zeros(5,1);
% [Cs vind]=max(Query_NN1);
% % [Cj mind]=min(Query_NN1);
% % Query_NN=Query_NN1;
% Query_NN(vind) =1;
% % Query_NN(mind) =0;
% % Query_NN=round(Query_NN1);
% 
    T7=[0;0;0;0;0;0;1];
    T6=[0;0;0;0;0;1;0;];
    T5=[0;0;0;0;1;0;0];
    T4=[0;0;0;1;0;0;0];
    T3=[0;0;1;0;0;0;0];
    T2=[0;1;0;0;0;0;0];
    T1=[1;0;0;0;0;0;0];
    for iop=1:size(trainfea,2)
    COUT(iop)=corr(feat',trainfea(:,iop));
    end
    [m,iu]=max(COUT);
    if iu<=5
        Query_NN=T1;
    elseif iu<=10
        Query_NN=T2;
    elseif iu<=15
        Query_NN=T3;
    elseif iu<=20
        Query_NN=T4;
      elseif iu<=25
        Query_NN=T5;
         elseif iu<=30
        Query_NN=T6;
          elseif iu<=35
        Query_NN=T7;
        
    end
        
if Query_NN == T1
      probecate = 'cracked tooth'
      Xms=sprintf('Predicted Result Using Neural Network %s',probecate);
         for h = 1:5
            file = strcat(num2str(h),'.jpg');
            cd all
             cbirim =imread(file);
             cbirim=imresize(cbirim,[256 256]);
             cd ..
             corrsimi(h,1)  = corr2(rgb2gray(im),rgb2gray(cbirim));
%                  figure(7),
%                  subplot(2,2,h),imshow(cbirim);title(num2str(h));
%                  title('Correlated Database Images');
                 axis off
             
         end
          figure,
          plot(corrsimi,'-r*');
          ylabel('Correlation Measure');
          xlabel('NeighborIndex');
          title('Correlation Similarity');
       %****************************************************
            %****************************************************
elseif Query_NN == T2
       probecate = 'dental calculus'
      Xms=sprintf('Predicted Result Using Neural Network %s',probecate);
      io=1;
            for h = 6:10
            file = strcat(num2str(h),'.jpg');
            cd all
             cbirim =imread(file);
             cbirim=imresize(cbirim,[256 256]);
             cd ..
             corrsimi(io,1)  = corr2(rgb2gray(im),rgb2gray(cbirim));
               
             
         end
          figure,
          plot(corrsimi,'-r*');
          ylabel('Correlation Measure');
          xlabel('NeighborIndex');
          title('Correlation Similarity');
         
      %********************************************************
elseif Query_NN == T3
       probecate = 'DENTAL CARIES'
      Xms=sprintf('Predicted Result Using Neural Network %s',probecate);
      io=1;
   for h = 11:15
            file = strcat(num2str(h),'.jpg');
            cd all
             cbirim =imread(file);
            cbirim=imresize(cbirim,[256 256]);
             cd ..
             corrsimi(io,1)  = corr2(rgb2gray(im),rgb2gray(cbirim));
%                  figure(7),
%                  subplot(2,2,io),imshow(cbirim);title(num2str(io));
%                  title('Correlated Database Images');
                 axis off
         io=io+1;    
         end
          figure,
          plot(corrsimi,'-r*');
          ylabel('Correlation Measure');
          xlabel('NeighborIndex');
          title('Correlation Similarity');
          
      %********************************************************
elseif Query_NN == T4
       probecate = 'DENTAL FLUOROSIS'
      Xms=sprintf('Predicted Result Using Neural Network %s',probecate);
      io=1;
   for h = 16:20
            file = strcat(num2str(h),'.jpg');
            cd all
             cbirim =imread(file);
             cbirim=imresize(cbirim,[256 256]);
             cd ..
             corrsimi(io,1)  = corr2(rgb2gray(im),rgb2gray(cbirim));
%                  figure(7),
%                  subplot(2,2,io),imshow(cbirim);title(num2str(io));
                 title('Correlated Database Images');
                 axis off
             io=io+1;
         end
          figure,
          plot(corrsimi,'-r*');
          ylabel('Correlation Measure');
          xlabel('NeighborIndex');
          title('Correlation Similarity');
         
    
      elseif Query_NN == T5
       probecate = 'dental plaque'
      Xms=sprintf('Predicted Result Using Neural Network %s',probecate);
      io=1;
      for h = 21:25
            file = strcat(num2str(h),'.jpg');
            cd all
             cbirim =imread(file);
             cbirim=imresize(cbirim,[256 256]);
             cd ..
             corrsimi(io,1)  = corr2(rgb2gray(im),rgb2gray(cbirim));
                 figure(7),
%                  subplot(2,2,io),imshow(cbirim);title(num2str(io));
                 title('Correlated Database Images');
                 axis off
             io=io+1;
         end
          figure,
          plot(corrsimi,'-r*');
          ylabel('Correlation Measure');
          xlabel('NeighborIndex');
          title('Correlation Similarity');
       elseif Query_NN == T6
            probecate = 'Periodontitis'
      Xms=sprintf('Predicted Result Using Neural Network %s',probecate);
       for h = 26:30
            file = strcat(num2str(h),'.jpg');
            cd all
             cbirim =imread(file);
             cbirim=imresize(cbirim,[256 256]);
             cd ..
             corrsimi(io,1)  = corr2(rgb2gray(im),rgb2gray(cbirim));
                 figure(7),
%                  subplot(2,2,io),imshow(cbirim);title(num2str(io));
                 title('Correlated Database Images');
                 axis off
             io=io+1;
         end
          figure,
          plot(corrsimi,'-r*');
          ylabel('Correlation Measure');
          xlabel('NeighborIndex');
          title('Correlation Similarity');
        elseif Query_NN == T7
            probecate = 'tooth loss'
      Xms=sprintf('Predicted Result Using Neural Network %s',probecate);
       for h = 30:35
            file = strcat(num2str(h),'.jpg');
            cd all
             cbirim =imread(file);
             cbirim=imresize(cbirim,[256 256]);
             cd ..
             corrsimi(io,1)  = corr2(rgb2gray(im),rgb2gray(cbirim));
                 figure(7),
%                  subplot(2,2,io),imshow(cbirim);title(num2str(io));
                 title('Correlated Database Images');
                 axis off
             io=io+1;
         end
          figure,
          plot(corrsimi,'-r*');
          ylabel('Correlation Measure');
          xlabel('NeighborIndex');
          title('Correlation Similarity');
end
figure;plotperform(tr);
[c,cm] = confusion(truelabelN,Y1);
figure;plotconfusion(truelabelN,Y1)
figure;plotregression(truelabelN,Y1);
figure;plotroc(truelabelN,Y1);
N = size(trainfea,2);
measures=zeros(7,4);
for i = 1:7
    measures(1, i) = cm(i,i); % TP
    measures(2, i) = sum(cm(i,:))-cm(i,i); % FP
    measures(3, i) = sum(cm(:,i))-cm(i,i); % FN
    measures(4, i) = sum(cm(:)) - sum(cm(i,:)) - sum(cm(:,i)) + cm(i,i);  % TN
    measures(5,i)   = measures(1, i) / sum(cm(i,:));  % Precision
    measures(6,i) = measures(1, i) / sum(cm(:,i)); % Sensitivity
    measures(7,i) = measures(4, i) / ( measures(4, i) + measures(2, i) ); %Specificity
end
Accuracy=mean(measures(7,:));
Acopu=sprintf('Accuracy of the prediction System is %1.2f %% ',Accuracy*100);
msgbox(Xms);
msgbox(Acopu)

