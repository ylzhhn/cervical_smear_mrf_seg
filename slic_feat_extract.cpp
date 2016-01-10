/**************************************************************************/
/*!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*/
/*                                                                        */
/*                        Lili  Zhao                                      */
/*  Copyright: Lili  Zhao, August, 2015                                   */
/*                                                                        */
/*------------------------------------------------------------------------*/
/*                                                                        */
/* Function:                                                              */
/*   extract SLIC superpixel features and ground truth labels             */
/*                                                                        */
/*------------------------------------------------------------------------*/
/*!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*/
/**************************************************************************/

#pragma once 

#include "slic_feat_extract.h"

Slic_feat_extract::Slic_feat_extract(){}
Slic_feat_extract::~Slic_feat_extract(){}

void Slic_feat_extract::SetInData(cv::Mat& nl_img, string& gt_path,arma::mat& slic_m, cv::Mat& CIE_img)
{
	FilteredImg = nl_img;
	if (!FilteredImg.data)
	{
		cout << "Can't open the FilteredImg image . Check your configure." << endl;
		exit(0);
	};
	GroundTruthImg = imread(gt_path, 1);
	if (!GroundTruthImg.data)
	{
		cout << "Can't open the GroundTruthImg image . Check your configure." << endl;
		exit(0);
	};
	SuperpixelLabels = slic_m;
	ImageLab = CIE_img;
}

void Slic_feat_extract::FeatureExtract()
{
	mat suPixels = unique(SuperpixelLabels);
	int NumofsuperPixels = suPixels.size();//the number of SLIC superpixels
	int ImgCol = FilteredImg.cols;
	MatIterator_<Vec3b> it = FilteredImg.begin<Vec3b>();
	MatIterator_<Vec3b> LABit = ImageLab.begin<Vec3b>();
	
	feat_m.zeros(NumofsuperPixels,15);
	gt_m.zeros(NumofsuperPixels,1);
	for (int i = 0; i < NumofsuperPixels; i++)
	{
		feat_m(i, 0) = i;
		int orgsuperpixelL = suPixels.at(i, 0);// original labels of SLIC superpixels
		feat_m(i, 1) = orgsuperpixelL;
		uvec indice = find(SuperpixelLabels == orgsuperpixelL);// find out the indic of org superpixelL
		int NumOfPixels = indice.size();//the number of indice

		mat  mapFilterValue = zeros<mat>(NumOfPixels, 3);
		mat maplabValue = zeros<mat>(NumOfPixels, 3);
		for (int j = 0; j < NumOfPixels; j++)
		{
			mapFilterValue.at(j, 0) = double((*(it + indice[j]))[0]);//B
			mapFilterValue.at(j, 1) = double((*(it + indice[j]))[1]);//G
			mapFilterValue.at(j, 2) = double((*(it + indice[j]))[2]);//R

			maplabValue.at(j, 0) = double((*(LABit + indice[j]))[0]);//L
			maplabValue.at(j, 1) = double((*(LABit + indice[j]))[1]);//a
			maplabValue.at(j, 2) = double((*(LABit + indice[j]))[2]);//b

		}

		vec mapVec = zeros<vec>(NumOfPixels);

		//*****RGB color*****
		mapVec = mapFilterValue.col(0);
		if (mapVec.size() == 0)
		{
			feat_m(i, 2) = 0;
			feat_m(i, 3) = 0;
		}
		else{
			feat_m(i, 2) = mean(mapVec);
			feat_m(i, 3) = median(mapVec);
		}

		mapVec = mapFilterValue.col(1);
		if (mapVec.size() == 0)
		{
			feat_m(i, 4) = 0;
			feat_m(i, 5) = 0;
		}
		else{
			feat_m(i, 4) = mean(mapVec);
			feat_m(i, 5) = median(mapVec);
		}
		mapVec = mapFilterValue.col(2);
		if (mapVec.size() == 0)
		{
			feat_m(i, 6) = 0;
			feat_m(i, 7) = 0;
		}
		else{
			feat_m(i, 6) = mean(mapVec);
			feat_m(i, 7) = median(mapVec);
		}

		//*****CIELAB color*****
		mapVec = maplabValue.col(0);
		if (mapVec.size() == 0)
		{
			feat_m(i, 8) = 0;
			feat_m(i, 9) = 0;
		}
		else{
			feat_m(i, 8) = mean(mapVec);
			feat_m(i, 9) = median(mapVec);
		}
		
		mapVec = maplabValue.col(1);
		if (mapVec.size() == 0)
		{
			feat_m(i, 10) = 0;
			feat_m(i, 11) = 0;
		}
		else{
			feat_m(i, 10) = mean(mapVec);
			feat_m(i, 11) = median(mapVec);
		}
		mapVec = maplabValue.col(2);
		if (mapVec.size() == 0)
		{
			feat_m(i, 12) = 0;
			feat_m(i, 13) = 0;
		}
		else{
			feat_m(i, 12) = mean(mapVec);
			feat_m(i, 13) = median(mapVec);
		}
		if (NumOfPixels == 0){
			cout << "key points" << endl;
		}
		float ratio = CalculteRatio(suPixels, indice, NumOfPixels);
		feat_m(i, 14) = ratio;

		//calcuate the label of the current superpixel 
		int label = GetSuperPixelLabel(indice, NumOfPixels);
		gt_m(i, 0) = label;
	}
}


int Slic_feat_extract::GetSuperPixelLabel(uvec & indice, int & numofPixels)
{
	// assume that the initial superpixel label is "0"
	int label(0);
	MatIterator_<Vec3b> itGT = GroundTruthImg.begin<Vec3b>();
	int nucleusPnum(0), cytoplasmPnum(0), backGPnum(0), grayPnum(0);
	for (int j = 0; j < numofPixels; j++)
	{

		if (double((*(itGT + indice[j]))[0]) == 128//B
			&& double((*(itGT + indice[j]))[1]) == 0//G
			&& double((*(itGT + indice[j]))[2]) == 0)//R
		{
			cytoplasmPnum++;// cytoplasm 
		}
		if (double((*(itGT + indice[j]))[0]) == 255
			&& double((*(itGT + indice[j]))[1]) == 0
			&& double((*(itGT + indice[j]))[2]) == 0)
		{
			nucleusPnum++;//nuclei
		}
		if (double((*(itGT + indice[j]))[0]) == 0
			&& double((*(itGT + indice[j]))[1]) == 0
			&& double((*(itGT + indice[j]))[2]) == 255)
		{
			backGPnum++;//background
		}
		if (double((*(itGT + indice[j]))[0]) == 128
			&& double((*(itGT + indice[j]))[1]) == 128
			&& double((*(itGT + indice[j]))[2]) == 128)
		{
			grayPnum++;//gray region as background
		}
	}
	vector<int> NumList;
	// "1" presents neclei, "2" presents cytoplasm£¬"3" for background£¬"4" for gray region
	NumList.push_back(nucleusPnum);//1
	NumList.push_back(cytoplasmPnum);//2
	NumList.push_back(backGPnum);//3
	NumList.push_back(grayPnum);//4
	int maxNum(0);
	for (int i = 0; i != NumList.size(); i++)
	{
		if (maxNum < NumList[i])
		{
			maxNum = NumList[i];
			label = i + 1;
		}
	}
	if (label == 4 || label == 0) label = 3;

	return label;
}

float Slic_feat_extract::CalculteRatio(mat & suPixel, uvec & indice, int & numofPixels)
{
	float ratio(0);
	float PixelNum = numofPixels;
	int WeightOfRect(1), HeightOfRect(1);
	int minX(10000), minY(10000), maxX(-1), maxY(-1);
	int ImgCol = FilteredImg.cols;//width of image
	int ImgRow = FilteredImg.rows;//height of image
	for (int i = 0; i < PixelNum; i++)
	{
		int y = (0);
		if (indice(i) != 0)
		{
			y = (indice(i) % ImgCol);
		}
		int x = (indice(i) - y) / ImgCol;
		if (x < minX) minX = x;
		if (x > maxX) maxX = x;
		if (y < minY) minY = y;
		if (y > maxY) maxY = y;
	}
	WeightOfRect = maxX - minX + 1;
	HeightOfRect = maxY - minY + 1;
	ratio = float(PixelNum / (WeightOfRect* HeightOfRect));
	if (ratio <= 0) {
		cout << "ratio error !";
		exit(0);
	}
	return ratio;
}

arma::mat Slic_feat_extract::get_feat_m()
{
	return feat_m;
}
arma::mat Slic_feat_extract::get_gt_m()
{
	return gt_m;
}