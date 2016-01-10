/**************************************************************************/
/*!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*/
/*                                                                        */
/*                         Lili Zhao                                      */
/*  Copyright: Lili Zhao, January, 2016                                   */
/*                                                                        */
/*------------------------------------------------------------------------*/
/*                                                                        */
/* Function:                                                              */
/*    Header file for slic_feat_extract.cpp                               */
/*                                                                        */
/*-------------------------------------------------------------------------
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*/
/**************************************************************************/

#pragma once

#include "utility.h"

class Slic_feat_extract
{
public:
	Slic_feat_extract();
	~Slic_feat_extract();
	void SetInData(cv::Mat& nl_img, string& gt_path, arma::mat& slic_m, cv::Mat& CIE_img);
	void FeatureExtract();
	arma::mat get_feat_m();
	arma::mat get_gt_m();
private:
	mat SuperpixelLabels;
	/*string FeatSaveName;*/
	cv::Mat FilteredImg;
	cv::Mat GroundTruthImg;
	cv::Mat ImageLab;
	arma::mat feat_m;
	arma::mat gt_m;
	int GetSuperPixelLabel(uvec &, int &);
	float CalculteRatio(mat &, uvec &, int &);
	void GetSuperpixelMoment(uvec & indice, int, int &);

};