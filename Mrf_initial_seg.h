/**************************************************************************/
/*!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*/
/*                                                                        */
/*                        Lili  Zhao                                      */
/*  Copyright: Lili  Zhao, January, 2016                                   */
/*                                                                        */
/*------------------------------------------------------------------------*/
/*                                                                        */
/* Function:                                                              */
/*   Header file for Mrf_initial_seg.cpp                                  */
/*                                                                        */
/*------------------------------------------------------------------------*/
/*!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*/
/**************************************************************************/

#pragma once 

#include "utility.h"



class Init_kmeans
{
public:
	Init_kmeans();
	~Init_kmeans();
	void SetInData(arma::mat& slic_feat, arma::mat& slic_mat);
	void KmeansSeg();
	arma::mat get_kmeans_matrix();
	cv::Mat get_kmeans_img();
	arma::mat get_kmeans_label_list();
private:
	/*----------------------input data-------------------------*/
	arma::mat slic_f, slic_m;

	/*---------------------output data-------------------------*/
	arma::mat kmeans_m, kmeans_label_list;
	cv::Mat kmeans_img;

	/*------------------kmeans parameters----------------------*/
	int clusterNum;
	int Numkmeans;
	TermCriteria termC;
	int InitalMethod;
	/*---------------------------------------------------------*/
	/*---------------------kmeans data-------------------------*/
	/*------sample matrix£¬sample labels£¬center of cluster----*/
	/*---------------------------------------------------------*/
	cv::Mat kmeansData, kmeansLabel, clusterCenter;

	/*------------assistent variables and functions------------*/
	arma::mat feature_m, old_label_list, new_label_list;
	void Mat2mat_kmeans();
	void FindLabelMap(map<float, int>&, map<float, float>);
	void ReSortLabel(mat&, cv::Mat&);
	void findNewLabelMap(int minI, int maxI, map<int, int> & tmp);
	void CreateNewLabelSeq(map<int, int> old2newMap, vector<int>& newLabelSeq);
	void Savemat2Img(cv::Mat kmeansImg, mat& rst);
	void mat2Mat(arma::mat&);
};