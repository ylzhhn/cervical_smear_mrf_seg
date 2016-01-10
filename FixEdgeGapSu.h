#pragma once 

#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <math.h>

#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>

#include <armadillo>

#include "laplacianZC.h"

using namespace arma;
using namespace std;
using namespace cv;

class SearchEdgeSu
{
public:
	SearchEdgeSu();
	~SearchEdgeSu();
	void SetInData(cv::Mat kmeanSeg);
	void DetectEdge();	
	arma::mat GetGapEdgeSuSet();
	
	arma::mat SearchPointMatch(string FeatFile,
		arma::mat su_mat/*, arma::mat search_mat*/);

	void SaveSuSet(arma::mat , string SaveName);
private:
	cv::Mat kmeans_Mat;//input 
	//mid rst, contoursInv is "edge.bmp"
	cv::Mat sobelImage, /*contours, */contoursInv;

	arma::mat edge_mat;//±ßÔµÏñËØµã¾ØÕóarma::mat edge_mat;
	arma::mat gapEdge_mat;
	
	cv::Mat gapEdgeImg;

	
	void GetFixedEdgeSuSet();
	void DetecSobel();
	void CandyDetetEdge();
	void DrawEdgeGapPoints();

	
	arma::mat initEFL(vector<vector<int>> suFeat, arma::mat& i);
	vector<vector<int>> GetDouVec(string file);
	vector<int> CutLine2Num(string line);
};