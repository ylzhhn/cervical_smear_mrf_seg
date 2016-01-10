#pragma once 

#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <math.h>

#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>

#include <armadillo>

using namespace arma;
using namespace std;
using namespace cv;

class MrfSolve
{
public:
	MrfSolve();
	~MrfSolve();

	void SetInData(string SuDataFile,
		string SuFlagFile, string InitLabelFile,
		string GraphLinkFile,
		double T, double Beta,
		int MaxIter, int ClassNum);//input function
	void MakeSolve();//solution main function
	arma::mat GetRst();//output function
private:
	//data variables
	//string suDataFile, suFlagFile, initLabelFile ;
	arma::mat suData_mat, suFlag_mat, initLabel_list;
	arma::mat suGtList;

	//string graphLinkFile;
	vector<vector<int>> graphLink;

	//parameters
	double t, beta;
	int maxIter;
	int classNum;

	//output variables
	arma::mat mrfLabel_list;


	//assistent varialbes & function
	int suPixelNum, Featdim;
	arma::mat Mu;//mean vector
	arma::cube Sigma;//cov matrix
	void Load2Vec(string graphLinkFile);
	vector<int> CutLine2Num(string line);
	vector<vector<int>> deleteNoneNeighbor();
	void CalculateMeanSigma(mat);

	double CalculateEnergy();
	double Singleton(int index, int label);
	double Doubleton(int index, int label);
	int  usingNoFindlabel(int slicNo);
	double CalculateLocalEnergy(int indice, int label);

	//added function
	map<int, int> InitLDict;
	void slicSeqNo_initNo();
};