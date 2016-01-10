/**************************************************************************/
/*!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*/
/*                                                                        */
/*                        Lili  Zhao                                      */
/*  Copyright: Lili  Zhao, August, 2015                                   */
/*                                                                        */
/*------------------------------------------------------------------------*/
/*                                                                        */
/* Function:                                                              */
/*   superpixe-based initial segmentation and label-sort algorithm        */
/*                                                                        */
/*------------------------------------------------------------------------*/
/*!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*/
/**************************************************************************/

#include "Mrf_initial_seg.h"

Init_kmeans::Init_kmeans(){}
Init_kmeans::~Init_kmeans(){}

void Init_kmeans::SetInData(arma::mat& slic_feat, arma::mat& slic_mat)
{
	slic_f = slic_feat;
	slic_m = slic_mat;
	/*---------------------kmeans data-------------------------*/
	feature_m = slic_feat.cols(2, slic_f.n_cols-1);// bug check
	mat2Mat(feature_m);
}

void Init_kmeans::KmeansSeg()
{
	//kmeans parameters
	clusterNum = 3;// number of class
	termC = TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 100, 1.0);// stop condition
	Numkmeans = 100; //cluster 100 times
	InitalMethod = 2;//KMEANS_PP_CENTERS, Uses k-Means++ algorithm for initialization

	kmeans(kmeansData, clusterNum, kmeansLabel,
		termC,
		Numkmeans, InitalMethod, clusterCenter);
}
/*---------------------------------------------------------*/
/*           need after get_kmeans_img()                   */
/*---------------------------------------------------------*/
arma::mat Init_kmeans::get_kmeans_matrix()
{
	return kmeans_m;
}
/*---------------------------------------------------------*/
/*        need after get_kmeans_label_list()               */
/*---------------------------------------------------------*/
cv::Mat Init_kmeans::get_kmeans_img()
{
	Savemat2Img(kmeans_img, new_label_list);
	return kmeans_img;
}
/*---------------------------------------------------------*/
/* initial segmentation and label-sort algorithm           */
/*---------------------------------------------------------*/
arma::mat Init_kmeans::get_kmeans_label_list()
{
	Mat2mat_kmeans();
	new_label_list = old_label_list;
	ReSortLabel(new_label_list, clusterCenter);//find out new_label_list
	return new_label_list;
}

void Init_kmeans::Mat2mat_kmeans()
{
	int width = kmeansLabel.cols;
	int height = kmeansLabel.rows;
	map<float, float> label;
	for (int i = 0; i < height; i++)
	{
		label.insert(pair<float, float>(kmeansLabel.at<float>(i, 0), kmeansLabel.at<float>(i, 0)));
	}
	//mapLabel change float label to int label
	map<float, int> mapLabel;
	FindLabelMap(mapLabel, label);
	old_label_list.zeros( slic_f.n_rows, 2);
	old_label_list.insert_cols(0, slic_f.col(1));
	int index(0);
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			//find out int labels
			map<float, int>::iterator curLabel = mapLabel.find(kmeansLabel.at<float>(i, j));
			old_label_list(index++, 1) = double(curLabel->second);
		}
	}

}

void Init_kmeans::FindLabelMap(map<float, int>& mapLabel, map<float, float> label)
{
	for (map<float, float>::iterator it = label.begin(); it != label.end(); it++)
	{
		float kl = it->first;
		mapLabel.insert(pair<float, int>(kl, 0));
	}
	int c = 1;
	for (map<float, int>::iterator it = mapLabel.begin(); it != mapLabel.end(); it++)
	{
		mapLabel[it->first] = c;
		c++;
	}
}

void Init_kmeans::ReSortLabel(mat& New_matkmeans, cv::Mat& clusterCenter)
{
	float max(-100000000000), min(10000000000);
	int maxindex, minindex;
	for (int i = 0; i < clusterNum; i++)
	{
		if (clusterCenter.at<float>(i, 0) > max)
		{
			max = clusterCenter.at<float>(i, 0);
			maxindex = i;
		}

		if (clusterCenter.at<float>(i, 0) < min)
		{
			min = clusterCenter.at<float>(i, 0);
			minindex = i;
		}
	}
	map<int, int> old2newMap;
	findNewLabelMap(minindex, maxindex, old2newMap);//<old_label£¬new_label> map relationship
	vector<int> newLabelSeq;
	CreateNewLabelSeq(old2newMap, newLabelSeq);

	for (int i = 0; i < old_label_list.n_rows; i++)
	{
		New_matkmeans(i, 1) = newLabelSeq[i];
	}

}

void Init_kmeans::findNewLabelMap(int minI, int maxI, map<int, int> & tmp)
{
	//find out the cytoplasm's label
	int oldmedian(2);// initially suppose the cytoplasm's label is "2"
	set<int> LabelSet;//label set
	for (int i = 1; i <= 3; ++i)
	{
		LabelSet.insert(i);
	}
	for (int i = 1; i <= 3; ++i)
	{
		if (minI + 1 == i)
			LabelSet.erase(i);
		if (maxI + 1 == i)
			LabelSet.erase(i);
	}
	std::set<int>::iterator it = LabelSet.begin();
	oldmedian = (*it);
	tmp.insert(pair<int, int>(minI + 1, 1));//<old_label£¬new_label>
	tmp.insert(pair<int, int>(maxI + 1, 3));
	tmp.insert(pair<int, int>(oldmedian, 2));
}

void Init_kmeans::CreateNewLabelSeq(map<int, int> old2newMap, vector<int>& newLabelSeq)
{
	for (int i = 0; i < old_label_list.n_rows; i++)
	{
		int old_label = int(old_label_list(i, 1));
		for (std::map<int, int>::iterator it = old2newMap.begin(); it != old2newMap.end(); ++it)
		{
			if (old_label == it->first)
			{
				newLabelSeq.push_back(int(it->second));
				break;
			}
		}
	}
}

void Init_kmeans::Savemat2Img(cv::Mat kmeansImg, mat& label_list)
{
	int H = slic_m.n_rows, W = slic_m.n_cols;//H for width, W for heigh
	kmeans_m.zeros(H, W);
	//creat binary image of the result
	IplImage *bin = cvCreateImage(cvSize(H, W), IPL_DEPTH_8U, 1);

	int val = 0;
	//step according to the number of class
	float step = 255 / (float(clusterNum));

	//set pixel values of different regions
	for (int i = 0; i<H; i++)
	{
		for (int j = 0; j<W; j++)
		{
			for (int k = 0; k < label_list.n_rows; k++)
			{
				if (label_list(k, 0) == slic_m(i, j))
				{
					val = label_list(k, 1);//val is the result label
					kmeans_m(i, j) = val;
					break;
				}
			}
			CvScalar s;
			s.val[0] = 255 - val*step;//set different regions as different values
			cvSet2D(bin, j, i, s);
		}
	}

}

void Init_kmeans::mat2Mat(arma::mat& featureSet)
{
	mat NorSamples = normalise(featureSet, 2);
	int numRows = featureSet.n_rows;
	int numCols = featureSet.n_cols;
	kmeansData.create(numRows, numCols, CV_32F);
	for (int i = 0; i < numRows; i++)
	{
		for (int j = 0; j < numCols; j++)
		{
			kmeansData.at<float>(i, j) = NorSamples(i, j);
		}
	}
}
