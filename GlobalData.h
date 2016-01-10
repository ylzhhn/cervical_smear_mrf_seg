/**************************************************************************/
/*!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*/
/*                                                                        */
/*                         Lili Zhao                                      */
/*  Copyright: Lili Zhao, January, 2016                                    */
/*                                                                        */
/*------------------------------------------------------------------------*/
/*                                                                        */
/* Function:                                                              */
/*   Storage                                                              */
/*           input data: original image path, ground truth image path     */
/*                        org_img, gt_img, img_w, img_h                   */
/*------------------------------------------------------------------------*/
/*           preprocess_parameters:                                       */
/*------------------------------------------------------------------------*/
/*           preprocess_out_data:                                         */
/*                 nl_filter_img, gray_img, bi_filter_img                 */
/*------------------------------------------------------------------------*/
/*           slic_parameters: slic_nr, slic_nc                            */
/*------------------------------------------------------------------------*/
/*           slic_out_data:                                               */
/*                 slic_img,slic_neighobr,slic_matrix                     */
/*------------------------------------------------------------------------*/
/*           slic_feature_data:                                           */
/*                 slic_feature, gt_label                                 */
/*                slic_feature format: order_No, slic_No, feat1~feat13    */
/*------------------------------------------------------------------------*/
/*           kmeans++_parameters:                                         */
/*------------------------------------------------------------------------*/
/*           kmeans++_out_data:                                           */
/*                 kmeans_matrix, kmeans_img,kmeans_label_list            */
/*------------------------------------------------------------------------*/
/*           mrf_parameters:                                              */
/*------------------------------------------------------------------------*/
/*           mrf_out_data:                                                */
/*                 mrf_matrix,mrf_img,mrf_label_list                      */
/*------------------------------------------------------------------------*/
/*                                                                        */
/*-------------------------------------------------------------------------
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*/
/**************************************************************************/

#pragma once 

#include "utility.h"

/*---------------------------------------------------------*/
/* input data: original image path, ground truth image path*/
/*             org_img, gt_img, img_w, img_h               */
/*---------------------------------------------------------*/
string org_path, gt_path;
cv::Mat org_img, gt_img;

int img_w, img_h;

/*---------------------------------------------------------*/
/* preprocess_parameters:                                  */
/*---------------------------------------------------------*/


/*---------------------------------------------------------*/
/* preprocess_out_data:                                    */
/*         nl_filter_img, gray_img, bi_filter_img,CIE_img  */
/*---------------------------------------------------------*/
cv::Mat nl_filter_img, gray_img, bi_filter_img, CIE_img;

/*---------------------------------------------------------*/
/* slic_parameters: slic_nr, slic_nc                       */
/*---------------------------------------------------------*/
int slic_nr = 4000, slic_nc = 25;

/*---------------------------------------------------------*/
/* slic_out_data: slic_img,slic_neighobr,slic_matrix       */
/*---------------------------------------------------------*/
cv::Mat slic_img;
arma::mat slic_matrix;
vector<vector<int>> slic_neighbor;

/*---------------------------------------------------------*/
/*   slic_feature_data:                                    */
/*         slic_feature, slic_gt_label                     */
/*     slic_feature format: order_No, slic_No, feat1~feat13*/
/*---------------------------------------------------------*/
arma::mat slic_feature, slic_gt_label;

/*---------------------------------------------------------*/
/*           kmeans++_parameters:                          */
/*---------------------------------------------------------*/


/*---------------------------------------------------------*/
/* kmeans++_out_data:                                      */
/*     kmeans++_matrix, kmeans++_img,kmeans++_label_list   */
/*---------------------------------------------------------*/
arma::mat kmeans_matrix;
cv::Mat kmeans_img;
arma::mat kmeans_label_list;

/*---------------------------------------------------------*/
/*  mrf_out_data:                                          */
/*        mrf_matrix,mrf_img,mrf_label_list                */
/*---------------------------------------------------------*/
double t(0.06);//能量改变终止阈值
int maxIter(40);//最大迭代次数

int classNum = 3;//标签数目
double beta = 0.5;

arma::mat mrf_matrix;
cv::Mat mrf_img;
arma::mat mrf_label_list;

string  mat2txt(arma::mat slic_feat)
{
	string rst = ".//tmp.txt";
	ofstream save_rst(rst);
	int w = slic_feat.n_rows;
	int h = slic_feat.n_cols;
	for (int i = 0; i < w; i++)
	{
		for (int j = 0; j < h; j++)
		{
			save_rst << slic_feat(i, j) << " ";
		}
		save_rst << endl;
	}
	save_rst.close();
	return rst;
}

void SaveMrfLabelList(arma::mat list, string savefile)
{
	ofstream out(savefile);
	int r = list.n_rows;
	int c = list.n_cols;
	for (int i = 0; i < r; i++)
	{
		for (int j = 0; j < c; j++)
			out << list(i, j) << " ";
		out << endl;
	}
	out.close();
}

string VecM2string(vector<vector<int>> vecM)
{
	string rst = ".//tmpVec2.txt";
	ofstream save_rst(rst);
	int r = vecM.size();
	for (int i = 0; i < r; i++)
	{
		int c = vecM[i].size();
		for (int j = 0; j < c; j++)
		{
			save_rst << vecM[i][j] << " ";
		}
		save_rst << endl;
	}
	save_rst.close();
	return rst;
}