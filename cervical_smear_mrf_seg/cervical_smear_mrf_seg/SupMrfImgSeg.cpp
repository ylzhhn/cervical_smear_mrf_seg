/**********************************************************************************/
/*!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*/
/*                                                                                                                                   */
/*                         Lili Zhao, Kuan Li, Mao Wang, Jianping Yin.                                    */
/*The copyright is reserved by Lili Zhao, Kuan Li, Mao Wang, Jianping Yin.              */
/*                               January, 2016                                                                              */
/*   If you have any questions, please do not hesitate to contact Lili Zhao.               */
/*      Email address :      yilinzhaohenan@126.com  (Lili Zhao)                                  */
/*   This project just is used for academic research.                                                   */
/*  If you use this codes in your work, you must cite:                                                 */
/* "Automatic Cytoplasm and Nuclei Segmentation for Color Cervical Smear Image 
using an efficient Gap-search MRF "                                                                           */
/*                                                                                                                                  */
/*   If the codes use for commercial application, you must contact the first author. */
/*!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*/

/*------------------------------------------------------------------------*/
/*                                                                                                     */
/* Main Function:                                                                            */
/*   Segmentation from cervical smear image                               */
/*         First:   denoising processing                                              */
/*         Second:   SLIC superpixel processing                                */
/*         Third:   superpixel-based MRF                                           */
/*                                                                                                     */
/*-------------------------------------------------------------------------

/**************************************************************************/

#include "GlobalData.h"
#include "utility.h"
#include "preprocess.h"
#include "slic.h"
#include "slic_feat_extract.h"
#include "Mrf_initial_seg.h"
#include "Mrf_solution.h"
#include "FixEdgeGapSu.h"

int main()
{
	org_path = "..//Data//images//samples//148495553-148495585-001.BMP";
	gt_path = "..//Data//images//samples//148495553-148495585-001-d.bmp";
	/*##################################################################*/
	/*     First:   denoising processing                                                                                              */
	/*##################################################################*/
	Preprocess prep;
	prep.SetInData( org_path );
	nl_filter_img = prep.get_nl_img();
	CIE_img = prep.get_CIE_img();
	/*##################################################################*/
	/*     Second:   SLIC superpixel processing                                                                                */
	/*##################################################################*/
	Slic slic;
	slic.save_segmente_image(nl_filter_img, slic_nr, slic_nc);
	slic_img = slic.get_slic_img();
	slic_matrix = slic.get_slic_matrix();
	slic_neighbor = slic.get_slic_neighbor();
	/*---------------------------------------------------------*/
	/* extract each SLIC superpixel's features                 */
	/*---------------------------------------------------------*/
	Slic_feat_extract slic_feats;
	slic_feats.SetInData(nl_filter_img, gt_path, slic_matrix, CIE_img);
	slic_feats.FeatureExtract();
	slic_feature = slic_feats.get_feat_m();
	slic_gt_label = slic_feats.get_gt_m();

	/*##################################################################*/
	/*     Third:   superpixel-based MRF                                                                                           */
	/*##################################################################*/
	/*---------------------------------------------------------*/
	/* initial segmentation and label-sort algorithm      */
	/*---------------------------------------------------------*/
	Init_kmeans initSeg;
	initSeg.SetInData(slic_feature, slic_matrix);
	initSeg.KmeansSeg();
	kmeans_label_list = initSeg.get_kmeans_label_list();
	kmeans_img = initSeg.get_kmeans_img();

	/*---------------------------------------------------------*/
	/*      fix search gap                                                   */
	/*---------------------------------------------------------*/
	SearchEdgeSu suSet;
	suSet.SetInData(kmeans_img);
	suSet.DetectEdge();
	suSet.GetGapEdgeSuSet();
	arma::mat  su_mat = slic_matrix;
	
	string SampleData = mat2txt(slic_feature);
	arma::mat SelectGapSuSet = suSet.SearchPointMatch(SampleData, su_mat);
	arma::mat flag = SelectGapSuSet.col(1);
	uvec n = arma::find(flag == 1);
	cout << "The number of selected superpixel : " << n.n_rows << endl;

	string FlagFile = ".//su_gap_flag_seg.txt";
	suSet.SaveSuSet(SelectGapSuSet, FlagFile);


	/*---------------------------------------------------------*/
	/*     gap-search MRF                                                */
	/*---------------------------------------------------------*/
	string InitLabelsFile = mat2txt(kmeans_label_list);
	string NodeRelationFile = VecM2string(slic_neighbor);

	MrfSolve solver;
	solver.SetInData(SampleData,
		FlagFile, InitLabelsFile,
		NodeRelationFile,
		t, beta,
		maxIter, classNum);//input function
	arma::mat mrfL = solver.GetRst();

	system("pause");
	return 0;
}