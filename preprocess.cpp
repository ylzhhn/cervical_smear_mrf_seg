/**************************************************************************/
/*!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*/
/*                                                                        */
/*                         Lili Zhao                                 */
/*  Copyright: Lili Zhao, January, 2016              */
/*                                                                        */
/*------------------------------------------------------------------------*/
/*                                                                                                     */
/* Function:                                                                                     */
/*   Subroutines for original image preprocessing                        */
/*        preprocessing :   non-local filter,bilateralfilter,                  */
/*                          ExtractOneChannel2grayImg                            */
/*-------------------------------------------------------------------------
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*/
/**************************************************************************/

#include "preprocess.h"

Preprocess::Preprocess(){}
Preprocess::~Preprocess(){}
void Preprocess::SetInData(string imgpath)
{
	Img = cv::imread(imgpath, CV_LOAD_IMAGE_COLOR);
	if (!Img.data)
	{
		cout << "can't load original image data ! Please check configures." << endl;
		system("pause");
	}
}
cv::Mat Preprocess::get_bi_img()
{
	int MAX_KERNEL_LENGTH = 31;
	bi_filter_img = Img.clone();

	for (int i = 1; i < MAX_KERNEL_LENGTH; i = i + 2)
	{
		bilateralFilter(Img, bi_filter_img, i, i * 2, i / 2);
	}
	return bi_filter_img;
}
cv::Mat Preprocess::get_nl_img()
{
	float h = 3;
	float hColor = 3;
	int templateWindowSize = 7;
	int searchWindowSize = 21;

	nl_filter_img = Img.clone();
	fastNlMeansDenoisingColored(Img, nl_filter_img, h, hColor, templateWindowSize, searchWindowSize);

	return nl_filter_img;
}
/*---------------------------------------------------------*/
/* extract one channel as a gray image output         */
/*---------------------------------------------------------*/
cv::Mat Preprocess::get_gray_img()
{
	gray_img.create(Img.rows, Img.cols, CV_8U);
	MatIterator_<Vec3b> itGT = Img.begin<Vec3b>();
	Mat_<uchar>::iterator itout = gray_img.begin<uchar>();
	int nl = Img.rows;//the number of rows 
	int nc = Img.cols;//the number of cols
	for (int j = 0; j < nl; j++)
	{
		for (int i = 0; i < nc; i++)
		{
			int indice = j*nc + i;
			*(itout + indice) = (*(itGT + indice))[2];//extract one channel
		}
	}
	return gray_img;
}
cv::Mat Preprocess::get_CIE_img()
{
	CIE_img = Img.clone();
	cvtColor(Img, CIE_img, CV_BGR2Lab);
	return CIE_img;
}