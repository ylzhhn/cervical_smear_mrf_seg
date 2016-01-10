/**************************************************************************/
/*!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*/
/*                                                                        */
/*                        Lili  Zhao                                      */
/*  Copyright: Lili  Zhao, January, 2016                                   */
/*                                                                        */
/*------------------------------------------------------------------------*/
/*                                                                        */
/* Function:                                                              */
/*   Header file for preprocess.cpp                                       */
/*                                                                        */
/*------------------------------------------------------------------------*/
/*!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*/
/**************************************************************************/

#pragma once 

#include "utility.h"

class Preprocess
{
public:
	Preprocess();
	~Preprocess();
	void SetInData(string imgpath);
	cv::Mat get_bi_img();
	cv::Mat get_nl_img();
	cv::Mat get_gray_img();
	cv::Mat get_CIE_img();
private:
	
	cv::Mat Img; //original image

	cv::Mat bi_filter_img;//bilateral filter image
	cv::Mat nl_filter_img;//non local mean filter image
	cv::Mat gray_img;//gray result image
	cv::Mat CIE_img;//CIE color space image

};