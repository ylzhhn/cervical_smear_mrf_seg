#include "FixEdgeGapSu.h"

SearchEdgeSu::SearchEdgeSu()
{}
SearchEdgeSu::~SearchEdgeSu()
{}

void SearchEdgeSu::SetInData(cv::Mat kmeanSeg)
{
	kmeans_Mat = kmeanSeg; 
}

void SearchEdgeSu::DetectEdge()
{
	DetecSobel();
	CandyDetetEdge();
}

void SearchEdgeSu::GetFixedEdgeSuSet()
{
	int h = contoursInv.rows;
	int w = contoursInv.cols;
	cout << w << "*" << h << endl;
	edge_mat.zeros(w, h);
	IplImage* img = NULL;
	img = &(IplImage(contoursInv));
	for (int i = 0; i < w; i++)
	{
		for (int j = 0; j < h; j++)
		{
			CvScalar p;
			p = cvGet2D(img, j, i);
			edge_mat(i, j) = p.val[0];
		}
	}

}

void SearchEdgeSu::CandyDetetEdge()
{
	cv::Mat test = sobelImage;
	// Apply Canny algorithm
	cv::Mat contours;
	cv::Canny(test, contours, 20, 350);
	//cv::Mat contoursInv; //contoursInv需要带回到整个class中使用
	cv::threshold(contours, contoursInv, 128, 255, cv::THRESH_BINARY_INV);

	cv::imwrite("edge.bmp", contoursInv);
	//cv::waitKey();

}

void SearchEdgeSu::DetecSobel()
{

	cv::Mat sobelX;
	cv::Sobel(kmeans_Mat, sobelX, CV_8U, 1, 0, 3, 0.4, 128);

	// Compute Sobel Y derivative
	cv::Mat sobelY;
	cv::Sobel(kmeans_Mat, sobelY, CV_8U, 0, 1, 3, 0.4, 128);

	// Compute norm of Sobel
	cv::Sobel(kmeans_Mat, sobelX, CV_16S, 1, 0);
	cv::Sobel(kmeans_Mat, sobelY, CV_16S, 0, 1);
	cv::Mat sobel;
	//compute the L1 norm
	sobel = abs(sobelX) + abs(sobelY);

	double sobmin, sobmax;
	cv::minMaxLoc(sobel, &sobmin, &sobmax);

	sobel.convertTo(sobelImage, CV_8U, -255. / sobmax, 255);
	
}

arma::mat SearchEdgeSu::GetGapEdgeSuSet()
{
	//需要先画出gapEdgeImg
	DrawEdgeGapPoints();//得到gapEdgeImg

	int h = gapEdgeImg.rows;
	int w = gapEdgeImg.cols;

	
	gapEdge_mat.zeros(w, h);

	IplImage* img = NULL;
	img = &(IplImage(gapEdgeImg));
	for (int i = 0; i < w; i++)
	{
		for (int j = 0; j < h; j++)
		{
			CvScalar p;
			p = cvGet2D(img, j, i);
			gapEdge_mat(i, j) = p.val[0];
		}
	}
	return gapEdge_mat;
}

void SearchEdgeSu::DrawEdgeGapPoints()
{
	GetFixedEdgeSuSet();//需要先得到edge_mat

	int w = edge_mat.n_cols;
	int h = edge_mat.n_rows;
	gapEdgeImg = contoursInv;

	for (int i = 0; i < h; i++)
	{
		for (int j = 0; j < w; j++)
		{
			if (edge_mat(i, j) == 0)
			{
				//cout << i << "," << j << endl;
				int x1, y1, x2, y2;
				x1 = i - 7, y1 = j - 7;
				x2 = i + 7, y2 = j + 7;
				if (x1 < 0)
					x1 = 0;
				if (y1 < 0)
					y1 = 0;
				if (x2 > h)
					x2 = h;
				if (y2 > w)
					y2 = w;

				Point pt1(x1, y1), pt2(x2, y2);
				const Scalar color = Scalar(0, 0, 0);
				rectangle(gapEdgeImg, pt1, pt2,
					color, CV_FILLED, 8, 0);//矩形内填充为color的颜色

			}
		}
	}
	imwrite("gap.bmp", gapEdgeImg);
}

void SearchEdgeSu::SaveSuSet(arma::mat flag_mat,string SaveName)
{
	arma::mat suFlag = flag_mat;
	ofstream rst(SaveName);
	for (int i = 0; i < suFlag.n_rows; i++)
	{
		for (int j = 0; j < suFlag.n_cols; j++)
		{
			rst << suFlag(i, j) << " ";
		}

		rst << endl;
	}
	rst.close();
}

arma::mat SearchEdgeSu::SearchPointMatch(
	string FeatFile, arma::mat su_mat/*, arma::mat search_mat*/)
{
	arma::mat EdgeFlagL;
	int w = gapEdge_mat.n_cols;
	int h = gapEdge_mat.n_rows;

	vector<vector<int>> suFeat = GetDouVec(FeatFile);
	EdgeFlagL.zeros(suFeat.size(), 2);
	EdgeFlagL = initEFL(suFeat, EdgeFlagL);

	for (int i = 0; i < h; i++)
	{
		for (int j = 0; j < w; j++)
		{
			if (gapEdge_mat(i, j) == 0)
			{
				int su_orglabel_no = su_mat(i, j);//被选中超像素的原始slic标签
				for (int k = 0; k < suFeat.size(); k++)
				{
					vector<int> featV = suFeat[k];
					int su_feat_orglno = featV[1];//原始slic标签
					if (su_feat_orglno == su_orglabel_no)//匹配：“位置+标签”匹配成功
					{
						EdgeFlagL(k, 1) = 1;//顺序slic标号“k”置1
					}
				}
			}
		}
	}

	return EdgeFlagL;
}

arma::mat SearchEdgeSu::initEFL(vector<vector<int>> suFeat, arma::mat& i)
{
	arma::mat rst = i;
	for (int k = 0; k < suFeat.size(); k++)
	{
		vector<int> featV = suFeat[k];
		int su_no = featV[0];
		rst(k, 0) = su_no;
		rst(k, 1) = 0;
	}
	return rst;
}

vector<vector<int>> SearchEdgeSu::GetDouVec(string file)
{
	vector<vector<int>> rst;
	ifstream In(file);
	string line;
	while (getline(In, line))
	{
		vector<int> NumList;
		NumList = CutLine2Num(line);
		rst.push_back(NumList);
		NumList.erase(NumList.begin(), NumList.begin() + NumList.size());
	}

	In.close();
	return rst;
}

vector<int> SearchEdgeSu::CutLine2Num(string line)
{
	vector<int> NumList;
	size_t found = line.find_first_of(" ");
	size_t Numbegin = 0;
	size_t Numend = found;
	//按空格切分出每个数字
	while (found != std::string::npos)
	{
		string curIndex = line.substr(Numbegin, Numend - Numbegin);
		int n = atof(curIndex.c_str());
		NumList.push_back(n);
		//获取下一个数字切分位置
		Numbegin = Numend + 1;
		found = line.find_first_of(" ", found + 1);
		Numend = found;
	}
	return NumList;
}